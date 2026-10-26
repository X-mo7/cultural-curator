"""Small async Mistral boundary with bounded retries and payload-free telemetry."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import math
import os
from time import perf_counter
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel


Model = TypeVar("Model", bound=BaseModel)
Result = TypeVar("Result")
DEFAULT_MISTRAL_MODEL = "mistral-small-latest"


class ExternalCallError(RuntimeError):
    """A Mistral call failed after its permitted attempts."""


class InvalidModelResponse(ValueError):
    """The provider returned unusable or incomplete output."""


def validate_finish_reason(choice: Any) -> None:
    """Reject incomplete turns before using their text, parsed data or tool calls."""
    if choice.finish_reason in {"length", "model_length", "error"}:
        raise InvalidModelResponse(
            f"Mistral returned an incomplete response: {choice.finish_reason}"
        )


@dataclass(frozen=True)
class CallMetrics:
    operation: str
    model: str
    elapsed_seconds: float
    attempts: int
    status: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


MetricsObserver = Callable[[CallMetrics], None]


@dataclass(frozen=True)
class AsyncCallPolicy:
    """Timeout applies to each attempt; total attempts include the first call."""

    timeout_ms: int = 30_000
    max_attempts: int = 3
    retry_delay_seconds: float = 0.1

    def __post_init__(self) -> None:
        if type(self.timeout_ms) is not int or self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if not math.isfinite(self.retry_delay_seconds) or self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be finite and non-negative")


def _transient(error: Exception) -> bool:
    return isinstance(error, (
        TimeoutError, httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError,
    )) or getattr(
        error, "status_code", None
    ) in {408, 429, 500, 502, 503, 504}


async def call_async(
    call: Callable[[], Awaitable[Any]],
    *,
    operation: str,
    model: str,
    policy: AsyncCallPolicy,
    observer: MetricsObserver | None = None,
    validate: Callable[[Any], Result],
) -> Result:
    """Retry transient transport/status failures, never invalid model output.

    The SDK call must pass ``retries=None``: in Mistral SDK 2.x this disables
    its own retry loop, including when a client-level retry policy is set.
    Metrics describe the entire operation and the final response's usage;
    tokens consumed by failed attempts are unknown. Observers are best effort.
    """
    started = perf_counter()
    response = None
    status = "error"
    attempt = 0
    try:
        for attempt in range(1, policy.max_attempts + 1):
            try:
                async with asyncio.timeout(policy.timeout_ms / 1000):
                    response = await call()
                result = validate(response)
                status = "success"
                return result
            except (ValueError, TypeError, AttributeError, IndexError) as error:
                status = "invalid_response"
                raise InvalidModelResponse("Mistral returned an invalid response") from error
            except Exception as error:
                if not _transient(error) or attempt == policy.max_attempts:
                    raise ExternalCallError(
                        f"Mistral {operation} failed after {attempt} attempt(s)"
                    ) from error
                await asyncio.sleep(policy.retry_delay_seconds * 2 ** (attempt - 1))
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    finally:
        if observer is not None:
            usage = getattr(response, "usage", None)
            tokens = {}
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, name, None)
                tokens[name] = value if type(value) is int and value >= 0 else None
            metrics = CallMetrics(
                operation, model, perf_counter() - started, attempt, status, **tokens
            )
            try:
                observer(metrics)
            except Exception:
                pass  # Observability must not turn a successful operation into a failure. Metrics observations must not be blocking hence should not raise exceptions.


@asynccontextmanager
async def sdk_client(api_key: str | None, client: Any = None):
    """Injected clients belong to the caller; owned clients close after each call."""
    if client is not None:
        yield client
    else:
        from mistralai.client import Mistral

        async with Mistral(api_key=api_key) as owned:
            yield owned


class MistralChat:
    """Native async chat/structured-output calls without hidden SDK retries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client: Any = None,
        timeout_ms: int = 30_000,
        max_attempts: int = 3,
        observer: MetricsObserver | None = None,
    ) -> None:
        self.model = model or os.getenv("MISTRAL_MODEL", DEFAULT_MISTRAL_MODEL)
        if not self.model.strip():
            raise ValueError("chat model must not be blank")
        self.policy = AsyncCallPolicy(timeout_ms=timeout_ms, max_attempts=max_attempts)
        self.observer = observer
        self._client = client
        self._api_key = None
        if client is None:
            self._api_key = api_key or os.getenv("MISTRAL_API_KEY")
            if not self._api_key or not self._api_key.strip():
                raise RuntimeError("MISTRAL_API_KEY is required")

    async def _call(
        self, method: str, validate: Callable[[Any], Result], **kwargs: Any
    ) -> Result:
        async with sdk_client(self._api_key, self._client) as client:
            return await call_async(
                lambda: getattr(client.chat, method)(
                    model=self.model,
                    timeout_ms=self.policy.timeout_ms,
                    retries=None,
                    **kwargs,
                ),
                operation=f"chat.{method.removesuffix('_async')}",
                model=self.model,
                policy=self.policy,
                observer=self.observer,
                validate=validate,
            )

    async def parse(self, *, messages: list, response_format: type[Model]) -> Model :# this is the method we use to obtain structured responses (so client.chat.parse_async)
        def validate(response: Any) -> Model:
            choice = response.choices[0]
            validate_finish_reason(choice)
            parsed = choice.message.parsed
            if parsed is None:
                raise InvalidModelResponse("Mistral returned no parsed response")
            return response_format.model_validate(parsed)

        return await self._call(
            "parse_async", validate, messages=messages, response_format=response_format
        )

    async def complete(self, *, messages: list, tools: list) -> Any: # this is the method we use for function calling (so client.chat.complete_async)
        def validate(response: Any) -> Any:
            choice = response.choices[0]
            validate_finish_reason(choice)
            if choice.message is None:
                raise InvalidModelResponse("Mistral returned no message")
            return response

        return await self._call(
            "complete_async", validate, messages=messages, tools=tools
        )
