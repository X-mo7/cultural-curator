"""Core domain records for Cultural Curator."""

import math
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class DomainModel(BaseModel):
    """Shared validation policy for persisted and output domain records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PreferenceOrigin(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class PreferencePolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class CulturalMemory(DomainModel):
    """An unchanged observation supplied by a user."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    created_at: datetime

    @field_validator("id", "text")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("must include a timezone")
        return value


class Preference(DomainModel):
    """A structured taste signal with explicit provenance."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    polarity: PreferencePolarity
    origin: PreferenceOrigin
    source_memory_ids: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("id", "label")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("source_memory_ids")
    @classmethod
    def validate_source_memory_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not memory_id.strip() for memory_id in value):
            raise ValueError("source memory IDs must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("source memory IDs must be unique")
        return value

    @model_validator(mode="after")
    def require_evidence_for_inference(self) -> "Preference":
        if self.origin is PreferenceOrigin.INFERRED and not self.source_memory_ids:
            raise ValueError("inferred preferences require at least one source memory ID")
        return self


class CulturalEvent(DomainModel):
    """A generic cultural event that may be recommended."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: str | None = None
    venue: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    source_url: HttpUrl | None = None

    @field_validator("id", "title", "description")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_date_range(self) -> "CulturalEvent":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end date must not be before start date")
        return self


class PreferenceContribution(DomainModel):
    """One term of the positive-minus-negative cosine ranking formula.

    ``weight`` is signed: +1 / positive count or -1 / negative count.
    Preference confidence is provenance only and is not used by this baseline.
    """

    preference: Preference
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    weight: float = Field(ge=-1.0, le=1.0)
    contribution: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_contribution(self) -> "PreferenceContribution":
        if self.weight == 0 or ((self.weight > 0) != (self.preference.polarity is PreferencePolarity.POSITIVE)):
            raise ValueError("weight must agree with preference polarity")
        if not math.isclose(self.contribution, self.cosine_similarity * self.weight, abs_tol=1e-12):
            raise ValueError("contribution must equal cosine similarity times weight")
        return self


class Recommendation(DomainModel):
    """A transient recommendation result, not persisted user truth."""

    event_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
    supporting_memory_ids: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    event: CulturalEvent | None = None
    ranking_score: float | None = Field(default=None, allow_inf_nan=False)
    # None distinguishes legacy scores without attribution from computed zeros.
    preference_contributions: tuple[PreferenceContribution, ...] | None = None

    @model_validator(mode="after")
    def validate_ranking_details(self) -> "Recommendation":
        if self.event is not None and self.event.id != self.event_id:
            raise ValueError("event ID must match the recommended event")
        if len(set(self.supporting_memory_ids)) != len(self.supporting_memory_ids):
            raise ValueError("supporting memory IDs must be unique")
        if self.preference_contributions is not None:
            if self.ranking_score is None or not math.isclose(
                self.ranking_score,
                math.fsum(item.contribution for item in self.preference_contributions),
                abs_tol=1e-12,
            ):
                raise ValueError("ranking score must equal the sum of contributions")
            source_ids = {
                memory_id
                for item in self.preference_contributions
                if item.contribution != 0.0
                for memory_id in item.preference.source_memory_ids
            }
            if set(self.supporting_memory_ids) != source_ids:
                raise ValueError("supporting memory IDs must match nonzero contribution sources")
        return self


class RankedEvent(DomainModel):
    """An event ID paired with its raw semantic ranking score."""

    event_id: str = Field(min_length=1)
    score: float
    preference_contributions: tuple[PreferenceContribution, ...] | None = None

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @model_validator(mode="after")
    def validate_score_contributions(self) -> "RankedEvent":
        if self.preference_contributions is not None and not math.isclose(
            self.score,
            math.fsum(item.contribution for item in self.preference_contributions),
            abs_tol=1e-12,
        ):
            raise ValueError("score must equal the sum of contributions")
        return self
