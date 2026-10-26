"""Run one real explicit-preference extraction with Mistral."""

import argparse
import asyncio

from cultural_curator.llm import MistralPreferenceExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="One cultural-memory sentence to extract")
    args = parser.parse_args()

    extraction = asyncio.run(MistralPreferenceExtractor().extract_async(args.text))
    print(extraction.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
