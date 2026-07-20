#!/usr/bin/env python3
"""Create controlled SSv2 attribute-vocabulary ablations.

The source vocabulary may map class names to strings (the current SSv2 format) or
lists.  The generated JSON preserves that top-level structure and value type.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DUMMY_TOKENS = [
    "generic", "visual", "action", "motion", "object", "scene", "video", "person",
    "hand", "movement", "temporal", "context", "interaction", "dynamic", "sequence", "event",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source attribute JSON file")
    parser.add_argument("--output", required=True, type=Path, help="Destination JSON file")
    parser.add_argument("--mode", required=True, choices=("dummy", "shuffled"))
    parser.add_argument("--seed", type=int, default=0, help="Random seed used by shuffled mode")
    parser.add_argument("--num_attributes", type=int, default=16, help="Number of dummy tokens")
    return parser.parse_args()


def token_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value.split())
    if isinstance(value, list):
        return len(value)
    return len(str(value).split())


def preview(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def make_dummy_like(value: Any, tokens: list[str]) -> Any:
    if isinstance(value, list):
        return tokens.copy()
    if isinstance(value, str):
        return " ".join(tokens)
    raise TypeError(f"Unsupported attribute value type: {type(value).__name__}")


def deranged_indices(size: int, seed: int) -> list[int]:
    if size < 2:
        raise ValueError("Shuffled ablation requires at least two classes.")
    indices = list(range(size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if all(source == target for target, source in enumerate(indices)):
        indices = indices[1:] + indices[:1]
    if any(source == target for target, source in enumerate(indices)):
        # A cyclic shift is always a derangement for size >= 2.
        indices = list(range(1, size)) + [0]
    return indices


def main() -> None:
    args = parse_args()
    if args.num_attributes <= 0:
        raise ValueError("--num_attributes must be positive.")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must not overwrite --input.")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not source:
        raise ValueError("Expected a non-empty JSON object mapping class names to attributes.")
    if not all(isinstance(key, str) for key in source):
        raise ValueError("All JSON object keys must be class-name strings.")
    value_types = {type(value) for value in source.values()}
    if not value_types.issubset({str, list}):
        found = ", ".join(sorted(value_type.__name__ for value_type in value_types))
        raise ValueError(f"Only string/list values are supported; found: {found}")

    keys = list(source)
    dummy_tokens = [DUMMY_TOKENS[index % len(DUMMY_TOKENS)] for index in range(args.num_attributes)]
    if args.mode == "dummy":
        generated = {key: make_dummy_like(value, dummy_tokens) for key, value in source.items()}
        identity_assignments = None
    else:
        permutation = deranged_indices(len(keys), args.seed)
        generated = {key: source[keys[source_index]] for key, source_index in zip(keys, permutation)}
        identity_assignments = sum(value == source[key] for key, value in generated.items())
        if identity_assignments:
            raise RuntimeError("Shuffled output unexpectedly retained class-to-attribute assignments.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = [token_count(value) for value in generated.values()]
    print(f"mode={args.mode}")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"classes={len(generated)}")
    print(f"attribute_tokens: min={min(counts)}, max={max(counts)}, avg={sum(counts) / len(counts):.2f}")
    if identity_assignments is not None:
        print(f"shuffled_identity_assignments={identity_assignments}")
    print("first_five:")
    for key, value in list(generated.items())[:5]:
        print(f"  {key}: {preview(value)}")


if __name__ == "__main__":
    main()
