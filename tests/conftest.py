"""Shared pytest fixtures + path shim so test modules can ``import reference_impl``."""

import os
import random
import sys

import pandas as pd
import pytest

# Make the verbatim golden oracle importable as a top-level module.
sys.path.insert(0, os.path.dirname(__file__))


# A small pool of realistic problem types (mirrors the competition's families).
_TYPES = [
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_guess",
    "gravity",
    "unit_conversion",
    "numeral_system",
    "text_cipher",
    "bit_manipulation",
]


def _random_cot(rng: random.Random) -> str:
    """Generate an adversarial chain-of-thought string covering the edge cases
    build_records must handle: empty, ``"nan"``, too-short, embedded ``\\boxed{}``,
    unicode/whitespace."""
    roll = rng.random()
    if roll < 0.08:
        return ""  # empty -> dropped
    if roll < 0.16:
        return "nan"  # the string "nan" -> dropped
    if roll < 0.24:
        return rng.choice(["ok", "  ", "abc", "x"])  # < 5 chars (after strip) -> maybe dropped
    body = rng.choice(
        [
            "We reason step by step about ⌈x⌉ and the Greek α.",
            "First compute the carries, then deduce each letter.",
            "Convert to hex 0x1F, count the set bits, verify.",
            "Let A=1, B=2; substitute and check the equation.",
        ]
    )
    # sometimes embed one or more upstream \boxed{...} to be stripped
    if rng.random() < 0.6:
        body += f" \\boxed{{{rng.randint(0, 999)}}}"
    if rng.random() < 0.3:
        body += f" and also \\boxed{{wrong-{rng.randint(0,9)}}} trailing"
    if rng.random() < 0.3:
        body += "\n\n   "  # trailing whitespace to exercise rstrip
    return body


def random_dataframe(rng: random.Random, n: int) -> pd.DataFrame:
    """A fuzzed canonical DataFrame (prompt/generated_cot/answer/type)."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "prompt": rng.choice(
                    ["Solve: A+B=C", "What is 0b101 << 2?", "Roman numeral for 49?", ""]
                )
                + (f" (#{i})" if rng.random() < 0.5 else ""),
                "generated_cot": _random_cot(rng),
                "answer": rng.choice([rng.randint(0, 1000), "XLIX", "1.5", "True", ""]),
                "type": rng.choice(_TYPES),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def rng() -> random.Random:
    return random.Random(20260622)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return random_dataframe(random.Random(7), 40)
