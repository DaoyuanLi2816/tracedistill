"""Golden + behavior tests for the format contract."""

import random

import pandas as pd
import reference_impl as ref

import tracedistill as td
from tracedistill.formatting import build_record
from conftest import random_dataframe


def test_default_suffix_matches_competition():
    # The library's default suffix must be byte-identical to the grader's suffix.
    assert td.DEFAULT_PROMPT_SUFFIX == ref.PROMPT_SUFFIX


def test_build_records_matches_reference(sample_df):
    got_records, got_types = td.build_records(sample_df, prompt_suffix=ref.PROMPT_SUFFIX)
    exp_records, exp_types = ref.build_records(sample_df)
    assert got_records == exp_records
    assert got_types == exp_types


def test_build_records_matches_reference_fuzz():
    # 300 randomized DataFrames: library output must equal the verbatim oracle exactly.
    rng = random.Random(12345)
    for _ in range(300):
        df = random_dataframe(rng, rng.randint(1, 30))
        got = td.build_records(df, prompt_suffix=ref.PROMPT_SUFFIX)
        exp = ref.build_records(df)
        assert got == exp


def test_default_call_equals_reference(sample_df):
    # Calling with the library default suffix (no override) also matches the oracle,
    # since the default IS the competition suffix.
    assert td.build_records(sample_df) == ref.build_records(sample_df)


def test_build_record_strips_upstream_boxed_and_reattaches_answer():
    rec = build_record("Q?", "reason \\boxed{99} done", 42)
    asst = rec["messages"][1]["content"]
    assert "\\boxed{99}" not in asst  # upstream answer stripped
    assert asst.endswith("</think>\n\\boxed{42}")  # official answer reattached
    assert rec["messages"][0]["content"] == "Q?" + td.DEFAULT_PROMPT_SUFFIX


def test_build_record_drops_unusable_cot():
    assert build_record("Q", "", 1) is None
    assert build_record("Q", "nan", 1) is None
    assert build_record("Q", "   ", 1) is None  # whitespace-only, < min_cot_len after strip
    assert build_record("Q", "abc", 1) is None  # 3 chars < default min_cot_len=5
    assert build_record("Q", "abcde", 1) is not None


def test_build_record_custom_suffix_and_min_len():
    rec = build_record("Q", "yo", 1, prompt_suffix=" SUFFIX", min_cot_len=2)
    assert rec is not None
    assert rec["messages"][0]["content"] == "Q SUFFIX"


def test_strip_boxed():
    assert td.strip_boxed("a \\boxed{1} b \\boxed{two} c") == "a  b  c"
    assert td.strip_boxed("no box here") == "no box here"


def test_strip_boxed_handles_nested_braces():
    # The frozen `reference_impl.py` oracle uses `\\boxed\{[^}]*\}`, which stops at the
    # *first* `}` and leaves a dangling fragment for any boxed answer containing its own
    # braces (fractions, sets, intervals -- common in math reasoning traces). The live
    # library tracks brace depth instead, so it must NOT match the oracle here; this is
    # a deliberate improvement, not drift (see tests/reference_impl.py's own docstring).
    assert td.strip_boxed("ans \\boxed{\\frac{1}{2}} done") == "ans  done"
    assert td.strip_boxed("set \\boxed{\\{1, 2, 3\\}} done") == "set  done"
    assert td.strip_boxed("nested \\boxed{\\sqrt{\\frac{1}{2}}} done") == "nested  done"
    # An unterminated \boxed{ (a truncated trace) is left alone rather than consuming
    # the rest of the string.
    assert td.strip_boxed("truncated \\boxed{oops") == "truncated \\boxed{oops"


def test_build_record_reattaches_answer_cleanly_after_nested_boxed():
    rec = build_record("Q?", "reasoning \\boxed{\\frac{1}{2}} more reasoning", 42)
    asst = rec["messages"][1]["content"]
    assert "{2}}" not in asst  # no dangling fragment from the stripped upstream answer
    assert asst == "reasoning  more reasoning\n</think>\n\\boxed{42}"


def test_build_records_accepts_list_of_dicts():
    rows = [
        {"prompt": "p", "generated_cot": "reasoning here", "answer": 7, "type": "t1"},
        {"prompt": "q", "generated_cot": "", "answer": 8, "type": "t2"},  # dropped
    ]
    records, types = td.build_records(rows)
    assert len(records) == 1 and types == ["t1"]
    # Equivalent to passing the same data as a DataFrame.
    assert td.build_records(rows) == td.build_records(pd.DataFrame(rows))
