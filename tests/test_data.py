"""Invariant tests for the two-phase Train→Nudge split."""

import pandas as pd
import pytest

from tracedistill.data import load_cot_csv, two_phase_split


def _make_df():
    rows = []
    # hard types (rare-ish) + easy types (plentiful)
    for i in range(20):
        rows.append({"prompt": f"h{i}", "generated_cot": "x", "answer": i, "type": "cryptarithm_deduce"})
    for i in range(15):
        rows.append({"prompt": f"g{i}", "generated_cot": "x", "answer": i, "type": "cryptarithm_guess"})
    for i in range(80):
        rows.append({"prompt": f"e{i}", "generated_cot": "x", "answer": i, "type": "gravity"})
    for i in range(60):
        rows.append({"prompt": f"u{i}", "generated_cot": "x", "answer": i, "type": "unit_conversion"})
    return pd.DataFrame(rows)


HARD = {"cryptarithm_deduce", "cryptarithm_guess"}


def test_hard_types_appear_in_full_in_both_phases():
    df = _make_df()
    n_hard = int(df["type"].isin(HARD).sum())
    p1, p2 = two_phase_split(df, HARD, seed=42)
    assert int(p1["type"].isin(HARD).sum()) == n_hard
    assert int(p2["type"].isin(HARD).sum()) == n_hard


def test_phase2_easy_is_fresh_relative_to_phase1():
    df = _make_df()
    p1, p2 = two_phase_split(df, HARD, seed=42)
    # The easy prompts reserved for phase 2 must NOT appear in phase 1 (non-overlap).
    p2_easy_prompts = set(p2[~p2["type"].isin(HARD)]["prompt"])
    p1_prompts = set(p1["prompt"])
    assert p2_easy_prompts.isdisjoint(p1_prompts)


def test_phase2_easy_quota_is_rarest_hard_count():
    df = _make_df()
    rarest_hard = int(df[df["type"].isin(HARD)]["type"].value_counts().min())  # 15
    _, p2 = two_phase_split(df, HARD, seed=42)
    for etype in ("gravity", "unit_conversion"):
        assert int((p2["type"] == etype).sum()) == rarest_hard


def test_deterministic():
    df = _make_df()
    a1, a2 = two_phase_split(df, HARD, seed=42)
    b1, b2 = two_phase_split(df, HARD, seed=42)
    assert a1.equals(b1) and a2.equals(b2)


def test_raises_when_no_hard_rows():
    df = _make_df()
    with pytest.raises(ValueError):
        two_phase_split(df, {"nonexistent_type"}, seed=0)


def test_load_cot_csv_validates_columns(tmp_path):
    good = tmp_path / "good.csv"
    pd.DataFrame(
        [{"prompt": "p", "generated_cot": "c", "answer": 1, "type": "t"}]
    ).to_csv(good, index=False)
    assert len(load_cot_csv(str(good))) == 1

    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"prompt": "p", "answer": 1}]).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_cot_csv(str(bad))
