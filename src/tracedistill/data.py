"""Data loading and the two-phase ``Train → Nudge`` split.

The canonical input is a table with four columns: ``prompt`` (the problem), a teacher
``generated_cot`` (the reasoning trace to distill), ``answer`` (the authoritative
label), and ``type`` (a problem-family label used for stratification). :func:`load_cot_csv`
reads such a table; :func:`two_phase_split` carves it into the two non-overlapping
training sets the ``Train → Nudge`` schedule consumes.
"""

from __future__ import annotations

import random
from typing import Iterable

import pandas as pd

__all__ = ["CANONICAL_COLUMNS", "load_cot_csv", "two_phase_split"]

#: The columns :func:`tracedistill.formatting.build_records` expects by default.
CANONICAL_COLUMNS = ("prompt", "generated_cot", "answer", "type")


def load_cot_csv(path: str, *, require_columns: bool = True) -> pd.DataFrame:
    """Read a trace-distillation CSV into a DataFrame.

    With ``require_columns=True`` (default), validates that the canonical columns are
    present and raises a clear error otherwise.
    """
    df = pd.read_csv(path)
    if require_columns:
        missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV {path!r} is missing required columns {missing}; "
                f"found {list(df.columns)}. Expected {list(CANONICAL_COLUMNS)}."
            )
    return df


def two_phase_split(
    df: pd.DataFrame,
    hard_types: Iterable[str],
    *,
    seed: int = 42,
    type_key: str = "type",
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into ``(phase1_df, phase2_df)`` for the ``Train → Nudge`` schedule.

    - **Phase 1 (Train)**: everything *except* the easy rows reserved for Phase 2 —
      i.e. all hard-type rows plus most easy-type rows. Broad coverage.
    - **Phase 2 (Nudge)**: all hard-type rows plus ``n`` fresh rows of *each* easy type,
      where ``n`` is the size of the rarest hard type. Hard-focused but type-balanced,
      with fresh easy data to anchor against catastrophic forgetting.

    The hard types appear **in full in both** phases — the basis for emphasising them in
    both passes. The easy rows are partitioned so Phase 2 sees only *unseen* easy data.
    Deterministic given ``seed``.
    """
    hard_types = set(hard_types)
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    hard_mask = df[type_key].isin(hard_types)
    hard_df = df[hard_mask].reset_index(drop=True)
    easy_df = df[~hard_mask].copy()  # keep original df row indices for set-based selection
    if hard_df.empty:
        raise ValueError(f"No rows of any hard type {sorted(hard_types)} found in df.")

    # n = size of the rarest hard type → per-easy-type quota for the nudge set, so easy
    # and hard land in the same order of magnitude (neither swamps the other).
    n = int(hard_df[type_key].value_counts().min())

    rng = random.Random(seed + 1)  # decoupled from the shuffle's RNG
    nudge_easy_idx: set[int] = set()
    for _etype, egroup in easy_df.groupby(type_key):
        sampled = rng.sample(list(egroup.index), min(n, len(egroup)))
        nudge_easy_idx.update(sampled)

    phase1_df = df[~df.index.isin(nudge_easy_idx)].reset_index(drop=True)
    nudge_easy_df = df[df.index.isin(nudge_easy_idx)]
    phase2_df = pd.concat([hard_df, nudge_easy_df]).reset_index(drop=True)
    return phase1_df, phase2_df
