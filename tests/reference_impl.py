"""Verbatim copies of the original competition training functions.

These are pasted UNCHANGED from the silver-medal solution (now preserved under
``competition/training.py``) and serve as the **golden oracle**: the tests assert that
the ``tracedistill`` library reproduces their output byte-for-byte / value-for-value.

DO NOT refactor, reformat, or "clean up" anything in this file — drift away from the
medal-winning code is exactly the bug these characterization tests exist to catch.
"""

import math
import random
import re
from collections import defaultdict

# Verbatim from code/training.py (the grader's suffix, reused at train time).
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def build_records(source_df):
    """Convert a dataframe into SFT records + a parallel list of type labels (for the sampler)."""
    records, types = [], []
    for _, row in source_df.iterrows():
        cot = str(row["generated_cot"])
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            continue
        cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
        user_content  = str(row["prompt"]) + PROMPT_SUFFIX
        asst_content  = cot_cleaned + f"\n</think>\n\\boxed{{{str(row['answer'])}}}"
        records.append({"messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ]})
        types.append(str(row["type"]))
    return records, types


def build_stratified_index_order(labels, batch_size, seed):
    """Precompute a type-balanced ("stratified") sample order."""
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    rng = random.Random(seed)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)
    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches   = [[] for _ in range(n_batches)]
    b_order   = list(range(n_batches))
    rng.shuffle(b_order)
    assigned  = 0
    for label in sorted(by_label.keys()):
        for idx in by_label[label]:
            batches[b_order[assigned % n_batches]].append(idx)
            assigned += 1
    order = [idx for batch in batches for idx in batch]
    if len(order) != len(labels):
        raise ValueError("Stratified order size mismatch")
    return order
