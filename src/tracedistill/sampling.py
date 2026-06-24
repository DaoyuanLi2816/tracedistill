"""Type-stratified, round-robin batch ordering.

With a tiny effective batch size (e.g. ``batch=1 × grad_accum=8`` on a 30B model), a
naive shuffle can make an entire effective batch a single problem type, which swings
the gradient toward that type. :func:`build_stratified_index_order` precomputes a
type-balanced sample order ("deal the cards round-robin") so every effective batch is
a balanced mix of types. :class:`PrecomputedOrderSampler` then feeds that fixed order
through a DataLoader without touching the trainer's internals.

Pure standard library — no torch — so it imports and unit-tests anywhere. The order
algorithm is deterministic given ``seed`` and is pinned byte-for-byte by the golden
tests.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Iterator, Sequence

__all__ = ["build_stratified_index_order", "PrecomputedOrderSampler"]


def build_stratified_index_order(
    labels: Sequence[str], batch_size: int, seed: int
) -> list[int]:
    """Return a permutation of ``range(len(labels))`` in which each consecutive
    ``batch_size`` block is type-balanced.

    Algorithm: bucket indices by label, shuffle each bucket (seeded), then deal each
    bucket's indices round-robin into ``ceil(len/batch_size)`` batch slots (whose fill
    order is itself shuffled), and flatten. Deterministic for a fixed ``seed``.
    """
    # 1) bucket sample indices by label (insertion order preserved)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    # 2) shuffle within each bucket (seeded, reproducible)
    rng = random.Random(seed)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)
    # 3) prepare n_batches slots; shuffle the slot fill order too
    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches: list[list[int]] = [[] for _ in range(n_batches)]
    b_order = list(range(n_batches))
    rng.shuffle(b_order)
    # 4) deal: hand each label's indices one at a time round-robin into the slots
    assigned = 0
    for label in sorted(by_label.keys()):
        for idx in by_label[label]:
            batches[b_order[assigned % n_batches]].append(idx)
            assigned += 1
    # 5) flatten the slots into a single 1-D order
    order = [idx for batch in batches for idx in batch]
    if len(order) != len(labels):
        raise ValueError("Stratified order size mismatch")
    return order


class PrecomputedOrderSampler:
    """A "take-it-as-given" sampler that yields indices in a fixed precomputed order
    with no reshuffling. Works as a ``torch.utils.data.DataLoader`` sampler (any object
    with ``__iter__``/``__len__`` is accepted) without importing torch.
    """

    def __init__(self, order: Iterable[int]) -> None:
        self.order = list(order)

    def __iter__(self) -> Iterator[int]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)
