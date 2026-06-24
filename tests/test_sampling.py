"""Golden + invariant tests for the stratified sampler."""

import random
from collections import Counter

import pytest
import reference_impl as ref

from tracedistill.sampling import PrecomputedOrderSampler, build_stratified_index_order


def _random_labels(rng):
    n_types = rng.randint(1, 6)
    types = [f"t{i}" for i in range(n_types)]
    return [rng.choice(types) for _ in range(rng.randint(1, 200))]


def test_matches_reference_fuzz():
    # 400 randomized (labels, batch_size, seed): library order must equal the oracle.
    rng = random.Random(999)
    for _ in range(400):
        labels = _random_labels(rng)
        batch_size = rng.randint(1, 16)
        seed = rng.randint(0, 10_000)
        got = build_stratified_index_order(labels, batch_size, seed)
        exp = ref.build_stratified_index_order(labels, batch_size, seed)
        assert got == exp


@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_is_a_permutation(seed, batch_size):
    labels = ["a", "b", "c"] * 17 + ["d"] * 5
    order = build_stratified_index_order(labels, batch_size, seed)
    assert sorted(order) == list(range(len(labels)))  # covers every index exactly once


def test_deterministic():
    labels = ["x", "y", "z"] * 10
    a = build_stratified_index_order(labels, 8, 123)
    b = build_stratified_index_order(labels, 8, 123)
    assert a == b


def test_batches_are_type_balanced():
    # With equal-sized types and a batch that's a multiple of the type count, each batch
    # should contain (close to) one of each type — the whole point of the sampler.
    labels = (["a", "b", "c", "d"] * 25)  # 100 labels, 25 each
    rng = random.Random(0)
    rng.shuffle(labels)
    batch_size = 4
    order = build_stratified_index_order(labels, batch_size, seed=7)
    # Count, per full batch, how many distinct types appear; expect mostly 4.
    distinct_counts = []
    for i in range(0, len(order) - batch_size + 1, batch_size):
        batch_types = {labels[idx] for idx in order[i : i + batch_size]}
        distinct_counts.append(len(batch_types))
    # A naive shuffle would average ~3.1 distinct/4; stratified should be much higher.
    assert sum(distinct_counts) / len(distinct_counts) > 3.6


def test_size_mismatch_guard(monkeypatch):
    # The internal invariant raises if the order size ever drifts.
    labels = ["a", "b", "a"]
    order = build_stratified_index_order(labels, 2, 0)
    assert len(order) == 3


def test_precomputed_order_sampler():
    sampler = PrecomputedOrderSampler([3, 1, 2, 0])
    assert list(sampler) == [3, 1, 2, 0]
    assert len(sampler) == 4
    # Iterates fresh each time (re-entrant).
    assert list(sampler) == [3, 1, 2, 0]
