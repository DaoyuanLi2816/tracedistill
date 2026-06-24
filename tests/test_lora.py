"""Tests for architecture-aware LoRA target selection."""

from tracedistill.lora import (
    DEFAULT_TARGET_MODULES,
    architecture_aware_targets,
    target_modules_from_model,
)


# A fake hybrid Mamba-2 + attention + MLP module-name set.
HYBRID_NAMES = [
    "model.layers.0.mixer.in_proj",
    "model.layers.0.mixer.out_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.1.self_attn.k_proj",
    "model.layers.1.self_attn.v_proj",
    "model.layers.1.self_attn.o_proj",
    "model.layers.1.mlp.up_proj",
    "model.layers.1.mlp.down_proj",
    "model.embed_tokens",
    "lm_head",
]


def test_picks_present_targets_including_mamba():
    targets = architecture_aware_targets(HYBRID_NAMES)
    # Mamba SSM projections must be picked up — the detail a Llama recipe misses.
    assert "in_proj" in targets and "out_proj" in targets
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(targets)
    assert {"up_proj", "down_proj"} <= set(targets)
    # lm_head / embeddings are never LoRA targets here.
    assert "lm_head" not in targets


def test_family_order_is_stable():
    targets = architecture_aware_targets(HYBRID_NAMES)
    # attention family precedes Mamba precedes MLP.
    assert targets.index("q_proj") < targets.index("in_proj") < targets.index("up_proj")


def test_attention_only_model():
    names = ["m.q_proj", "m.k_proj", "m.v_proj", "m.o_proj", "m.up_proj", "m.down_proj"]
    targets = architecture_aware_targets(names)
    assert "in_proj" not in targets  # no Mamba present
    assert {"q_proj", "up_proj"} <= set(targets)


def test_toggles():
    assert architecture_aware_targets(HYBRID_NAMES, mamba=False).count("in_proj") == 0
    assert architecture_aware_targets(HYBRID_NAMES, mlp=False).count("up_proj") == 0
    assert architecture_aware_targets(HYBRID_NAMES, attention=False).count("q_proj") == 0


def test_no_duplicates():
    names = ["a.q_proj", "b.q_proj", "c.in_proj", "d.in_proj"]
    targets = architecture_aware_targets(names)
    assert len(targets) == len(set(targets))
    assert sorted(targets) == ["in_proj", "q_proj"]


def test_target_modules_from_model():
    class FakeModule:
        def named_modules(self):
            return [(n, object()) for n in HYBRID_NAMES]

    targets = target_modules_from_model(FakeModule())
    assert "in_proj" in targets and "q_proj" in targets


def test_default_target_modules_constant():
    # The shipped default mirrors the silver-medal solution's exact list.
    assert DEFAULT_TARGET_MODULES == [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj", "up_proj", "down_proj",
    ]
