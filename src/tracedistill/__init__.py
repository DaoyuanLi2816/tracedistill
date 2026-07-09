"""tracedistill — distill deterministic reasoning traces into a LoRA adapter.

The generalised core of team **VCDAD**'s silver-medal solution to the NVIDIA Nemotron
Model Reasoning Challenge, extracted into a small, tested library you can run on your
own data. The technique: wrap ``(problem, teacher chain-of-thought, answer)`` triples
into a strict ``<think>…</think>\\boxed{}`` **format contract**, then SFT a LoRA adapter
with a two-phase **Train → Nudge** schedule, **type-stratified** batching, **architecture-
aware** targets (covering Mamba-2 SSM projections, not just attention), and **NEFTune** —
so the model re-derives each answer itself at inference time.

The light core (numpy / pandas / pyyaml) is import-safe without a GPU stack; torch /
transformers / trl / peft live in the optional ``[train]`` extra and are imported lazily.
"""

from __future__ import annotations

from .data import CANONICAL_COLUMNS, load_cot_csv, two_phase_split
from .formatting import DEFAULT_PROMPT_SUFFIX, build_record, build_records, strip_boxed
from .lora import (
    ATTENTION_TARGETS,
    DEFAULT_TARGET_MODULES,
    MAMBA_TARGETS,
    MLP_TARGETS,
    architecture_aware_targets,
    target_modules_from_model,
)
from .sampling import PrecomputedOrderSampler, build_stratified_index_order

__version__ = "0.1.1"

# Heavy (torch/trl) symbols are imported lazily so `import tracedistill` works with only
# the light core installed.
_LAZY = {
    "PhaseConfig": "training",
    "TwoPhaseConfig": "training",
    "StratifiedSFTTrainer": "training",
    "make_formatting_func": "training",
    "train_two_phase": "training",
}


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f"{__name__}.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # formatting (the format contract)
    "DEFAULT_PROMPT_SUFFIX",
    "build_record",
    "build_records",
    "strip_boxed",
    # sampling
    "build_stratified_index_order",
    "PrecomputedOrderSampler",
    # data
    "CANONICAL_COLUMNS",
    "load_cot_csv",
    "two_phase_split",
    # lora targets
    "DEFAULT_TARGET_MODULES",
    "architecture_aware_targets",
    "target_modules_from_model",
    "ATTENTION_TARGETS",
    "MAMBA_TARGETS",
    "MLP_TARGETS",
    # training (lazy, needs the [train] extra)
    "PhaseConfig",
    "TwoPhaseConfig",
    "StratifiedSFTTrainer",
    "make_formatting_func",
    "train_two_phase",
]
