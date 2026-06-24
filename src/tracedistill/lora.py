"""Architecture-aware LoRA target selection.

The reasoning base used in the competition is a **hybrid Mamba-2 + MoE** model, so a
vanilla "attention-only" LoRA recipe misses the SSM projections. This module picks the
target modules that actually exist on a model, covering three families:

- **attention**: ``q_proj``/``k_proj``/``v_proj``/``o_proj``
- **Mamba-2 (SSM)**: ``in_proj``/``out_proj`` — *the* detail a Llama recipe omits
- **MLP / MoE**: ``gate_proj``/``up_proj``/``down_proj``

Standard library only — no torch — so it can be unit-tested without loading a model.
"""

from __future__ import annotations

from typing import Iterable

__all__ = [
    "ATTENTION_TARGETS",
    "MAMBA_TARGETS",
    "MLP_TARGETS",
    "DEFAULT_TARGET_MODULES",
    "architecture_aware_targets",
    "target_modules_from_model",
]

ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MAMBA_TARGETS = ("in_proj", "out_proj")  # Mamba-2 SSM projections
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")

#: The exact list used by the silver-medal solution (attention + Mamba + up/down MLP).
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj",
    "up_proj", "down_proj",
]


def architecture_aware_targets(
    module_names: Iterable[str],
    *,
    attention: bool = True,
    mamba: bool = True,
    mlp: bool = True,
) -> list[str]:
    """Return the LoRA ``target_modules`` present among *module_names*.

    *module_names* is any iterable of dotted module paths (e.g. from
    ``model.named_modules()``); only the leaf name of each is considered. The result
    preserves a stable family order (attention → Mamba → MLP) and de-duplicates.
    """
    leaves = {name.rsplit(".", 1)[-1] for name in module_names}
    pools: list[str] = []
    if attention:
        pools += ATTENTION_TARGETS
    if mamba:
        pools += MAMBA_TARGETS
    if mlp:
        pools += MLP_TARGETS

    targets: list[str] = []
    for name in pools:
        if name in leaves and name not in targets:
            targets.append(name)
    return targets


def target_modules_from_model(model, **kwargs) -> list[str]:
    """Convenience wrapper: :func:`architecture_aware_targets` over a live model's
    ``named_modules()``. Does not import torch itself."""
    return architecture_aware_targets((name for name, _ in model.named_modules()), **kwargs)
