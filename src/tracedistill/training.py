"""Two-phase ``Train → Nudge`` SFT for reasoning-trace distillation.

This is the torch/trl-dependent layer (the optional ``[train]`` extra). It wires the
torch-free pieces — the :mod:`~tracedistill.formatting` contract, the
:mod:`~tracedistill.sampling` stratified order, the :mod:`~tracedistill.data` split, and
the :mod:`~tracedistill.lora` targets — into a reusable two-phase fine-tuning loop:

- **Phase 1 · Train**: a hard, fast pass (high LR, gradient clipping off) for broad
  coverage of all problem types.
- **Phase 2 · Nudge**: a tiny continuation (≈1/40 LR, cosine, clipping back on) focused
  on the hard types, with a balanced sprinkle of fresh easy data to prevent forgetting.

Both phases train the **same** model object (Phase 2 continues from Phase 1's weights),
use the stratified sampler, and apply NEFTune.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Callable, Iterable, Sequence

import pandas as pd
from datasets import Dataset as HFDataset
from trl import SFTConfig, SFTTrainer
from torch.utils.data import DataLoader

# trl renamed SFTConfig's `max_seq_length` to `max_length`; pick whichever this trl has,
# so the library works across the supported trl range.
_SFT_LEN_FIELD = "max_length" if any(f.name == "max_length" for f in fields(SFTConfig)) else "max_seq_length"

from .data import two_phase_split
from .formatting import DEFAULT_PROMPT_SUFFIX, build_records
from .lora import DEFAULT_TARGET_MODULES, target_modules_from_model
from .sampling import PrecomputedOrderSampler, build_stratified_index_order

__all__ = [
    "PhaseConfig",
    "TwoPhaseConfig",
    "StratifiedSFTTrainer",
    "make_formatting_func",
    "train_two_phase",
]


@dataclass
class PhaseConfig:
    """Hyper-parameters for one fine-tuning phase (a thin, typed view over the subset of
    :class:`trl.SFTConfig` fields the recipe varies)."""

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1e9  # 1e9 ≈ clipping OFF
    neftune_noise_alpha: float = 5.0
    logging_steps: int = 50

    @classmethod
    def train(cls, **overrides) -> "PhaseConfig":
        """Phase 1 defaults: aggressive, clipping off, broad coverage."""
        return cls(**overrides)

    @classmethod
    def nudge(cls, **overrides) -> "PhaseConfig":
        """Phase 2 defaults: 1/40 LR, cosine, warmup, clipping on, hard-focused."""
        base = dict(
            learning_rate=5e-6,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            max_grad_norm=1.0,
            logging_steps=5,
        )
        base.update(overrides)
        return cls(**base)


@dataclass
class TwoPhaseConfig:
    """End-to-end configuration for :func:`train_two_phase`."""

    hard_types: Sequence[str]
    output_dir: str = "tracedistill_output"
    max_length: int = 8192
    seed: int = 42
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX
    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Sequence[str] | None = None  # None -> architecture-aware autodetect
    # phases
    phase1: PhaseConfig = field(default_factory=PhaseConfig.train)
    phase2: PhaseConfig = field(default_factory=PhaseConfig.nudge)
    bf16: bool = True
    enable_thinking: bool = True


def make_formatting_func(tokenizer, *, enable_thinking: bool = True) -> Callable:
    """Return an SFTTrainer ``formatting_func`` that renders ``{"messages": [...]}`` rows
    through *tokenizer*'s chat template (inserting the opening ``<think>`` when
    ``enable_thinking`` is supported)."""

    def formatting_prompts_func(example):
        messages = example["messages"]
        conversations = [messages] if (messages and isinstance(messages[0], dict)) else messages
        texts = []
        for conversation in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                )
            except TypeError:  # older tokenizers don't accept enable_thinking
                text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            texts.append(text)
        return texts

    return formatting_prompts_func


class StratifiedSFTTrainer(SFTTrainer):
    """:class:`trl.SFTTrainer` that feeds a precomputed type-balanced order through the
    train DataLoader. Only ``get_train_dataloader`` is overridden — everything else is
    stock TRL (no patching of internal shuffling)."""

    def __init__(self, *args, stratified_order: Sequence[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratified_order = stratified_order

    def get_train_dataloader(self):
        if self.stratified_order is None:
            return super().get_train_dataloader()
        if len(self.stratified_order) != len(self.train_dataset):
            raise ValueError("Stratified order length does not match train dataset")
        kwargs = {
            "batch_size": self.args.per_device_train_batch_size,
            "sampler": PrecomputedOrderSampler(self.stratified_order),
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
        }
        if self.args.dataloader_num_workers > 0:
            kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(self.train_dataset, **kwargs)


def _sft_config(phase: PhaseConfig, cfg: TwoPhaseConfig, output_dir: str) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=phase.num_train_epochs,
        per_device_train_batch_size=phase.per_device_train_batch_size,
        gradient_accumulation_steps=phase.gradient_accumulation_steps,
        learning_rate=phase.learning_rate,
        lr_scheduler_type=phase.lr_scheduler_type,
        warmup_steps=phase.warmup_steps,
        **{_SFT_LEN_FIELD: cfg.max_length},
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=phase.max_grad_norm,
        logging_steps=phase.logging_steps,
        save_strategy="no",
        bf16=cfg.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=cfg.seed,
        report_to="none",
        packing=False,
        neftune_noise_alpha=phase.neftune_noise_alpha,
    )


def _run_phase(model, tokenizer, df, phase: PhaseConfig, cfg: TwoPhaseConfig, name: str):
    records, types = build_records(df, prompt_suffix=cfg.prompt_suffix)
    if not records:
        raise ValueError(f"Phase {name!r} produced 0 usable records.")
    dataset = HFDataset.from_list(records)
    eff_batch = phase.per_device_train_batch_size * phase.gradient_accumulation_steps
    order = build_stratified_index_order(types, eff_batch, cfg.seed)
    trainer = StratifiedSFTTrainer(
        model=model,
        args=_sft_config(phase, cfg, f"{cfg.output_dir}/{name}"),
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=make_formatting_func(tokenizer, enable_thinking=cfg.enable_thinking),
        stratified_order=order,
    )
    trainer.train()
    return trainer


def train_two_phase(model, tokenizer, df: pd.DataFrame, cfg: TwoPhaseConfig):
    """Run the full ``Train → Nudge`` schedule on an already-loaded ``model``/``tokenizer``.

    *model* must already have a LoRA adapter attached (e.g. via
    ``FastLanguageModel.get_peft_model`` or ``peft.get_peft_model`` with
    :func:`tracedistill.lora.target_modules_from_model` / :data:`DEFAULT_TARGET_MODULES`).
    Both phases train the same object; Phase 2 continues from Phase 1's weights. Returns
    the trained ``model``.
    """
    phase1_df, phase2_df = two_phase_split(df, cfg.hard_types, seed=cfg.seed)
    _run_phase(model, tokenizer, phase1_df, cfg.phase1, cfg, "phase1")
    _run_phase(model, tokenizer, phase2_df, cfg.phase2, cfg, "phase2")
    return model
