"""Config-driven CLI: ``tracedistill --cfg config.yaml`` (or ``python -m tracedistill.run``).

Loads a base model, attaches an architecture-aware LoRA adapter, reads a CoT dataset,
runs the two-phase ``Train → Nudge`` schedule, and saves (and optionally packages) the
resulting adapter. ``load_config`` and ``RunConfig`` are import-safe without torch so the
config surface can be unit-tested; the heavy model code is imported lazily inside
:func:`run`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from typing import Any, Sequence

import yaml

__all__ = ["RunConfig", "load_config", "run", "main"]


@dataclass
class RunConfig:
    """Everything ``tracedistill --cfg`` needs. Unknown YAML keys are rejected by
    :func:`load_config` (typo guard)."""

    base_model: str
    data_path: str
    hard_types: Sequence[str]
    output_dir: str = "tracedistill_output"
    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Sequence[str] | None = None  # None -> architecture-aware autodetect
    # training
    max_length: int = 8192
    seed: int = 42
    bf16: bool = True
    enable_thinking: bool = True
    phase1: dict[str, Any] = field(default_factory=dict)  # PhaseConfig.train() overrides
    phase2: dict[str, Any] = field(default_factory=dict)  # PhaseConfig.nudge() overrides
    # model loading
    use_unsloth: bool = True
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    attn_implementation: str = "eager"
    # submission packaging
    base_model_name_for_submission: str | None = None  # written into adapter_config.json
    package_zip: bool = True


def load_config(path: str) -> RunConfig:
    """Parse a YAML file into a :class:`RunConfig`, rejecting unknown keys."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path!r} must be a YAML mapping, got {type(raw).__name__}.")
    known = {fld.name for fld in fields(RunConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown config keys {sorted(unknown)} in {path!r}. Known keys: {sorted(known)}."
        )
    if "base_model" not in raw or "data_path" not in raw or "hard_types" not in raw:
        raise ValueError("Config must set 'base_model', 'data_path' and 'hard_types'.")
    return RunConfig(**raw)


def _build_model(cfg: RunConfig):
    """Load the base model + tokenizer and attach a LoRA adapter. Tries Unsloth first
    (matches the competition), falling back to transformers + peft."""
    from .lora import DEFAULT_TARGET_MODULES, target_modules_from_model

    if cfg.use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.base_model,
                max_seq_length=cfg.max_length,
                load_in_4bit=cfg.load_in_4bit,
                load_in_8bit=False,
                full_finetuning=False,
                trust_remote_code=cfg.trust_remote_code,
                attn_implementation=cfg.attn_implementation,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            targets = list(cfg.target_modules) if cfg.target_modules else (
                target_modules_from_model(model) or DEFAULT_TARGET_MODULES
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=targets,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=cfg.seed,
            )
            return model, tokenizer
        except ImportError:
            pass  # fall through to transformers + peft

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # transformers renamed the `torch_dtype` argument to `dtype`; try the new name and
    # fall back to the old one so the library works across the supported transformers range.
    _dt = torch.bfloat16 if cfg.bf16 else None
    _load_kwargs = dict(trust_remote_code=cfg.trust_remote_code, attn_implementation=cfg.attn_implementation)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=_dt, **_load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=_dt, **_load_kwargs)
    targets = list(cfg.target_modules) if cfg.target_modules else (
        target_modules_from_model(model) or DEFAULT_TARGET_MODULES
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return model, tokenizer


def _package_adapter(adapter_dir: str, cfg: RunConfig) -> None:
    """Patch ``adapter_config.json`` for inference + zip the two required files."""
    import json
    import os
    import zipfile

    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path) as f:
        ac = json.load(f)
    if cfg.base_model_name_for_submission:
        ac["base_model_name_or_path"] = cfg.base_model_name_for_submission
    ac["inference_mode"] = True
    ac["lora_dropout"] = 0.0
    with open(cfg_path, "w") as f:
        json.dump(ac, f, indent=2)

    if cfg.package_zip:
        zip_path = os.path.join(adapter_dir, "submission.zip")
        required = ["adapter_config.json", "adapter_model.safetensors"]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in required:
                fpath = os.path.join(adapter_dir, fname)
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Missing {fpath}")
                zf.write(fpath, fname)
        print(f"Wrote {zip_path}")


def run(cfg: RunConfig) -> None:
    """Execute a full training run from a :class:`RunConfig`."""
    import os

    from .data import load_cot_csv
    from .training import PhaseConfig, TwoPhaseConfig, train_two_phase

    model, tokenizer = _build_model(cfg)
    df = load_cot_csv(cfg.data_path)
    two_phase = TwoPhaseConfig(
        hard_types=list(cfg.hard_types),
        output_dir=cfg.output_dir,
        max_length=cfg.max_length,
        seed=cfg.seed,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        phase1=PhaseConfig.train(**cfg.phase1),
        phase2=PhaseConfig.nudge(**cfg.phase2),
        bf16=cfg.bf16,
        enable_thinking=cfg.enable_thinking,
    )
    train_two_phase(model, tokenizer, df, two_phase)

    adapter_dir = os.path.join(cfg.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    _package_adapter(adapter_dir, cfg)
    print(f"Done. Adapter saved to {adapter_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tracedistill",
        description="Two-phase reasoning-trace distillation into a LoRA adapter.",
    )
    parser.add_argument("--cfg", required=True, help="Path to a YAML run config.")
    args = parser.parse_args(argv)
    run(load_config(args.cfg))


if __name__ == "__main__":
    main()
