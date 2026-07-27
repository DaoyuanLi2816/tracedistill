"""Config-driven CLI: ``tracedistill --cfg config.yaml`` (or ``python -m tracedistill.run``).

Loads a base model, attaches an architecture-aware LoRA adapter, reads a CoT dataset,
runs the two-phase ``Train → Nudge`` schedule, and saves (and optionally packages) the
resulting adapter. ``load_config`` and ``RunConfig`` are import-safe without torch so the
config surface can be unit-tested; the heavy model code is imported lazily inside
:func:`run`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import yaml

__all__ = ["RunConfig", "load_config", "summarize_data", "run", "main"]

_PHASE_FIELDS = {
    "learning_rate",
    "lr_scheduler_type",
    "warmup_steps",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_grad_norm",
    "neftune_noise_alpha",
    "logging_steps",
}


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
    use_unsloth: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
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
    required = {"base_model", "data_path", "hard_types"}
    if not required.issubset(raw):
        raise ValueError("Config must set 'base_model', 'data_path' and 'hard_types'.")
    for phase_name in ("phase1", "phase2"):
        phase = raw.get(phase_name, {})
        if not isinstance(phase, dict):
            raise ValueError(f"{phase_name!r} must be a YAML mapping.")
        unknown_phase = set(phase) - _PHASE_FIELDS
        if unknown_phase:
            raise ValueError(
                f"Unknown {phase_name} keys {sorted(unknown_phase)}; "
                f"known keys: {sorted(_PHASE_FIELDS)}."
            )
    cfg = RunConfig(**raw)
    if not cfg.base_model.strip() or not cfg.data_path.strip():
        raise ValueError("base_model and data_path must not be empty.")
    if not cfg.hard_types or any(not str(value).strip() for value in cfg.hard_types):
        raise ValueError("hard_types must contain at least one non-empty value.")
    if cfg.max_length <= 0:
        raise ValueError("max_length must be positive.")
    if cfg.lora_rank <= 0 or cfg.lora_alpha <= 0:
        raise ValueError("lora_rank and lora_alpha must be positive.")
    return cfg


def summarize_data(cfg: RunConfig) -> dict[str, Any]:
    """Validate a run's dataset and return a GPU-free preflight summary."""
    from .data import load_cot_csv, two_phase_split
    from .formatting import build_records

    df = load_cot_csv(cfg.data_path)
    records, types = build_records(df)
    phase1_df, phase2_df = two_phase_split(df, cfg.hard_types, seed=cfg.seed)
    return {
        "data_path": cfg.data_path,
        "rows": len(df),
        "usable_records": len(records),
        "dropped_records": len(df) - len(records),
        "types": {str(key): int(value) for key, value in df["type"].value_counts().items()},
        "phase1_rows": len(phase1_df),
        "phase2_rows": len(phase2_df),
        "phase2_types": {
            str(key): int(value) for key, value in phase2_df["type"].value_counts().items()
        },
        "hard_types": list(cfg.hard_types),
        "completion_only_loss": True,
        "max_length": cfg.max_length,
    }


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
            targets = (
                list(cfg.target_modules)
                if cfg.target_modules
                else (target_modules_from_model(model) or DEFAULT_TARGET_MODULES)
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

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # transformers renamed the `torch_dtype` argument to `dtype`; try the new name and
    # fall back to the old one so the library works across the supported transformers range.
    _dt = torch.bfloat16 if cfg.bf16 else None
    _load_kwargs = dict(
        trust_remote_code=cfg.trust_remote_code, attn_implementation=cfg.attn_implementation
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=_dt, **_load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=_dt, **_load_kwargs
        )
    targets = (
        list(cfg.target_modules)
        if cfg.target_modules
        else (target_modules_from_model(model) or DEFAULT_TARGET_MODULES)
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
    import os
    import zipfile

    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        ac = json.load(f)
    if cfg.base_model_name_for_submission:
        ac["base_model_name_or_path"] = cfg.base_model_name_for_submission
    ac["inference_mode"] = True
    ac["lora_dropout"] = 0.0
    with open(cfg_path, "w", encoding="utf-8") as f:
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and data split without loading a model.",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.cfg)
    if args.dry_run:
        print(json.dumps(summarize_data(cfg), indent=2))
        return
    run(cfg)


if __name__ == "__main__":
    main()
