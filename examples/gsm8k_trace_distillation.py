"""Headline experiment — does trace distillation beat answer-only SFT?

Runs four arms on **Qwen2.5-0.5B-Instruct + LoRA**, on a single RTX 4080 (16 GB), and
measures **boxed-answer accuracy** on held-out **GSM8K** (greedy decode, parse
``\\boxed{}`` exactly like the competition grader):

  1. **zero-shot**            — the base model, no training
  2. **answer-only SFT**      — distil ``question -> \\boxed{answer}`` with NO reasoning
                                (the failure mode: teaches the model to skip thinking)
  3. **trace-distill, 1 phase** — tracedistill's format contract over teacher CoT
  4. **trace-distill, 2 phase** — the two-phase Train -> Nudge schedule

The only difference between arms 2 and 3 is whether a reasoning *trace* sits between the
``<think>`` tags, so the gap isolates the value of distilling the trace.

Everything goes through tracedistill's public API (``build_records``, ``two_phase_split``,
``make_formatting_func``, ``train_two_phase``), so this doubles as a usage example.

Reproduce (~1h on one RTX 4080):
    pip install -e .[train] datasets
    python examples/gsm8k_trace_distillation.py            # full run
    python examples/gsm8k_trace_distillation.py --smoke    # tiny, ~2 min sanity run
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import tracedistill as td
from tracedistill.training import PhaseConfig, TwoPhaseConfig, make_formatting_func, train_two_phase

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_GSM_GOLD = re.compile(r"####\s*([\-0-9,\.]+)")


# ----------------------------------------------------------------------------- data
def _num(s: str):
    """Parse a loosely-formatted number ('1,234', '$5', '7.0') to float, or None."""
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


def _step_bucket(solution: str) -> str:
    """Difficulty proxy = number of <<...>> calculator steps in the GSM8K solution."""
    n = solution.count("<<")
    if n <= 2:
        return "steps_le2"
    if n <= 4:
        return "steps_3_4"
    return "steps_ge5"


def load_gsm8k_df(split: str, limit: int | None, seed: int) -> pd.DataFrame:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    rows = []
    for ex in ds:
        gold = _GSM_GOLD.search(ex["answer"])
        if not gold:
            continue
        answer = gold.group(1).replace(",", "").strip()
        # teacher trace = the worked solution with the "#### N" line and <<>> annotations removed
        cot = _GSM_GOLD.sub("", ex["answer"]).strip()
        cot = re.sub(r"<<[^>]*>>", "", cot)
        rows.append(
            {
                "prompt": ex["question"],
                "generated_cot": cot,
                "answer": answer,
                "type": _step_bucket(ex["answer"]),
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- model io
def fresh_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    return model, tok


def attach_lora(model, seed):
    targets = td.target_modules_from_model(model) or td.DEFAULT_TARGET_MODULES
    return get_peft_model(
        model,
        LoraConfig(
            r=32, lora_alpha=32, lora_dropout=0.0,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        ),
    )


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, tok, test_df: pd.DataFrame, max_new_tokens: int) -> dict:
    model.eval()
    correct = parsed = 0
    by_type_total: dict[str, int] = {}
    by_type_correct: dict[str, int] = {}
    for _, row in test_df.iterrows():
        messages = [{"role": "user", "content": str(row["prompt"]) + td.DEFAULT_PROMPT_SUFFIX}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        boxes = _BOXED.findall(gen)
        t = str(row["type"])
        by_type_total[t] = by_type_total.get(t, 0) + 1
        if boxes:
            parsed += 1
            pred, gold = _num(boxes[-1]), _num(str(row["answer"]))
            if pred is not None and gold is not None and abs(pred - gold) < 1e-2:
                correct += 1
                by_type_correct[t] = by_type_correct.get(t, 0) + 1
    n = len(test_df)
    return {
        "accuracy": correct / n,
        "parse_rate": parsed / n,
        "n": n,
        "by_type_acc": {t: by_type_correct.get(t, 0) / by_type_total[t] for t in sorted(by_type_total)},
    }


# --------------------------------------------------------------------------- training
def _sft(model, tok, records, lr, epochs, bs, out):
    ds = HFDataset.from_list(records)
    args = SFTConfig(
        output_dir=out, num_train_epochs=epochs, per_device_train_batch_size=bs,
        gradient_accumulation_steps=2, learning_rate=lr, lr_scheduler_type="linear",
        warmup_steps=0, max_length=1024, bf16=True, logging_steps=25, save_strategy="no",
        report_to="none", packing=False, remove_unused_columns=False, dataloader_num_workers=0,
        neftune_noise_alpha=5.0, max_grad_norm=1e9,
    )
    SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
               formatting_func=make_formatting_func(tok)).train()


def answer_only_records(df) -> list[dict]:
    """Same boxed format, but with NO reasoning trace between the <think> tags."""
    recs = []
    for _, row in df.iterrows():
        recs.append({"messages": [
            {"role": "user", "content": str(row["prompt"]) + td.DEFAULT_PROMPT_SUFFIX},
            {"role": "assistant", "content": f"</think>\n\\boxed{{{row['answer']}}}"},
        ]})
    return recs


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=1500)
    ap.add_argument("--test-size", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out", default="output/gsm8k")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run")
    args = ap.parse_args()
    if args.smoke:
        args.train_size, args.test_size, args.epochs, args.max_new_tokens = 60, 20, 1, 200
    os.makedirs(args.out, exist_ok=True)
    seed = 42

    train_df = load_gsm8k_df("train", args.train_size, seed)
    test_df = load_gsm8k_df("test", args.test_size, seed)
    hard_types = ["steps_ge5"]
    print(f"train={len(train_df)} test={len(test_df)} | types={dict(train_df['type'].value_counts())}")

    results: dict[str, dict] = {}

    # Arm 1 — zero-shot
    model, tok = fresh_model_and_tokenizer()
    results["zero_shot"] = evaluate(model, tok, test_df, args.max_new_tokens)
    del model; torch.cuda.empty_cache()
    print("zero_shot:", results["zero_shot"])

    # Arm 2 — answer-only SFT (the failure mode)
    model, tok = fresh_model_and_tokenizer()
    model = attach_lora(model, seed)
    _sft(model, tok, answer_only_records(train_df), 2e-4, args.epochs, 4, f"{args.out}/answer_only")
    results["answer_only_sft"] = evaluate(model, tok, test_df, args.max_new_tokens)
    del model; torch.cuda.empty_cache()
    print("answer_only_sft:", results["answer_only_sft"])

    # Arm 3 — trace-distill, single phase (the format contract over teacher CoT)
    model, tok = fresh_model_and_tokenizer()
    model = attach_lora(model, seed)
    records, _ = td.build_records(train_df)
    _sft(model, tok, records, 2e-4, args.epochs, 4, f"{args.out}/trace_1phase")
    results["trace_distill_1phase"] = evaluate(model, tok, test_df, args.max_new_tokens)
    del model; torch.cuda.empty_cache()
    print("trace_distill_1phase:", results["trace_distill_1phase"])

    # Arm 4 — trace-distill, two-phase Train -> Nudge (via the library)
    model, tok = fresh_model_and_tokenizer()
    model = attach_lora(model, seed)
    cfg = TwoPhaseConfig(
        hard_types=hard_types, output_dir=f"{args.out}/trace_2phase", max_length=1024, seed=seed,
        phase1=PhaseConfig.train(num_train_epochs=args.epochs, per_device_train_batch_size=4),
        phase2=PhaseConfig.nudge(num_train_epochs=1, per_device_train_batch_size=4),
    )
    train_two_phase(model, tok, train_df, cfg)
    results["trace_distill_2phase"] = evaluate(model, tok, test_df, args.max_new_tokens)
    del model; torch.cuda.empty_cache()
    print("trace_distill_2phase:", results["trace_distill_2phase"])

    # ---- report ----
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    order = ["zero_shot", "answer_only_sft", "trace_distill_1phase", "trace_distill_2phase"]
    label = {
        "zero_shot": "zero-shot (no training)",
        "answer_only_sft": "answer-only SFT",
        "trace_distill_1phase": "trace-distill, 1 phase",
        "trace_distill_2phase": "trace-distill, 2 phase (Train→Nudge)",
    }
    print("\n| arm | boxed acc | parse rate |")
    print("|---|--:|--:|")
    for k in order:
        r = results[k]
        print(f"| {label[k]} | {r['accuracy']*100:.1f}% | {r['parse_rate']*100:.1f}% |")
    print(f"\nWrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
