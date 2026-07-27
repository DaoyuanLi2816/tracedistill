# GSM8K reproducibility

The repository's public ablation is designed to answer two questions:

1. Does supervising a worked reasoning trace outperform supervising only the final
   boxed answer?
2. Does the competition's hard-focused Train → Nudge schedule remain competitive with
   a single pass on this small public transfer task?

## Controlled setup

| setting | value |
|---|---|
| dataset | `openai/gsm8k`, official train and test splits |
| base model | `Qwen/Qwen2.5-0.5B` (non-instruction-tuned) |
| train / test | 1,200 / 200, deterministically shuffled with seed 42 |
| adapter | LoRA rank 32, alpha 32, dropout 0 |
| training | 3 epochs, effective batch 8, BF16 |
| evaluation | greedy boxed-answer accuracy, 320 generated tokens, batch 8 |
| hardware | one NVIDIA RTX 4080 16 GB |

All trained arms use completion-only labels: prompt tokens and any token crossing the
prompt/completion boundary receive `-100`. The answer-only and one-phase trace arms use
the same optimizer settings and differ only in whether a worked trace appears before the
boxed label. The two-phase arm keeps the same effective batch size, reserves fresh easy
examples for Nudge, and repeats the hard examples across both phases.

The script fixes the initialization seed before every arm. Its machine-readable result
records the exact package versions, CUDA version, GPU, commit, dataset fingerprints, and
configuration.

## Run

```bash
python -m pip install -e ".[train]"
python examples/gsm8k_trace_distillation.py
```

A tiny end-to-end check is also available:

```bash
python examples/gsm8k_trace_distillation.py --smoke
```

The smoke run validates the pipeline but is not used for conclusions.

The release result is stored in
[`results/gsm8k-qwen2.5-0.5b-v0.2.0.json`](../results/gsm8k-qwen2.5-0.5b-v0.2.0.json).
This is a single-seed, 200-example evaluation of a 0.5B model; it is intended as a
compact public transfer test alongside the independently scored Kaggle medal result.
