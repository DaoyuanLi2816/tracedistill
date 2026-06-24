<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/tracedistill/main/docs/banner.svg" alt="tracedistill — distill reasoning traces into a LoRA adapter (NVIDIA Nemotron silver medal)" width="100%">
</p>

<p align="center">
  <a href="https://github.com/DaoyuanLi2816/tracedistill/actions/workflows/ci.yml"><img src="https://github.com/DaoyuanLi2816/tracedistill/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/tracedistill/"><img src="https://img.shields.io/pypi/v/tracedistill" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9+-blue" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License: MIT">
  <a href="https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge"><img src="https://img.shields.io/badge/Kaggle-Silver%20·%2065%2F4163%20(Top%201.6%25)-C0C0C0" alt="Kaggle Silver"></a>
</p>

<h1 align="center">tracedistill</h1>

<p align="center"><b>Distill teacher chains-of-thought into a LoRA adapter — so a model re-derives every answer itself, where no code may run.</b></p>

`tracedistill` is the generalized core of team **VCDAD**'s **silver-medal** solution to the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
(**65 / 4163, Top 1.6%**), extracted into a small, tested library you can run on your own
data. The medal-winning code is preserved verbatim in [`competition/`](competition/) and
[pinned to this library byte-for-byte](tests/) by golden tests.

Give it `(problem, teacher chain-of-thought, answer)` triples and it trains a LoRA adapter
that **reasons step-by-step and then emits a parseable `\boxed{}`** — the recipe that wins
"the grader can't run your code, the solving procedure has to live inside the model's
chain-of-thought" tasks.

---

## Why not just `SFTTrainer` on your traces?

Four design choices, each implemented as a library piece:

1. **A strict format contract** ([`formatting.py`](src/tracedistill/formatting.py)). The SFT
   target is built byte-for-byte identical to the eval protocol —
   `<think> … </think>\boxed{answer}` — and the reasoning (from the teacher trace) is
   **decoupled** from the final answer (rewritten with the *authoritative* label). Train
   input ≈ eval input, so the model reliably boxes a correct answer instead of trailing off.
2. **Two-phase `Train → Nudge`** ([`training.py`](src/tracedistill/training.py)). A hard,
   fast pass (high LR, clipping off) for broad coverage, then a tiny continuation (1/40 LR,
   cosine, clipping on) that squeezes the hard problem types while a balanced sprinkle of
   *fresh* easy data prevents catastrophic forgetting.
3. **Type-stratified batching** ([`sampling.py`](src/tracedistill/sampling.py)). With a tiny
   effective batch, a naive shuffle can make a whole batch one problem type and swing the
   gradient. A round-robin "deal the cards" order keeps every effective batch type-balanced.
4. **Architecture-aware LoRA** ([`lora.py`](src/tracedistill/lora.py)). The competition base
   is a hybrid **Mamba-2 + MoE** model, so targets cover the SSM `in_proj/out_proj` *and*
   attention *and* MLP — the detail a vanilla Llama recipe misses.

```mermaid
flowchart LR
  D["CoT dataset<br/>prompt · cot · answer · type"] --> F["format contract<br/>&lt;think&gt;…&lt;/think&gt;\boxed{}"]
  F --> S["two-phase split<br/>(hard in both)"]
  S --> P1["Phase 1 · Train<br/>lr 2e-4 · clip off"]
  P1 --> P2["Phase 2 · Nudge<br/>lr 5e-6 · cosine · clip on"]
  P2 --> A["LoRA adapter"]
```

## Install

```bash
pip install tracedistill            # light core: numpy / pandas / pyyaml
pip install "tracedistill[train]"   # + torch / transformers / trl / peft / datasets to train
```

The core (`build_records`, the stratified order, the split, target selection, config) is
**torch-free** — it imports and unit-tests without a GPU stack.

## 60 seconds

```python
import tracedistill as td

# Your data: a DataFrame (or list of dicts) with prompt / generated_cot / answer / type.
records, types = td.build_records(df)      # the <think>…</think>\boxed{} format contract
order = td.build_stratified_index_order(types, batch_size=8, seed=42)  # type-balanced order
targets = td.target_modules_from_model(model)   # attention + Mamba SSM + MLP, auto-detected

# The two non-overlapping training sets for Train → Nudge:
phase1_df, phase2_df = td.two_phase_split(df, hard_types=["cryptarithm_deduce"], seed=42)
```

Full two-phase training on an already-LoRA'd model:

```python
from tracedistill import TwoPhaseConfig, PhaseConfig, train_two_phase

cfg = TwoPhaseConfig(hard_types=["cryptarithm_deduce", "cryptarithm_guess"],
                     phase1=PhaseConfig.train(), phase2=PhaseConfig.nudge())
train_two_phase(model, tokenizer, df, cfg)   # Phase 2 continues from Phase 1's weights
```

## CLI

One YAML config drives an end-to-end run (load base model → architecture-aware LoRA →
`Train → Nudge` → save / package the adapter):

```bash
tracedistill --cfg examples/configs/quickstart.yaml             # small single-GPU
tracedistill --cfg examples/configs/reproduce_competition.yaml  # the medal setup (Kaggle)
```

## Measured: does distilling the *trace* actually help? (GSM8K, one RTX 4080)

[`examples/gsm8k_trace_distillation.py`](examples/gsm8k_trace_distillation.py) runs four
arms on **Qwen2.5-0.5B-Instruct + LoRA** through the public API and scores **boxed-answer
accuracy** on held-out GSM8K (greedy, parse `\boxed{}` exactly like a grader). The only
difference between *answer-only SFT* and *trace-distill* is whether a reasoning trace sits
between the `<think>` tags — so the gap isolates the value of distilling the trace.

<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/tracedistill/main/docs/results.svg" alt="GSM8K results: answer-only SFT collapses to 4.5%; trace distillation reaches ~100% parse rate and doubles hard-problem accuracy" width="92%">
</p>

| arm | boxed accuracy | parse rate | hard-problem acc (≥5 steps) |
|---|--:|--:|--:|
| zero-shot (no training) | 32.0% | 64% | 6.1% |
| answer-only SFT | **4.5%** | 100% | 6.1% |
| **trace-distill, 1 phase** | 28.0% | 100% | **12.1%** |
| **trace-distill, 2 phase (Train→Nudge)** | 29.5% | 99.5% | **12.1%** |

**Distil the trace, not the answer.** Answer-only SFT learns to *always* emit a `\boxed{}`
(100% parse rate) but, taught to skip the reasoning, **collapses to 4.5%**. Trace
distillation keeps the reasoning intact and **doubles accuracy on the hard, ≥5-step
problems (6.1% → 12.1%)** while lifting the parse rate from 64% to ~100%.

**The Nudge rebalances.** Phase 2 recovers the mid-tier (3–4-step: 24.8% → 28.7%) without
giving up the hard-tier gain, edging 1-phase 28.0% → 29.5%.

**Honest caveat.** Qwen2.5-0.5B already has strong native GSM8K chain-of-thought, and the
terse GSM8K solutions are a weak teacher, so overall accuracy stays just under the 32%
zero-shot baseline — the wins here are *reliability* (≈100% parse), *hard-slice* accuracy
(2×), and *not* destroying the model the way answer-only SFT does. On the competition's
harder, code-derived puzzles — where the base model can't already solve them and a strong
teacher trace exists — the same recipe took **silver**.

```bash
pip install "tracedistill[train]" datasets
python examples/gsm8k_trace_distillation.py        # ~1h on one RTX 4080 (16 GB)
```

## The competition result

On the hidden test set, the two-phase recipe on `Nemotron-3-Nano-30B-A3B` reached a
**silver medal (65 / 4163, Top 1.6%)**. ~84% of the benchmark is "free" points that almost
everyone clears (gravity, unit conversion, Roman numerals, ciphers); the ranking is decided
by two hard families — **cryptarithm** and **bit-manipulation** — which is exactly what the
two-phase `Nudge` and the hard/easy split target. See [`docs/solution.md`](docs/solution.md),
[`docs/dataset.md`](docs/dataset.md) and [`docs/model-card.md`](docs/model-card.md) for the
full methodology (中文), and [`competition/`](competition/) for the verbatim solution.

## How it compares

| | vanilla `SFTTrainer` | `tracedistill` |
|---|---|---|
| Target format | freeform text | strict `<think>…</think>\boxed{}` contract |
| Answer source | as written in the trace | decoupled — official label re-boxed |
| Schedule | single pass | two-phase `Train → Nudge` |
| Batching | shuffle | type-stratified round-robin |
| LoRA targets | attention (+ MLP) | + **Mamba-2 SSM** `in_proj/out_proj` |

## Provenance & validation

- [`competition/`](competition/) — the original silver-medal solution, **unmodified**.
- [`tests/`](tests/) — **golden tests**: `tests/reference_impl.py` holds verbatim copies of
  the competition's `build_records` / `build_stratified_index_order`, and the suite asserts
  `tracedistill` reproduces them **byte-for-byte** over hundreds of fuzzed cases. 46 tests,
  torch-free, run in well under a second.

## 中文简介

`tracedistill` 是 **VCDAD 队** Kaggle [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) **银牌方案（65/4163，Top 1.6%）** 的可复用内核,打包成一个带测试的开源库。给它 `(题目, 教师思维链, 答案)`,它训练出一个 LoRA adapter,让模型**自己一步步推理、再输出可解析的 `\boxed{}`** —— 应对"评测端不能跑代码、解题算法必须蒸馏进思维链"的任务。

核心四件套:**格式契约**(`<think>…</think>\boxed{}` 与评测逐字一致,推理用上游 CoT、答案用官方标签解耦)、**两阶段 Train→Nudge**(大 lr 铺开 + 1/40 lr 精修难题防遗忘)、**类型分层采样**、**架构感知 LoRA**(覆盖 Mamba 的 `in_proj/out_proj`)。原始竞赛代码逐字保留在 [`competition/`](competition/),并由 [`tests/`](tests/) 的 golden 测试逐字节锁定。完整中文方法论见 [`docs/`](docs/)。

## Citation

```bibtex
@misc{li2026tracedistill,
  title  = {tracedistill: Two-Phase LoRA Trace-Distillation for Reasoning Models},
  author = {Li, Daoyuan},
  year   = {2026},
  note   = {Silver medal (65/4163), NVIDIA Nemotron Model Reasoning Challenge},
  url    = {https://github.com/DaoyuanLi2816/tracedistill}
}
```

## License

[MIT](LICENSE). The license covers the code and documentation in this repository; it does
**not** extend to the competition data or the base model, which remain under their
respective terms (see [`data/README.md`](data/README.md)).
