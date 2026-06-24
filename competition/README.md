# Competition solution (provenance)

> This directory preserves team **VCDAD**'s original, **unmodified** Kaggle solution to
> the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
> — **Silver Medal, 65 / 4182 (Top 1.6%)**.
>
> The reusable core of this code has been extracted into the **[`tracedistill`](../README.md)**
> library at the repository root. The library is pinned to this original here by the
> golden/characterization tests in [`../tests/`](../tests/), which assert that
> `tracedistill` reproduces these functions **byte-for-byte** (see
> `tests/reference_impl.py`). This is the medal-winning code, not a reimplementation.

## What's here

| File | What it is |
|---|---|
| [`training.py`](training.py) | The full single-script solution, exactly as run on Kaggle. Authored as Jupyter `# %%` cells: offline wheel install + Blackwell `ptxas` patch → load `Nemotron-3-Nano-30B-A3B` (BF16) + attach LoRA → two-phase `Train → Nudge` fine-tuning → package `submission.zip`. |

## The approach in one paragraph

Two-phase LoRA **trace-distillation SFT** on the frozen, competition-fixed
`NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` base (a hybrid **Mamba-2 + MoE** model). Each
problem's teacher chain-of-thought is wrapped into a `<think>…</think>\boxed{answer}`
**format contract** that is byte-for-byte identical to the grader's protocol, with the
reasoning taken from the upstream CoT but the final `\boxed{}` rewritten with the
*authoritative* answer. Phase 1 (**Train**, lr `2e-4`, gradient clipping off) covers all
problem types fast; Phase 2 (**Nudge**, lr `5e-6`, cosine, clipping on) squeezes the hard
types while a balanced sprinkle of fresh easy problems prevents forgetting. LoRA is
architecture-aware (covers the Mamba `in_proj/out_proj`, not just attention), batches are
type-stratified, and training is pure BF16 with NEFTune. Because the grader cannot run
code, the *solving procedure itself* is distilled into the model's chain-of-thought.

See the in-depth write-ups in [`../docs/`](../docs/) (`solution.md`, `dataset.md`,
`model-card.md`) for the full methodology.

## Environment note

`training.py` targets the competition's Kaggle environment (a Blackwell GPU, **offline**):
it installs Triton / `unsloth` / `trl` / `peft` / `mamba_ssm` / `causal_conv1d` from
mounted wheel datasets and patches in a Blackwell-capable `ptxas`. The `tracedistill`
library at the root strips that Kaggle-specific plumbing so the same recipe runs on an
ordinary single-GPU box (see [`../examples/`](../examples/)).
