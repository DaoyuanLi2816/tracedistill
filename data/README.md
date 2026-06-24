# Data

The competition's data and base model belong to NVIDIA / Kaggle and are **not
redistributed here** (per the competition rules). Download them from the sources below.

## Where to get it

| Resource | Link |
|---|---|
| Competition (official data) | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge |
| Base model (`Nemotron-3-Nano-30B-A3B-BF16`) | https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16 |
| Offline wheels (`mamba_ssm`, `unsloth`, …) | https://www.kaggle.com/datasets/mayukh18/nemotron-packages |
| **CoT training data** (`type` + `generated_cot`) | https://www.kaggle.com/datasets/dgxchen/nemotron-cot-tong |

The original `competition/training.py` reads the **community CoT dataset** (last row), not
the official `train.csv` — that file has only `id`/`prompt`/`answer` (no problem-type label
and no chain-of-thought), whereas the CoT dataset adds the `type` label and the
pre-generated `generated_cot` reasoning trace used as the distillation target.

## Expected schema (for `tracedistill`)

The `tracedistill` library and the `examples/` configs expect a CSV with these columns
(see `tracedistill.data.CANONICAL_COLUMNS`):

| Column | Meaning |
|---|---|
| `prompt` | the problem statement |
| `generated_cot` | a teacher chain-of-thought / reasoning trace to distill |
| `answer` | the authoritative final answer (rewritten into `\boxed{}`) |
| `type` | a problem-family label used for stratified batching + the hard/easy split |

The headline experiment (`examples/gsm8k_trace_distillation.py`) builds exactly this
schema from the public **GSM8K** dataset, so it needs no Kaggle download.

## Attribution

The competition data and base model belong to NVIDIA / the Kaggle competition. For any use,
follow the [competition rules](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules).
