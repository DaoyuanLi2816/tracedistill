# Data

## Files in this folder

| File | What it is |
|---|---|
| `train.csv` | The **official competition training set** — 9,500 rows, 3 columns: `id`, `prompt`, `answer`. Note it contains **no problem-type label and no chain-of-thought**; the type/rule of each puzzle has to be reverse-engineered. |
| `test.csv` | The small **sample** test file shipped with the competition. It exists only to debug the submission format — at grading time it is replaced by the hidden test set ("hundreds" of problems). |

## What the training script actually reads

`code/training.py` does **not** read the official `train.csv`. It reads a community dataset that adds two columns the official file lacks — a `type` label and a pre-generated `generated_cot` (the reasoning trace used as the distillation target):

| Resource | Link |
|---|---|
| Competition (official data) | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge |
| Base model | https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1 |
| Offline wheels (`mamba_ssm`, `unsloth`, …) | https://www.kaggle.com/datasets/mayukh18/nemotron-packages |
| CoT training data (`type` + `generated_cot`) | https://www.kaggle.com/datasets/dgxchen/nemotron-cot-tong |

## Attribution

The competition data and base model belong to NVIDIA / the Kaggle competition and are included here for reproducibility and reference only. For any further use, follow the
[competition rules](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules).
