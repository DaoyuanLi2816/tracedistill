# NVIDIA Nemotron Model Reasoning Challenge Solution

---

## 1. Understanding the Competition

### 1.1 Background and Goal

**The NVIDIA Nemotron Model Reasoning Challenge (NVIDIA-NMRC for short)** was a Kaggle competition hosted by NVIDIA Research in 2026 (registration opened 2026-03-16, submissions closed 2026-06-15, with a final field of **4182 teams**).

The central premise of the competition is this: **on top of a single fixed foundation model shared by everyone, use technical means to improve its solve-rate accuracy on a brand-new reasoning benchmark.**

- **Shared foundation**: every participant uses the same open-source model, **NVIDIA-Nemotron-3-Nano-30B-A3B** (see §1.3), and cannot swap it out. This way the competition is about *technique*, not "whose base model is stronger."
- **The single hard requirement**: the final submission must be a **LoRA adapter compatible with that foundation model** (rank ≤ 32), packaged as `submission.zip`.
- **An open technical playing field**: the organizers permit any direction — prompt engineering, data filtering and selection, synthetic data generation, reinforcement learning, lightweight fine-tuning, and so on — with no restriction on framework (Hugging Face / Unsloth / TRL / Axolotl all allowed).

The input and output in one sentence:

> **Input**: a batch of procedurally generated logic puzzles (each puzzle gives several "input → output" examples; you reverse-engineer the hidden rule, then apply it to a query that needs solving).
> **Output**: a LoRA adapter; at evaluation time the organizers use it to **run inference live** on a hidden test set, and the model must **work out each puzzle itself** within the chain-of-thought it generates.

### 1.2 Why This Competition Is "Different"

This competition differs from an ordinary Kaggle contest, and from ordinary SFT, in a few fundamental ways that dictate the entire approach:

The single most important point: **you cannot run Python at evaluation time.** Solving the training puzzles 100% with code locally is useless; what matters is writing the "solving process" as a chain-of-thought (CoT), distilling it into a LoRA via SFT, and getting the model to **reproduce that process itself** at inference time to compute the answer.

### 1.3 The Foundation Model: Nemotron-3-Nano-30B-A3B

Only by understanding this model can you understand why all of this competition's techniques look the way they do.

Breaking down `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`:

- **30B / A3B**: **30B total parameters, but only about 3.5B activated per token** (A3B = Active 3B). This is the hallmark of a **MoE (Mixture-of-Experts)**: each MoE layer has 128 + 1 experts, and the router picks only 6 per token to compute with. "Capacity like a mid-size model, compute like a small one."
- **Hybrid architecture**: it is **not a pure Transformer**. The backbone is **23 layers of Mamba-2 (a state space model, SSM, whose complexity grows linearly with sequence length)**, interleaved with just **6 Attention layers**. This is what lets it claim a native **1 million token** context.
- **A native reasoning model**: by default it **first emits a `<think>...</think>` chain-of-thought, then gives the conclusion**. This is precisely the bedrock of this competition's "teach it to write solving traces via SFT" approach.
- **BF16**: shipped as bfloat16; the competition loads it in BF16 and does **not quantize**.

These three properties directly shape the engineering battlefield of the competition:

1. **A native 1M context, yet hard-capped by the competition to 8192 (generation ≤ 7680 tokens).** As a result, "can the solving process be squeezed into the token budget" matters just as much as "is the answer correct" → which gives rise to a whole series of techniques: HEX compression, signature catalogs, "memorize vs. compute" trade-offs, and more.
2. **The Mamba component → painfully slow training and a finicky environment.** Mamba relies on dedicated CUDA kernels such as `mamba_ssm` and `causal_conv1d`, which have poor compatibility with Kaggle's new **Blackwell GPUs**. Kaggle's 3-hour limit allows training only about 20 million tokens, whereas the 1st-place team used their own workstation to train **hundreds of millions of tokens over hundreds of hours** — a substantial compute barrier.
3. **MoE → noisy scoring.** Even with `temperature=0` (greedy decoding), the non-determinism of expert routing plus floating-point accumulation causes the same adapter's score to jitter slightly across repeated evaluations. **Conclusion: you must rely on local stratified CV, not just the Public LB.**
4. **Its tokenizer is "token-hungry"** (a hands-on observation from the 1st-place team): close to "one digit / one binary character = one token." On hard puzzles full of binary and symbols, just writing out the problem statement is extremely token-expensive, so compressing the trace is a hard requirement.

## 2. Dataset Overview

### 2.1 Fields and Scale

The official training set `train.csv` has only **3 columns** and **9500 puzzles** in total:

| Column | Meaning |
|---|---|
| `id` | Unique identifier for the puzzle |
| `prompt` | The problem statement: several "input → output" examples + one query to solve |
| `answer` | The reference answer |

⚠️ **Key fact: the official `train.csv` contains neither a "puzzle category" nor a "chain-of-thought."** Which class a puzzle belongs to and what its hidden rule is must all be reverse-engineered by the participant. The `training.py` in this solution does **not** read the official csv; instead it reads a **community third-party dataset**, `dgxchen/nemotron-cot-tong`, whose fields are `type, prompt, answer, generated_cot`. The extra `generated_cot` column provides a pre-prepared solving chain-of-thought for each puzzle — precisely the raw material for the trace distillation in this solution. This is crucial, and §4.1 expands on it.

### 2.2 The Seven Puzzle Categories and Difficulty Distribution

The training set splits into **7 major categories (9 sub-classes)**, with extremely uneven difficulty:

| Category | Count | Approx. solve rate of top solutions | Nature |
|---|---:|---:|---|
| gravity | 1597 | 100% | Free points |
| unit conversion | 1594 | 100% | Free points |
| numeral system (Roman numerals) | 1576 | 100% | Free points |
| text cipher | 1576 | ~99–100% | Free points |
| bit manipulation | 1602 | ~93–97% | ⭐ One of the battlegrounds |
| equation numeric | 732 (deduce 596 + guess 136) | deduce ~95% / guess ~50% | Medium |
| cryptarithm | 823 (deduce 659 + guess 164) | deduce ~30% / guess ~10% | ⭐⭐ The decisive class |
| **Total** | **9500** | | |

The hidden rule of each category in one line (all include the "Alice's Wonderland" story wrapper, which must be ignored):

- **gravity**: the formula `d = 0.5·g·t²` is handed to you, but `g` is random per puzzle; reverse-solve `g` from the examples.
- **unit conversion**: `output = input × coefficient`, with the coefficient random per puzzle (isomorphic to gravity, just linear).
- **numeral system**: integer ↔ Roman numeral (**a string puzzle; must be character-for-character exact**).
- **text cipher**: a 26-letter bijective substitution cipher. The clever part: **the entire dataset's plaintext uses only 77 fixed words**, turning an open search into a closed one.
- **bit manipulation**: 8-bit input → 8-bit output, where each output bit is a boolean function of several input bits (AND/OR/XOR/NOT/MAJ…). The model **cannot compute bit operations in parallel** and must **write them out bit by bit**, or accuracy plunges to ~9%.
- **equation numeric**: `two-digit operator two-digit`, where **the visual meaning of the operator symbol is noise**; the real rule must be inferred from the example outputs (e.g. `` ` `` might be subtraction, `+` might be multiplication).
- **cryptarithm**: equation numeric with **another layer of symbolic encryption** on top: even the digits themselves are replaced by symbols, so you must simultaneously crack three things — "symbol ↔ digit," "symbol ↔ operator rule," and "re-encode the result back into symbols." **Ceiling-level difficulty.**

**deduce vs. guess (runs through the last two categories; make sure you grasp this)**: the difference is just one sentence — whether the operator used in the query was actually demonstrated in the examples above. Demonstrated → deduce (copy it directly); never appears even once → guess (you must infer which family it belongs to using structural constraints like "each operator family is used only once"). guess is far harder than deduce, and it is the sub-class that truly separates the field.

### 2.3 The Test Set and "the 0.86 Wall"

- **Test set**: the `test.csv` provided in the competition is just a handful of sample puzzles (for debugging the submission format); official scoring swaps in the full hidden test set of **"hundreds of puzzles,"** split into a Public LB (visible during the competition) and a Private LB (final ranking). **The Private LB is what truly decides the ranking.**
- **Test distribution ≈ training distribution** (not stated explicitly by the organizers, but the evidence is strong: same procedural "factory" generation, and the 3rd-place team reported that their stratified local CV on the training set correlated highly with the Private LB). **So you can confidently use stratified local CV on the training set as a proxy for the Private LB.**
- **Each puzzle is run once**: `temperature=0.0` greedy, each puzzle generated exactly once, one shot to settle it — no retries, no multi-sampling. **Implication:** since you cannot vote at evaluation time, "verification / self-correction" must be **written into the chain-of-thought** (the model checks its own work, spots a contradiction, and backs out).

Understanding "the 0.86 wall" is the key to understanding the whole competition:

> The first 5 free-point categories + the easy part of bit manipulation add up to about **84%**, and almost everyone can get close to a perfect score here; this part is worth roughly **0.85–0.86**, and everyone is clustered right there.
> **The 3 percentage points that take you from 0.86 to 0.89 come almost entirely from cryptarithm and bit manipulation.** Top teams spend all their effort on these two categories.

---

## 3. Evaluation Metric

The competition uses **accuracy** — the fraction of correctly answered puzzles out of the total:

$$
\text{Accuracy} = \frac{\#\{\text{correctly predicted puzzles}\}}{\#\{\text{all puzzles}\}}
$$

The scoring pipeline (run on the organizers' back end, not in participant scripts):

1. **Load**: the vLLM inference engine loads the Nemotron-3-Nano-30B base model + your submitted LoRA adapter (which must include `adapter_config.json`).
2. **Generate**: for each puzzle, prompt the model to answer and require it to write the final answer into `\boxed{}`.
3. **Extract**: pull the answer from the generated text — **parse the contents of `\boxed{}` first**, then heuristic rules, and finally fall back to "the last numeric value."
4. **Judge**: the prediction is counted correct if it is an **exact string match** with the ground truth, or its **relative error is within $10^{-2}$**.

Fixed evaluation parameters (be sure to align training with these):

| Parameter | Value | Meaning |
| :--- | :--- | :--- |
| `max_lora_rank` | 32 | Upper bound on the adapter's rank (hard constraint) |
| `max_tokens` | 7680 | **Per-puzzle generation cap**; the trace must fit within this budget |
| `top_p` | 1.0 | Has no effect under greedy decoding |
| `temperature` | 0.0 | Greedy decoding, once per puzzle |
| `max_num_seqs` | 64 | vLLM batching concurrency (**not** 64 attempts per puzzle) |
| `gpu_memory_utilization` | 0.85 | Fraction of GPU memory vLLM occupies |
| `max_model_len` | 8192 | Context window (training `max_seq_length` is aligned with it) |

**Submission form**: a LoRA adapter with rank ≤ 32, packaged as `submission.zip`, containing only the two files `adapter_config.json` + `adapter_model.safetensors` (**no base weights**; the evaluator brings its own base model).

> Two intuitions about the metric: (1) **numeric puzzles have a 1e-2 tolerance**, so computing to two decimal places is enough, but the answer must be written in `X.XX` format; (2) **string puzzles (Roman numerals, ciphers) have zero tolerance** — one wrong character is 0 — so the solution has the model "split into pieces → concatenate → read back to verify."

---

## 4. Solution Walkthrough

> This section follows the order of the execution pipeline, presenting the **two-phase fine-tuning, Train → Nudge**, implemented in this solution's `competition/training.py`, with the emphasis on "why it's designed this way."

### 4.1 Top-Level Mental Model: Trace-Distillation SFT

Before diving into the code, grasp what the whole approach is doing:

> **Take data that "already comes with a solving chain-of-thought (CoT)," use SFT to pour that chain-of-thought + the final answer into a LoRA adapter, and get the base model to reproduce it at inference time.**

The training data uses the public dataset `dgxchen/nemotron-cot-tong`, whose fields are `type, prompt, answer, generated_cot`, where `generated_cot` already provides a solving chain-of-thought for each puzzle — that is the raw material for distillation. All the approach has to do is organize "problem statement → chain-of-thought + answer" into SFT samples and fine-tune a LoRA that can reproduce the solving process at inference time.

### 4.2 Solution Pipeline Overview

> Open `pipeline.svg` directly in your browser for the high-resolution version.

### 4.3 Phase-by-Phase Walkthrough

#### Phase 0: Configuration and Environment Patches

First, a few lines of core configuration:

```python
HARD_TYPES = {"cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_guess"}
LORA_RANK = 32 ; LORA_ALPHA = 32 ; LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj",
                  "in_proj","out_proj","up_proj","down_proj"]  # lm_head is commented out
```

- **`HARD_TYPES` is the core judgment of the whole approach.** It matches the dataset conclusion: what decides the ranking is cryptarithm (both sub-classes are hard) + the guess sub-class of equation_numeric. Note that `equation_numeric_deduce` is classified as easy (it can copy the pattern straight from the context, so it doesn't count as hard). This set determines **which puzzles are trained in full across both phases and which are trained only in the first phase**.
- **`rank=32`** is pushed to the competition's upper limit; the larger the capacity, the more it can "memorize" what needs to be remembered (signature tables, patterns).
- **`dropout=0`** turns off dropout: SFT distillation wants "exact replication of the trace," so it would rather overfit than learn imprecisely.
- **The presence of `in_proj` / `out_proj` in `target_modules`** is a key, architecture-tied detail: they are the **input/output projections of the Mamba-2 (SSM) blocks**, which a vanilla Llama LoRA recipe does not include. Here LoRA is attached to both the Attention and the Mamba layers, for broader coverage.

The environment patches (3 cells) are pure engineering trench-work: installing the Triton wheel offline, **patching ptxas for the Blackwell GPU** (even monkey-patching it to lie about its version number as `'12.0'`), and installing `mamba_ssm` / `causal_conv1d` offline. Their very existence is a piece of information: **a new model (released only in 2025-12) + new hardware (Blackwell) + no internet access = whoever gets the environment running first wins at the starting line.**

#### Phase 1: Data Split (into two non-overlapping sets)

```python
n = int(hard_df["type"].value_counts().min())   # the count of the scarcest of the three hard types
# Randomly sample n rows from each easy type to reserve for Nudge; everything else + all hard go to Epoch1
```

| | **Epoch1 set (used by Phase 1)** | **Nudge set (used by Phase 2)** |
|---|---|---|
| Hard puzzles | ✅ **All** | ✅ **All** |
| Easy puzzles | ✅ All remaining after the `n × (number of easy types)` rows are taken out | ✅ Exactly `n` rows per easy type |

Design intent: (1) **hard puzzles appear in full in both phases**, reinforced repeatedly; (2) easy puzzles are split into two non-overlapping halves, so **Phase 2 sees fresh easy puzzles that Phase 1 never trained on** (guarding against forgetting without repeatedly overfitting); (3) the Nudge set is **type-balanced**, well suited for a small-step, balanced finishing pass.

#### Phase 2: Constructing the SFT Samples — `build_records` (the function most worth reading word for word in the entire write-up)

One wrong character in the format and answer extraction at inference time can score 0. This function establishes the **"format contract" of the training target**:

```python
cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()        # ② strip the \boxed that the original CoT came with
user_content = str(row["prompt"]) + PROMPT_SUFFIX                   # ③ problem statement + mandatory suffix (aligns with eval)
asst_content = cot_cleaned + f"\n</think>\n\\boxed{{{row['answer']}}}"  # ④ re-attach \boxed using the official answer
```

A complete training target looks like this:

```
<think>                         ← added automatically by the chat template (enable_thinking=True)
(upstream chain-of-thought text, with its own \boxed stripped out)
</think>                        ← added manually by build_records
\boxed{official answer}         ← added manually by build_records (the authoritative answer, not the possibly-wrong upstream one)
```

The essence is decoupling the **thinking process** (using the upstream one) from the **final answer** (using the official one), and making this structure **character-for-character identical to the evaluation protocol** (the evaluator extracts the answer from the `\boxed{}` after `</think>`). The suffix, the paired `<think>/</think>`, and the `\boxed{}` together form the "training distribution = evaluation distribution" contract; when revising the approach later, **the first rule is not to break it.**

#### Phase 3: The Stratified Sampler — "all seven categories in every batch"

`batch=1` + `grad_accum=8` gives an effective batch of 8. With random shuffling, a given effective batch could be **all of the same puzzle type**, swinging the gradient back and forth — bad for the scarce hard puzzles. The fix is to use **modulo round-robin** to spread each type's samples evenly across all the batch buckets (intuitively: dealing cards, handing one card of each suit in turn to N batches).

The engineering is deliberately restrained: rather than hacking TRL's internal shuffling logic, it **precomputes a fixed sample order** and then feeds it through a "take it exactly as given" `PrecomputedOrderSampler` to the DataLoader, with `StratifiedSFTTrainer` overriding just the single method `get_train_dataloader`. Clean, reproducible, minimal surface area.

#### Phase 4: Phase 1 Training (Train) — brute force wins the day

```python
SFTConfig(
    learning_rate=2e-4,      # ① large learning rate
    warmup_steps=0,          # ② no warmup
    num_train_epochs=1,      # only one pass
    adam_beta2=0.95,         # ③ smaller than the default 0.999, more sensitive to recent gradients
    max_grad_norm=1e9,       # ④ effectively turns off gradient clipping
    neftune_noise_alpha=5.0, # ⑤ NEFTune noise
    packing=False,           # ⑥ no packing (avoids cross-sample attention leakage)
)
```

Phase 1's role is to "**slam the trace into the LoRA hard and fast**": a large lr + no warmup + **no gradient clipping**, deliberately pulling the distribution over aggressively. The cost is that training may be rough and unstable — which is exactly what Phase 2 cleans up. As soon as training finishes, **archive it separately** as `phase1_adapter`, both as a safety net and as the "trained for one phase only" A/B comparison candidate.

#### Phase 5: Phase 2 Nudge — a gentle small-step push (the most elegant part of the whole approach)

Phase 2 **continues training on the same model** that Phase 1 left off with (the LoRA weights carry over, not reloaded), using the Nudge set as data. A side-by-side comparison of the two phases:

| Parameter | Phase 1 (Train) | Phase 2 (Nudge) | Meaning |
|------|----------------|------------------|------|
| Learning rate | **2e-4** | **5e-6** | **40× apart**; Phase 2 takes tiny steps |
| Scheduler | linear | **cosine** | Phase 2 finishes smoothly |
| warmup | 0 | **10** | Phase 2 warms up before moving, for more stability |
| `max_grad_norm` | **1e9 (no clip)** | **1.0 (standard clip)** | Phase 2 reins in the gradient to prevent damage |
| Data | all hard + most easy | all hard + a small, balanced amount of easy | Phase 2 focuses on hard puzzles |
| NEFTune | 5.0 | 5.0 | Consistent |

**Why two phases**: Phase 1 uses broad, aggressive strokes to get every puzzle type "roughly learned," but the result is rough and falls a little short on the hard puzzles; Phase 2 uses **1/40 the learning rate** on a small, balanced set dominated by hard puzzles to "carefully extract a few more points," while **mixing in a small amount of fresh easy puzzles as an anchor against catastrophic forgetting**. One handles coverage and speed, the other handles precision and stability. This squeezes out marginal points on the ranking-deciding hard puzzles better than "single-phase, all-in" would, without blowing up training or losing one thing while gaining another.

#### Phase 6: Package the Submission + Cleanup

After saving the final adapter, **patching `adapter_config.json`** is a pitfall that fails outright if not fixed:

```python
cfg["base_model_name_or_path"] = BASE_MODEL_NAME   # ① during training it got written as a local cache path; must be reverted to the official name
cfg["inference_mode"] = True                        # ② mark inference mode
cfg["lora_dropout"]   = 0.0                          # ③ turn dropout fully off at inference to avoid introducing randomness
```

Then pack only the two files the rules require, `adapter_config.json` + `adapter_model.safetensors`, into `submission.zip` (no base weights, no tokenizer; the evaluator brings its own).

### 4.4 Key Technical Points

Pulling out the core technical decisions of the whole approach:

1. **Trace-distillation SFT**: pour the "solving process + answer" into the LoRA as the SFT target, so the model reproduces the solve itself on the evaluation side (where code cannot run). This is the shared paradigm of all mainstream solutions in this competition.
2. **The "training distribution = evaluation distribution" format contract**: `PROMPT_SUFFIX`, the paired `<think>/</think>`, and re-attaching `\boxed{}` with the official answer, aligning the training target character-for-character with the evaluation protocol. **This is the place you can least afford to get wrong.**
3. **Two-phase Train → Nudge**: lr 40× apart, clipping from off to on, data narrowing from full to a hard-puzzle focus. Aggressive first, precise second — better than going all-in at once.
4. **Architecture-targeted LoRA**: covering Mamba's `in_proj/out_proj`, not just Attention's `q/k/v/o_proj` — a key detail for fine-tuning hybrid-architecture models.
5. **Stratified sampling + a type-balanced Nudge set**: ensuring the scarce hard-puzzle types always have a stable presence in the training signal.
6. **The "contradictory" NEFTune + dropout=0 combination**:
   - **NEFTune (Noisy Embedding Fine-Tuning)**: adds uniform noise to the embeddings during training (`X_noisy = X + (α/√(L·d))·ε`, with α=5.0 in this solution, on the conservative side), **applied only during training, never at inference**. In the paper it raised LLaMA-2-7B's AlpacaEval win rate from ~29% to ~64% at essentially zero cost.
   - **Why it's especially well suited here**: the problem statements are stuffed with adversarial fluff ("Alice's Wonderland," made-up stories, substituted operator symbols) — these are noise, not signal. NEFTune forces the model to **grasp the real rule through the fluff** rather than memorizing the literal problem-statement strings.
   - **Seemingly contradictory, but each handles its own job**: `dropout=0` encourages "memorizing" (you want the abstract rule remembered), while NEFTune suppresses "memorizing" (you want the concrete fluff forgotten) — **what should be memorized is the abstract rule; what should be forgotten is the concrete fluff.**
7. **No quantization, pure BF16 training**: a 30B MoE is memory-hungry, but quantization would hurt precision; a task that demands "exact replication of the trace" chooses not to quantize.

### 4.5 The Solving Logic of the Two Hard Categories: cryptarithm and bit manipulation

> cryptarithm and bit manipulation are the two categories that genuinely separate the field; 0.86 → 0.89 comes almost entirely from here. Below, the solving logic of each is laid out clearly.

- **bit manipulation**, three core insights:
  - (1) **Serial, bit by bit**: force the model to break "an 8-bit operation" into 8 "single-bit sub-problems" and write them out, or accuracy plunges to ~9%;
  - (2) **Column view + verification**: treat each output bit's values across all examples as a "column," find the boolean function that reproduces that column, then verify with the query (to prevent a coincidental match);
  - (3) **HEX compression to save tokens**: compress `01101001` into `69`, cutting about 28% of the tokens and freeing up budget to search for more complex gates (MAJ, etc.).

- **cryptarithm**:
  - Ceiling-level difficulty: the naive search space is roughly `10! × 24³ ≈ 5×10¹⁰`, utterly impossible to write out step by step within 7680 tokens.
  - The breakthrough is called the **signature**: abstract "which symbols repeat, and in which positions of the output they appear" in the equation into a pattern (e.g. `ABCCCDD`), **precompute the candidate digit combinations for each signature into a "signature catalog," and have the model memorize it**; at inference time the model does not search from scratch but instead "recalls" the candidates and then uses DFS to verify consistency.

- **This is precisely the deciding factor of the whole competition**: **"what to make the model memorize vs. what to make it compute live in the trace,"** turning the most expensive first step from "brute-force search" into "table-lookup recall." Even in the winning solution, cryptarithm is only ~30% (deduce) / ~10% (guess) solvable, **and this gap is the main source of those 3 points between 0.86 and 0.89.**

---

*The reusable core of this solution is packaged as the [`tracedistill`](../README.md) library; the original competition script is preserved verbatim under [`competition/`](../competition/).*
