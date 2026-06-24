# The Competition Base Model Explained: NVIDIA-Nemotron-3-Nano-30B-A3B

> This document explains **what** the base model locked in for this competition is, **what architecture** it uses, and **which family** it belongs to,
> as well as **what these design choices mean** for this particular competition.
> The specifications section follows the official Hugging Face model card (released 2025-12-15), with sources cited.
> Companion reading: `docs/dataset.md`.

---

## 0. One-Minute Overview

- This is an **open-source reasoning model** released by NVIDIA in late 2025. The competition uses it as a **single fixed base shared by everyone** — you may only train a LoRA on top of it, so the contest is a fair comparison of "technique" rather than "whose base model is stronger."
- It is a **hybrid-architecture MoE**: **30B total parameters, but only about 3.5B activated per token**. The backbone is made of **Mamba-2 state-space layers**, interleaved with only **a small number of attention layers** (23 : 6). This is not the pure Transformer you're used to.
- It is a **native reasoning model**: by default it first emits a `<think>` chain-of-thought before giving the conclusion. This is exactly the premise behind this competition's "use SFT to teach it to write solution traces" strategy.
- It natively supports **context lengths up to 1 million tokens**, but **the competition hard-caps inference at 8192** (generation ≤ 7680). This gap is the root of many of the competition's techniques.

---

## 1. Decoding the Name Piece by Piece

`NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

| Segment | Meaning |
|---|---|
| **Nemotron** | NVIDIA's open-source model family (see Section 5). "Open models + open data + open training recipes" is its signature. |
| **3** | The 3rd generation (Nemotron-3). |
| **Nano** | The **smallest tier** in the family (Nano / Super / Ultra, three tiers — see Section 5). Nano is designed to "run on a single GPU." |
| **30B** | **30B total parameters.** |
| **A3B** | **A**ctive **3B** — this is the hallmark of **MoE**: although there are 30B total parameters, each token actually activates only about **3.5B**. |
| **BF16** | The weights are provided in **bfloat16** (the competition loads them in BF16, without quantization). |

In one sentence: **"a parameter count like a mid-size model (30B), but a compute cost like a small one (3.5B)"** — and that is precisely the selling point of MoE.

---

## 2. Official Specifications Table (from the Hugging Face Model Card)

| Item | Value |
|---|---|
| Total / active parameters | **30B total / ~3.5B active** (per token) |
| Architecture | **Hybrid MoE**: 23 Mamba-2 + MoE layers, 6 attention layers |
| MoE configuration | **128 experts + 1 shared expert** per MoE layer, **6 experts** activated per token |
| Native context | **Up to 1,000,000 (1M) tokens**; HF default config is 256k |
| Model type | **Unified reasoning / non-reasoning model** (whether it "thinks" can be toggled via a switch in the chat template) |
| Training data | ~**25 trillion (25T) tokens** across 141 datasets |
| Data cutoff | Pre-training 2025-06-25, post-training 2025-11-28 |
| Languages | English, German, Spanish, French, Italian, Japanese + 43 programming languages |
| License | NVIDIA Nemotron Open Model License (commercial use permitted) |
| Release date | **2025-12-15** |
| Selected benchmarks | MMLU-Pro 78.3; AIME25 (with tools) 99.2%; MiniF2F pass@32 79.9%; SWE-Bench 38.8% |

> ⚠️ **Note the release date:** the model was only released on 2025-12-15, while the competition began on 2026-03-16.
> In other words, competitors were dealing with a **brand-new model with virtually no ready-made fine-tuning toolchain** —
> which explains why the forums were full of "environment pitfall" posts, and why the organizers had to pre-package a stack of wheels (see Section 6).

---

## 3. Architecture Deep Dive: Why It Isn't an Ordinary Transformer

This model has two non-standard design choices you must understand; they directly determine the difficulty of training it.

### 3.1 Hybrid Mamba-Transformer (23 Mamba-2 layers : 6 Attention layers)

- **Standard large models**: every layer is self-attention. Attention is powerful, but its compute/memory grows **quadratically** with sequence length, making long contexts expensive.
- **Mamba (state-space model / SSM)**: it models context using a **"state" that advances along the sequence**, giving **linear** complexity in sequence length — fast and memory-efficient on long text — but a single layer's "global information retrieval" ability is weaker than attention's.
- **NVIDIA's compromise**: **use Mamba-2 for the vast majority of layers, keeping attention only at a few key positions (6 layers)**. This captures Mamba's speed and long context while using a small amount of attention to recover the "precise retrieval" capability. This is what gives it the confidence to claim a **1M context**.

**Direct impact on you (expanded in Section 6):**
- During training, LoRA is attached not only to attention's `q/k/v/o_proj`, but also to the Mamba blocks' `in_proj / out_proj` (this is exactly what you see in `training.py`'s `target_modules`).
- Mamba relies on specialized CUDA kernels (`mamba_ssm`, `causal_conv1d`), and their compatibility on new hardware (Blackwell) is poor → **training is extremely slow**, which is one of this competition's major pain points.

### 3.2 MoE (Mixture-of-Experts): 128 + 1 experts, 6 selected per token

- Each MoE layer contains **128 "expert" sub-networks + 1 shared expert**. A **router** picks the **6 most relevant experts** for each token to compute, while the other 122 sit idle.
- That's why **total parameters are 30B, but a single token activates only about 3.5B** — this is the source of "A3B" and of the "large capacity + small compute" trade-off.
- **Costs / pitfalls**:
  - At inference time, even with `temperature=0` (greedy), the **non-determinism of expert routing + floating-point accumulation** still causes the same input to produce slightly different outputs → this is the root cause of the forum observation that "the same adapter's score jitters when you submit it multiple times."
  - When fine-tuning, you have to decide whether LoRA should cover the MoE-related layers and whether to untie the `MoE tie weights` (the 3rd-place report found untying them worked better).

### 3.3 Native Reasoning Model (built-in `<think>`)

- It is designed as a **unified reasoning model**: when faced with a problem, it **first generates a chain-of-thought (reasoning trace), then gives the conclusion**; whether it "thinks" can be controlled by a switch in the chat template.
- **This is the foundation of the entire competition's strategy**: the model already works via chain-of-thought, so what you do is use SFT to **replace the "content of the chain-of-thought"** with your own designed, deterministic solution process, so that at evaluation time it reproduces this process on its own and computes the answer.
- The engineering counterpart (which you'll see in `training.py`): `tokenizer.apply_chat_template(..., enable_thinking=True)`; the assistant segment of each training sample ends with `</think>\n\boxed{answer}`.

---

## 4. A Key Observation: Its Tokenizer Is "Very Token-Hungry"

This isn't in the official spec table; it's a hands-on observation that the 1st-place team pointed out in their writeup, but it is extremely important:

> **Nemotron's tokenizer is close to "one digit / binary character = one token."**

- Consequence: on problems like **bit manipulation** (screens full of 8-bit binary strings) and **cryptarithm** (screens full of symbols), simply writing out the problem statement and intermediate results is extremely **token-consuming**.
- And the competition's generation limit is only **7680 tokens**. So top solutions widely use one trick: **compress long binary strings into HEX** (`01101001` → `69`), cutting tokens by roughly 28% in one move, and spending the saved budget on searching more complex rules.
- Remember this rule: in this competition, **"token-hungry tokenizer" × "7680 hard cap" = you must compress the trace**, and this is the core engineering constraint for cryptarithm/bit problems.

---

## 5. Introducing the Nemotron Model Family

Understanding this family helps you see why the competition is designed the way it is.

### 5.1 What It Is

**Nemotron is NVIDIA's open-source large-model product line**, whose core philosophy is **"fully open"**: it opens not only the **model weights**, but also the **pre-training / post-training datasets** and the **training recipes**. It comes with a full open-source toolchain (collectively called NeMo):

| Tool | Purpose |
|---|---|
| NeMo Data Designer | Generate domain/task-specific **synthetic data** |
| NeMo Curator | **Filter and curate** data (text / image / video / audio) |
| NeMo RL | Large-scale **reinforcement learning** training (GRPO, etc.) |
| NeMo Gym | Build / manage **RL environments** |

> This is also why the competition chose Nemotron as its baseline: **open + a unified base = everyone competes on technique under consistent conditions, and results are reproducible.**
> The directions the competition allows (prompt engineering, data filtering, synthetic data, RL, lightweight fine-tuning) almost exactly mirror this toolchain.

### 5.2 Three Tiers: Nano / Super / Ultra

Nemotron is usually split into three tiers by scale, corresponding to different deployment scenarios:

| Tier | Positioning | Typical deployment |
|---|---|---|
| **Nano** | Smallest, **edge / single-GPU** | Runs on a single GPU (this is the tier used in this competition) |
| **Super** | Medium, **single high-end GPU** | A single data-center-class GPU |
| **Ultra** | Largest, **multi-GPU / data center** | Multi-GPU clusters |

This competition uses **Nano** precisely because it needs to be able to **run inference on a single GPU (including the Blackwell that Kaggle provides)** across hundreds of test problems.

### 5.3 Lineage (to Help Locate Where the "3rd Generation" Sits)

NVIDIA's Nemotron has gone through several generations of evolution; the key thread:

- **Early Nemotron-4** (2024): standard-Transformer open-source large models (e.g., 340B, 15B).
- **Llama-Nemotron / Nano-Super-Ultra** (first half of 2025): introduced the "**reasoning switch**" (reasoning on/off) and strengthened "thinking" ability.
- **The hybrid-architecture line** (2025): began replacing pure attention with a **Mamba-Transformer hybrid**, focusing on long context + high throughput.
- **Nemotron-3 Nano (this model, 2025-12)**: **merges** all the lines above — hybrid Mamba-Transformer + MoE + unified reasoning + 1M context. Think of it as the current culmination of this technical line in a small model.

> (The generational breakdown of the lineage is a simplification to aid understanding; the official model card's specs are the real authority.)

---

## 6. The Concrete Impact of These Designs on This Competition

Pulling all the points above together into "how it shaped this competition":

1. **Native 1M context, yet cut down to 8192 / 7680 generation.**
   The model itself has more than enough long-context capacity; the bottleneck is entirely an **artificial competition rule**. As a result, "can the solution process be compressed into 7680 tokens" became as important as "is the answer correct" → giving rise to HEX compression, signature catalogs, the "memorize vs. compute" trade-off, and every other technique.

2. **The Mamba component → extremely slow training, hard-to-set-up environment (a compute barrier).**
   Mamba's kernels (`mamba_ssm` / `causal_conv1d` / Triton) have poor compatibility on the **Blackwell GPU**. Kaggle's Blackwell can only train about 20 million tokens in 3 hours; meanwhile the 1st-place team trained **over 1 billion tokens across hundreds of hours** on their own Blackwell workstation. This gives the competition a strong **compute-barrier** flavor (forum post 690161, "Why GRPO is Painfully Slow," is about exactly this).

3. **MoE → extra fine-tuning decisions + noisy scoring.**
   - Whether LoRA's target_modules should cover the MoE/Mamba projection layers, whether to untie the `MoE tie weights`, whether to train `lm_head` (note: some competitors pointed out that `lm_head` actually has no effect during vLLM inference).
   - Routing non-determinism → the same adapter's score jitters across repeated evaluations → **you must rely on local stratified CV and not just watch the Public LB.**

4. **Native reasoning → "SFT-distilling solution traces" is the path of least resistance.**
   Because the model already thinks before it answers, you only need to swap the "thinking" content for a deterministic solution process. So the mainstream solution across the board was:
   **write a deterministic solver locally → generate CoT traces → SFT into a LoRA → let the model reproduce them on its own at inference time.**

5. **Token-hungry tokenizer → compression is a hard requirement.** (see Section 4)

6. **A new model only released in 2025-12 → an immature toolchain.**
   Competitors had to handle a large number of environment dependencies themselves, and the organizers had to pre-package wheels. Being "brand-new" was both its selling point (technically advanced) and a source of friction for participants.

---

## 7. One-Sentence Summary

> An open-source small reasoning model that is **a 30B-total / 3.5B-active MoE, a hybrid architecture built primarily on Mamba-2 with only a small amount of attention mixed in, natively equipped with chain-of-thought, and supporting a 1M context.**
> Its capacity is large enough to "memorize" a rule library, while its active size is small enough to run hundreds of problems on a single GPU;
> but Mamba makes it **hard to train on new hardware (a compute barrier)**, MoE makes scoring **noisy**, and the tokenizer is **token-hungry** —
> these three factors, stacked on top of the 7680 generation limit, together define the real engineering battlefield of this competition.
