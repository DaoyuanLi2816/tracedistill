# NVIDIA Nemotron Reasoning Challenge — A Deep Dive into the Dataset

---

## 0. One-Minute Overview

- This is a dataset of **"programmatically generated reasoning puzzles"**: each problem gives you a few `input → output` examples, asks you to **reverse-engineer the hidden transformation rule**, and then apply that rule to a final unsolved query.
- The "story wrapping" in the problem statements (things like Alice's Wonderland) and the **visual meaning of the operator symbols** are almost entirely **adversarial noise** — the model must be trained to ignore them.
- The training set has **9,500 problems**, split into **7 major families** (subdivided into 9 subtypes). About **84% are free points**; what actually decides your ranking are the last two families: **bit manipulation** and **cryptarithm**.
- You don't submit predictions — you submit a **LoRA adapter**. At evaluation time the model has to **work out each problem on its own** within a **7,680-token** chain of thought (the evaluation environment can't run code).

---

## 1. The Dataset at a Glance

### 1.1 Fields (train.csv)

The official training set `train.csv` has only **3 columns**:

| Column | Meaning |
|---|---|
| `id` | Unique identifier for the problem |
| `prompt` | The problem statement: several "input → output" examples + one unsolved query |
| `answer` | The ground-truth answer |

⚠️ **Note: the official `train.csv` has no "problem category" column.** Which family a problem belongs to, and what its hidden rule is, **must all be reverse-engineered by the competitor**. "Sorting these 9,500 problems into families and figuring out the playbook for each" is itself the first hurdle of the competition.

> In `solution/training.py` you'll see a `type` column — that comes from a community-curated, third-party reverse-engineered dataset
> (`dgxchen/nemotron-cot-tong`), and is **not an original official field**.

### 1.2 Training Set Size and Category Distribution

The training set contains **9,500 problems** in total. Based on the 1st-place competitor's analysis of `train.csv`, the rough distribution is:

| Category | Count | Approx. solve rate (top solutions) | Nature |
|---|---:|---:|---|
| bit manipulation | 1602 | ~93–97% | ⭐ one of the battlegrounds |
| text cipher | 1576 | ~99–100% | free points |
| numeral system (Roman numerals) | 1576 | 100% | free points |
| unit conversion | 1594 | 100% | free points |
| gravity | 1597 | 100% | free points |
| equation numeric | 732 (deduce 596 + guess 136) | deduce ~95% / guess ~50% | medium |
| cryptarithm | 823 (deduce 659 + guess 164) | deduce ~30% / guess ~10% | ⭐⭐ the decider |
| **Total** | **9500** | | |

**How to read this table:**
- The first five families (the simple families other than bit, plus the easy portion of bit) add up to roughly 7,945 problems (≈84%). Essentially everyone can get close to a perfect score here, and this portion is worth about **0.85–0.86** of the score.
- This is where the **"0.86 wall"** on the leaderboard comes from: the easy families cap out at this many points, so everyone piles up together with no separation.
- **The 3 points from 0.86 → 0.89 come almost entirely from cryptarithm and bit manipulation.** Top teams spend all of their effort on these two families.

### 1.3 The Test Set: Size and Distribution (Important Clarifications)

The official information here is thin, so I'll separate what's **certain** from what can only be **inferred**:

**Established facts:**
- The competition also ships a `test.csv`, but it's **just a handful of sample problems for debugging your submission format**. During real scoring it is swapped out for the **full hidden test set**.
- Full test set size: the official wording is **"hundreds of problems."** The organizers explained on the forum that they **deliberately kept enough test samples** to reduce scoring variance; the cost is that **evaluating one adapter takes about 2 hours**.
  - (Some in the community guessed the magnitude is around ~600 problems, but that's a guess — the organizers never gave an exact number.)
- The test set is split into a **Public LB (visible during the competition)** and a **Private LB (final ranking)**. The organizers repeatedly stressed: **the Private LB is what actually decides the ranking, and fixating on the Public LB invites overfitting.**
- Scoring **has randomness**: even submitting the **exact same** adapter, repeated runs will fluctuate slightly (temperature=0, but MoE routing + floating point make vLLM inference not fully deterministic).

**⭐ How many times is each problem run? — Just once. One shot, one chance.**
- At evaluation time vLLM uses `temperature=0.0` (greedy) and **generates exactly one output per problem**, then extracts `\boxed{}` from that single output to judge correctness.
  **No retries, no multiple sampling.**
- This means you **cannot do self-consistency (sample-and-vote) on the evaluation side**: the temperature is 0, the decoding pipeline is controlled by the organizers, and all you can submit is an adapter.
- The `max_num_seqs=64` in the config is just vLLM's **batch concurrency** (processing 64 different problems in parallel at once) — it is **not "doing each problem 64 times."**
- "The score wobbles when you resubmit the same adapter" refers to wobble **between two independent evaluation runs**, **not** retrying a single problem within one run. For any individual problem, it's always one shot, life or death.
- **Strategic implication:** since the evaluation side can't vote, "verification / self-correction" must be **written into the chain of thought** — the model has to **check its own work within 7,680 tokens, and backtrack when it finds a contradiction** (this is exactly why `VER`, `CHK`, and DFS backtracking show up again and again across the solutions). The model gets only one bullet, so it has to aim itself.

**On whether "the test set's category distribution matches the training set":**
- The organizers **never published** the test set's category distribution.
- But there is **strong evidence** that it is **very close** to the training distribution:
  1. The test and training problems come from the **same programmatic "factory"** (the same templates and rule library) — there's no reason to change the recipe.
  2. The 3rd-place report: the **local validation set (local CV)** he stratified-sampled from the training set was **highly correlated with the Private LB** (the scatter plot was almost a straight line). If the test distribution differed greatly from the training set, this correlation wouldn't hold.
  3. The organizers also directly advised competitors to "**evaluate your adapter via local CV**," which amounts to tacitly admitting "training distribution ≈ test distribution."
- **Conclusion (inferred, not official):** you can **safely assume the test set's category distribution is essentially the same as the training set's** — i.e., easy families dominate, and cryptarithm/bit set the ceiling. This is the shared premise of all top solutions: use a stratified CV from the training set and read it directly as a proxy for the Private LB.

---

## 2. The Seven Puzzle Families, One by One

> For each family: **the real English problem statement + translation + ground-truth answer + hidden rule + the hard part.**
> All examples are taken from real problems shared publicly on the competition forum.

The **common structure** of the dataset (the same for every family — memorize this first):
> **The examples are used to reverse-engineer the hidden parameters/rule; the final query tests whether you can apply it.**
> All examples within a single problem share the same hidden rule.

---

### 2.1 gravity — A free-points family, but it teaches you the "factory" playbook

**English problem statement:**
```
In Alice's Wonderland, the gravitational constant has been secretly changed.
Here are some example observations:
For t = 1.37s, distance = 14.92 m
For t = 4.27s, distance = 144.96 m
For t = 3.28s, distance = 85.54 m
For t = 3.67s, distance = 107.09 m
For t = 1.78s, distance = 25.19 m
Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2.
```

**Ground-truth answer:** `154.62`

**Hidden rule:** the formula is handed to you directly as `d = 0.5·g·t²`, but `g` is random per problem. Solve for it from any example,
`g = 2d/t²` (≈ 15.90 here), then substitute back: `0.5 × 15.90 × 4.41² ≈ 154.62`.

**Three things it teaches you (applicable to every family):**
1. "Alice's Wonderland" and "secretly changed" are **pure filler** — ignore them.
2. All examples in one problem share the same hidden parameter (here, g).
3. Numeric problems have a **1e-2 tolerance**: computing to two decimal places is enough, but the answer must be in **`X.XX` format**.

---

### 2.2 unit conversion — Isomorphic to gravity, but a linear relationship

**English problem statement:**
```
In Alice's Wonderland, a secret unit conversion is applied to measurements.
10.08 m becomes 6.69
17.83 m becomes 11.83
35.85 m becomes 23.79
17.06 m becomes 11.32
31.54 m becomes 20.93
Now, convert the following measurement: 25.09 m
```

**Ground-truth answer:** `16.65`

**Hidden rule:** `output = input × coefficient`, with the coefficient random per problem. `6.69/10.08 ≈ 0.6636`;
after confirming a few examples agree, `25.09 × 0.6636 ≈ 16.65`. The structure is identical to gravity — only the relationship changes from quadratic to linear.

---

### 2.3 numeral system — Integer → Roman numerals

**English problem statement:**
```
In Alice's Wonderland, numbers are secretly converted into a different numeral system.
11 -> XI
15 -> XV
94 -> XCIV
19 -> XIX
Now, write the number 38 in the Wonderland numeral system.
```

**Ground-truth answer:** `XXXVIII`

**Hidden rule:** it's just standard Roman numerals. Decompose digit by digit: `38 = 30 + 8` → `XXX` + `VIII` → `XXXVIII`.

**The hard part:** this is a **string problem — it must be character-for-character exact** (Roman numerals carry the risk of an error the 1e-2 tolerance can't rescue).
So the solutions have the model "decompose digit by digit → concatenate → read back to verify," preventing ordering errors like writing `XL` as `LX`.

---

### 2.4 text cipher — A word-level letter-substitution cipher

**English problem statement:**
```
In Alice's Wonderland, secret encryption rules are used on text.
hmxad apdhvdq vid ohexahm apwqvhm -> alice creates the magical crystal
zxuhpl zhvaidq xyqxld txmmhed -> wizard watches inside village
nfddy xohexydq xy ehpldy -> queen imagines in garden
osfqd qddq lssp -> mouse sees door
vid amdtdp zxuhpl dgjmspdq -> the clever wizard explores
Now, decrypt the following text: bxye aihqdq ahqvmd
```

**Ground-truth answer:** `king chases castle`

**Hidden rule:** a **bijective substitution** over the 26 letters (each letter maps to a fixed other letter).
From `hmxad → alice` you can extract `h→a, m→l, x→i, a→c, d→e`… Collect the mappings from all examples and decrypt the query character by character.

**The clever bit — the "closed vocabulary":**
- In the query, `bxye` decrypts to `?ing` (the letter `b` never appears in the examples, so its mapping is unknown).
- The trick: **the entire dataset's plaintext uses only 77 fixed words** (reverse-engineered by the community). So instead of searching the whole English dictionary for `?ing`,
  you search those 77 words for one matching `?ing` — and the only one is `king`. From this you infer `b→k`.
- This idea of **"locking an open search into a closed search"** gets amplified into a core weapon in the harder families.

---

### 2.5 bit manipulation — The first real battleground

**English problem statement:**
```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.
The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT,
and possibly majority or choice functions.
00010101 -> 10000011
01100011 -> 10111001
11000101 -> 01111000
00011010 -> 10100011
01010110 -> 00000011
... (about 10 examples)
Now, determine the output for: 01101001
```

**Ground-truth answer:** `10001101`

**Hidden rule (the actual transformation for this problem):**
```
s = not(rol1(x))    # rotate left by 1 bit, then NOT
a = not(shr3(x))    # shift right by 3 bits, then NOT
b = shl2(x)         # shift left by 2 bits
out = sel_nand_xnor(s, a, b)   # bits where s is 1 use nand(a,b); bits where s is 0 use xnor(a,b)
```

**Why it's hard + three core insights:**
- Both input and output are 8 bits. **Every output bit is some Boolean function of several input bits**: it could be a constant / identity / NOT /
  a two-input gate (AND/OR/XOR/NAND/NOR/XNOR…) / a three-input gate (MAJ majority, CHOICE, PAR3 parity…) / or even a four-input gate.
- **Insight 1 (bit-serial):** this model **can't compute multi-bit bitwise operations in its head in parallel**. Ask it directly to AND two 8-bit strings
  and accuracy plummets to ~9%. You must force it to **write things out bit by bit**: `0&1=0  1&1=1  0&0=0 …`, decomposing "8-bit operation" into 8 "single-bit subproblems."
- **Insight 2 (column view + verification):** treat the i-th output bit across all examples as a "vertical column," and find which Boolean function reproduces that column.
  But **a wrong function may coincidentally match on the examples**, so the chosen rule must be genuinely verified against the query.
- **Insight 3 (saving tokens):** Nemotron's tokenizer makes **nearly every binary character its own token**.
  8 bits × 10 examples × writing them repeatedly = token explosion. The pros compress the binary strings into **HEX** (`01101001` → `69`) to save tokens,
  and spend the saved budget searching for more complex gates (like MAJ).

---

### 2.6 equation numeric — The symbol itself is noise

**English problem statement:**
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations.
55`39 = 16
61\65 = 126
42>23 = 4223
17\21 = 38
Now, determine the result for: 81`20
```

**Ground-truth answer:** `61`

**Hidden rule:** for `AB operator CD` (two two-digit numbers), the middle symbol (`` ` ``, `\`, `>`) **has zero visual meaning, differs per problem,
and must be reverse-engineered from the examples**:
- `` 55`39 = 16 `` → `` ` `` = subtraction (55 − 39)
- `61\65 = 126`, `17\21 = 38` → `\` = addition
- `42>23 = 4223` → `>` = concatenate ABCD
- So the query `` 81`20 `` → 81 − 20 = **61**

**⭐ Key concept: deduce vs guess (runs through the last two families — be sure to understand it)**

The distinction comes down to one sentence: **whether the operator used in the query was actually demonstrated in the examples above.**

- **deduce (derivable):** the query's operator **appeared in the examples** → just reuse it directly.
  ```
  55`39 = 16     ← this demonstrates what ` does (55−39)
  61\65 = 126
  Find: 81`20     ← the query uses ` , which was demonstrated above → apply directly → deduce
  ```

- **guess (must be guessed):** the query's operator **never appeared even once in the examples** → there's no direct demonstration to look at.

  **Real example (training set id = `260f20c1`, answer `43`):**
  ```
  84[69 = 153
  13+97 = 1260
  46+47 = 2161
  52[80 = 132
  Find: 22\65       ← the query uses \ , but the four rows above only have [ and + ; \ never appears → guess
  ```

  At first glance `\` is never demonstrated, which looks like "insufficient information, unsolvable." But it **really can be solved**, as follows:

  1. **First crack the two operators that do appear in the examples (the deduce part):**
     - `[`: `84[69→153` means `84+69=153`; `52[80→132` means `52+80=132`. → **`[` = addition (the add group)**.
     - `+`: `13+97→1260`, while `13×97=1261`, off by 1 → it's `mul-1`; `46+47→2161=46×47−1` confirms it.
       → **`+` = multiply minus one (the mul group)**.
       (Note: the symbol `+` is actually doing **multiplication** — this proves the point that "operator symbols are noise; the real rule must be inferred from the outputs.")
  2. **Then pin down the query operator `\` (the guess part):** it never appears, but this factory has an iron law — **Observation 2 (within one problem, every operator comes from a different group; groups don't repeat)**.
     This problem has already used up the **add group (`[`)** and the **mul group (`+`)**, so the only unclaimed one left is the **sub group**.
     So even though we've never seen `\`, we can conclude: **`\` must belong to the subtraction family.**

     > **⚠️ The key premise that makes this step work: the candidate families are closed, and there is no "division."**
     > You might ask: "After ruling out add and mul, aren't subtraction **and division** both left? Why must it be subtraction?" — Because **this competition's rule list simply has no division family.**
     > The full set of rules competitors reverse-engineered from `train.csv`, once categorized, comprises **only 4 families**:
     > **join / add / sub / mul.** This list is an empirical fact derived by "adding rules until they explain every problem in the training set, and not one rule more" — not an assumption.
     > So after excluding add and mul, the **only non-join family left is sub** — it isn't a coin flip between "subtraction vs division"; division simply isn't in the candidate set at all.
     > (The one thing with a faint "division flavor," `max(a,b) mod min(a,b)`, is categorized under the sub group; here it gives `65 mod 22 = 21 ≠ 43`, so it's ruled out.)
  3. **Substitute and compute the answer:** within the subtraction family, `22\65 → |22−65| = 43` (the signed `22−65=−43` is not taken) → **answer 43**. ✓

  **This is the essence of why guess is solvable:** the operator symbol itself provides no information, but **the structure of the whole problem** (each family is used only once) in turn locks down "which family that never-seen operator belongs to."
  - The residual small uncertainty (e.g., within the subtraction family, is it `|a−b|` or `b−a`?) is what makes guess genuinely hard, with a solve rate of only ~50% —
    that step can only be bet on via **operator priors + training-set frequency**, not by logical necessity.

- **The difficulty gap:** guess is far harder than deduce. Solve rates: equation numeric ~95% (deduce) → ~50% (guess);
  cryptarithm ~30% (deduce) → ~10% (guess).
- Top solutions compressed this regularity into a **24-rule list** (add / add+1 / flip_add / sub / absdiff / mul / flip_mul / join …),
  and discovered two iron laws:
  - **Observation 1 (pattern consistency):** within one problem, all non-join operators use the same pattern (all normal or all flipped).
  - **Observation 2 (no group repeats):** within one problem, different operators never use the same group (add / sub / mul groups are each used once).
  - These two constraints shrink the search space enormously.

---

### 2.7 cryptarithm — The final decider, ceiling-level difficulty

**English problem statement:**
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations.
|!)<< = <[[
::$\{ = !{?
<<$'' = {\|(
Now, determine the result for: !')?<
```

**Ground-truth answer:** `:![`

**Hidden rule:** it's just equation numeric **with one more layer of symbol encryption** — **even the digits themselves are replaced by symbols.**
Each equation has the form `2 symbols + operator symbol + 2 symbols = output symbols`, and you must crack three things **simultaneously**:
1. Each symbol → which digit (a bijection, with up to ~10 digit-symbols per problem)
2. Each operator symbol → which of the 24 rules
3. Re-encode the computed numeric result **back into symbols**

The final crack for this problem:
```
|=0  !=4  <=7  [=1  :=3  {=2  \=8  ?=9  '=6  (=5
) = flip_add
$ = flip_mul
```
The query `!')?<` → mapped via the substitution becomes `46)97` → `)` is flip_add → flip each two-digit number `64 + 79 = 143`
→ flip the result to get `341` → encode 3, 4, 1 back into symbols = `:![`.

**Why it's ceiling-level hard:**
- **Naive search space:** digit assignments `10! ≈ 3.63 million` × operator rules `24³ ≈ 14 thousand` ≈ **5×10¹⁰** combinations.
- This is **absolutely impossible** to write out token by token within 7,680 tokens. So the difficulty isn't just "solving it,"
  it's **"compressing the search process into a short program the model can run to completion within 7,680 tokens."**
- The breakthrough in top solutions is called the **signature**: abstract "which symbols repeat, and where symbols appear in the output" of an equation into a pattern
  (e.g., `ABCCCDD`), **pre-compute the candidate digit combinations for each signature, and turn it into a "signature catalog" for the model to memorize**;
  at inference time the model doesn't search from scratch but "recalls" the candidates, then uses DFS to check consistency with the other equations.
- This is the very crux of the whole competition: **"what should the model memorize, and what should it compute on the fly in the trace"** —
  turning the most expensive first step (assigning values to 4–8 symbols at once) from "brute-force search" into "table lookup from memory."
- This is also why even the winning solution gets only ~30% (deduce) / ~10% (guess) solve rates on cryptarithm:
  many problems either won't fit in the token budget, or are simply underdetermined. **This 30% vs 10% gap is the main source of those 3 points between 0.86 and 0.89.**
