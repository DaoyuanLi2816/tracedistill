# NVIDIA Nemotron Model Reasoning Challenge 解决方案

---

## 1. 比赛理解

### 1.1 背景与目标

**NVIDIA Nemotron Model Reasoning Challenge（NVIDIA 推理挑战赛，简称 NVIDIA-NMRC）** 是 NVIDIA Research 在 2026 年举办的一场 Kaggle 比赛（报名 2026-03-16 起，提交截止 2026-06-15，最终 **4163 支队伍**参赛）。

比赛的核心命题是：**在一个所有人共用的固定基座大模型之上，通过技术手段提升它在一套全新推理基准上的解题准确率。**

- **共享基座**：所有参赛者都使用同一个开源模型 **NVIDIA-Nemotron-3-Nano-30B-A3B**（见 §1.3），不能换模型。这样比的就是"技术"，而不是"谁的底座更强"。
- **唯一硬性要求**：最终提交物必须是一个**适配该基座的 LoRA adapter**（rank ≤ 32），打包成 `submission.zip`。
- **开放的技术路线**：官方允许任意方向：提示工程、数据过滤与筛选、合成数据生成、强化学习、轻量微调等，框架不限（Hugging Face / Unsloth / TRL / Axolotl 皆可）。

一句话概括输入输出：

> **输入**：一批程序生成的逻辑推理谜题（每题给若干「输入 → 输出」示例，让你反推隐藏规则，再应用到一个待解 query 上）。
> **输出**：一个 LoRA adapter；评测时主办方用它在隐藏测试集上**现场跑推理**，模型要在生成的思维链里把每道题**自己算出来**。

### 1.2 这场比赛为什么"不一样"

这场比赛和普通 Kaggle / 普通 SFT 有几条根本差异，决定了它的全部打法：

最关键的一点：**评测端不能跑 Python。** 你在本地用代码 100% 解出训练题毫无用处，关键是把"解题过程"写成思维链（Chain-of-Thought, CoT），用 SFT 蒸馏进 LoRA，让模型推理时**自己复现这段过程**把题算出来。

### 1.3 基座模型：Nemotron-3-Nano-30B-A3B

理解这个模型，才能理解为什么本场比赛的技巧都长这个样子。

`NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` 拆开来看：

- **30B / A3B**：**总参数 30B，但每个 token 只激活约 3.5B**（A3B = Active 3B）。这是 **MoE（Mixture-of-Experts，混合专家）** 的标志：每个 MoE 层有 128 + 1 个专家，路由器为每个 token 只挑 6 个来算。"容量像中模型，算力像小模型"。
- **混合架构**：它**不是纯 Transformer**。主体是 **23 层 Mamba-2（状态空间模型 SSM，复杂度随序列长度线性增长）**，只夹了 **6 层 Attention**。这让它能宣称原生 **100 万 token** 上下文。
- **原生 reasoning 模型**：它默认**先吐一段 `<think>...</think>` 思维链，再给结论**。这正是本场"用 SFT 教它写解题 trace"打法的地基。
- **BF16**：以 bfloat16 提供，比赛就是用 BF16 加载，**不量化**。

这三个特性直接塑造了比赛的工程战场：

1. **原生 1M 上下文，却被比赛硬砍到 8192（生成 ≤ 7680 token）。** 于是"解题过程能不能压进 token 预算"和"解得对不对"同等重要 → 催生了 HEX 压缩、签名目录、"背 vs 算"取舍等一系列技巧。
2. **Mamba 成分 → 训练巨慢、环境难搞。** Mamba 依赖 `mamba_ssm`、`causal_conv1d` 等专用 CUDA kernel，它们在 Kaggle 的新款 **Blackwell GPU** 上兼容差。Kaggle 3 小时只能训约 2000 万 token，而第 1 名用自有工作站训了**上亿 token、上百小时**，这是很强的算力门槛。
3. **MoE → 评分有噪声。** 即便 `temperature=0`（贪心解码），专家路由 + 浮点累加的非确定性也会让同一 adapter 多次评测分数小幅抖动。**结论：必须靠本地分层 CV，别只盯 Public LB。**
4. **它的 tokenizer "很费 token"**（第 1 名实战观察）：接近"一个数字 / 二进制字符 = 一个 token"。在满屏二进制、符号的难题上，光写题面就极度耗 token，所以压缩 trace 是刚需。

### 1.4 领域知识入门

**核心要点（5 条）**

1. **这是"工厂"数据，不是自然数据。** 每类题有固定的生成模板和**有限的规则库**；逆向出规则库 ≈ 攻克这一类。
2. **表面全是噪声。** "爱丽丝仙境（Alice's Wonderland）"的故事包装、算子符号的视觉含义，都是用来骗模型的对抗性噪声，要训练它无视。
3. **示例定参数，query 考应用。** 同一题的所有「输入→输出」示例共享同一套隐藏规则；最后的 query 考你应用。所有类别都是这个结构。
4. **交 adapter，不交结果。** 解题算法必须蒸馏进模型，评测端现场跑、不能跑代码。
5. **格式即生命。** 字符串题错一字符 = 0 分；数值题格式不对 = 0；漏 `\boxed{}` = 0。

**术语速览（10 条）**

1. **LoRA（Low-Rank Adaptation，低秩适配）**：冻结基座，只训一小撮低秩矩阵，参数量极小、易部署。本场提交物就是它。
2. **SFT（Supervised Fine-Tuning，监督式微调）**：用「输入 → 标准输出」样本对模型做监督训练。
3. **CoT（Chain-of-Thought，思维链）**：模型在给最终答案前显式写出的推理过程。
4. **Trace distillation（轨迹蒸馏）**：把"解题过程（trace）"作为训练目标，用 SFT 灌进模型，让它学会复现这套过程。
5. **MoE（混合专家）**：用路由器为每个 token 只激活一小撮专家子网络，省算力。
6. **Mamba / SSM（状态空间模型）**：用随序列推进的"状态"建模上下文，复杂度对长度线性，长文本又快又省显存。
7. **`\boxed{}`**：LaTeX 命令，模型把最终答案写在里面，评测从中抠答案。
8. **NEFTune（Noisy Embedding Fine-Tuning）**：训练时给 embedding 加一点随机噪声的轻量正则技巧（见 §4.4）。
9. **deduce / guess**：难题的两个隐藏子类，区别在于 query 用的算子在示例里**出现过（deduce，可推）** 还是**没出现过（guess，需猜）**。
10. **Stratified sampling（分层采样）**：组 batch 时让每个 batch 内各类型样本都摊到一点，稳定梯度。

---

## 2. 数据集介绍

### 2.1 字段与规模

官方训练集 `train.csv` 只有 **3 列**，共 **9500 道**题：

| 列名 | 含义 |
|---|---|
| `id` | 题目唯一标识符 |
| `prompt` | 题面：若干「输入→输出」示例 + 一个待解 query |
| `answer` | 标准答案 |

⚠️ **关键事实：官方 `train.csv` 既没有"题目类别"，也没有"思维链"。** 题目属于哪一类、隐藏规则是什么，全部要选手自己逆向。本方案 `training.py` 读取的并非官方 csv，而是一个**社区第三方数据集** `dgxchen/nemotron-cot-tong`，字段为 `type, prompt, answer, generated_cot`，其中多出来的 `generated_cot` 列为每道题预先备好了解题思维链，这正是本方案做轨迹蒸馏的原料。这一点至关重要，§4.1 会展开。

### 2.2 七大类题型与难度分布

训练集分 **7 大类（细分 9 小类）**，难度极度不均：

| 类别 | 数量 | 顶尖方案大致可解率 | 性质 |
|---|---:|---:|---|
| gravity（重力） | 1597 | 100% | 送分 |
| unit conversion（单位换算） | 1594 | 100% | 送分 |
| numeral system（罗马数字） | 1576 | 100% | 送分 |
| text cipher（文字密码） | 1576 | ~99–100% | 送分 |
| bit manipulation（位运算） | 1602 | ~93–97% | ⭐ 战场之一 |
| equation numeric（数字方程） | 732（deduce 596 + guess 136） | deduce ~95% / guess ~50% | 中 |
| cryptarithm（密码算术） | 823（deduce 659 + guess 164） | deduce ~30% / guess ~10% | ⭐⭐ 最终决胜 |
| **合计** | **9500** | | |

各类题的隐藏规则一句话速记（均含"爱丽丝仙境"故事包装，需无视）：

- **gravity**：公式 `d = 0.5·g·t²` 白给，但 `g` 每题随机，从示例反解 `g`。
- **unit conversion**：`输出 = 输入 × 系数`，系数每题随机（与重力同构，只是线性）。
- **numeral system**：整数 ↔ 罗马数字（**字符串题，必须一字不差**）。
- **text cipher**：26 字母双射替换密码。精妙点：**整个数据集明文只用 77 个固定单词**，把开放搜索锁成封闭搜索。
- **bit manipulation**：8-bit 输入 → 8-bit 输出，每个输出 bit 是若干输入 bit 的布尔函数（AND/OR/XOR/NOT/MAJ…）。模型**无法并行算位运算**，必须**逐位写出来**，否则准确率暴跌到 ~9%。
- **equation numeric**：`两位数 算子 两位数`，**算子符号的视觉含义是噪声**，真规则要从示例输出反推（如 `` ` `` 可能是减法、`+` 可能是乘法）。
- **cryptarithm**：equation numeric **再套一层符号加密**：连数字本身都换成了符号，要同时破解"符号↔数字""符号↔算子规则""结果再编码回符号"三件事。**天花板级难度**。

**deduce vs guess（贯穿后两类，务必搞懂）**：区别只有一句话，就是 **query 里用的那个算子，在上面的示例里到底演示过没有。** 演示过 → deduce（直接照搬）；一次没出现 → guess（要靠"每个算子家族只用一次"等结构性约束反推它属于哪个家族）。guess 远难于 deduce，是真正拉开差距的子类。

### 2.3 测试集与"0.86 的墙"

- **测试集**：比赛里给的 `test.csv` 只是几道样例题（调试提交格式用），正式评分时换成**"数百道题（hundreds）"**的完整隐藏测试集，分 Public LB（比赛中可见）与 Private LB（最终排名）。**真正决定排名的是 Private LB。**
- **测试集分布 ≈ 训练集分布**（官方未明示，但证据充分：同一程序化"工厂"生成，且第 3 名报告其训练集分层 local CV 与 Private LB 高度相关）。**所以可以放心用训练集分层做 local CV 来对标 Private LB。**
- **每题只跑一次**：`temperature=0.0` 贪心，每题只生成一次、一锤定音，没有重试、没有多次采样。**推论：** 既然评测端不能投票，"验证 / 纠错"就必须**写进思维链里**（模型自己验算、发现矛盾再回退）。

理解"0.86 的墙"是理解整场比赛的钥匙：

> 前 5 类送分题 + bit 的简单部分加起来约 **84%**，几乎人人都能接近满分，这部分大概就值 **0.85~0.86** 分，大家全挤在这里。
> **从 0.86 拉到 0.89 的那 3 个百分点，几乎全部来自 cryptarithm 和 bit manipulation。** 顶尖队伍的全部精力都花在这两类上。

---

## 3. 评价指标介绍

比赛采用**准确率（Accuracy）**，即正确作答题数占总题数的比例：

$$
\text{Accuracy} = \frac{\#\{\text{预测正确的题}\}}{\#\{\text{全部题}\}}
$$

判分流程（主办方后台，非选手脚本）：

1. **加载**：vLLM 推理引擎加载 Nemotron-3-Nano-30B 基座 + 你提交的 LoRA adapter（必须含 `adapter_config.json`）。
2. **生成**：对每道题，提示模型作答，并要求把最终答案写进 `\boxed{}`。
3. **抽取**：从生成文本里提取答案，**优先解析 `\boxed{}` 内容**，其次启发式规则，最后回退到"最后一个数值"。
4. **判对**：预测与真值**精确字符串相等**，或**相对误差在 $10^{-2}$ 以内**，即视为正确。

评测固定参数（务必让训练对齐这些）：

| 参数 | 值 | 含义 |
| :--- | :--- | :--- |
| `max_lora_rank` | 32 | adapter 的 rank 上限（硬约束） |
| `max_tokens` | 7680 | **单题生成上限**，trace 必须压进这个预算 |
| `top_p` | 1.0 | 贪心解码下不起作用 |
| `temperature` | 0.0 | 贪心解码，每题一次 |
| `max_num_seqs` | 64 | vLLM 批处理并发数（**不是**每题做 64 遍） |
| `gpu_memory_utilization` | 0.85 | vLLM 占用显存的比例 |
| `max_model_len` | 8192 | 上下文窗口（训练 `max_seq_length` 与之对齐） |

**提交形式**：rank ≤ 32 的 LoRA adapter，打包为 `submission.zip`，只含 `adapter_config.json` + `adapter_model.safetensors` 两个文件（**不含基座权重**，基座评测端自带）。

> 对指标的两条直觉：（1）**数值题有 1e-2 容差**，算到两位小数即可，但答案要写成 `X.XX` 格式；（2）**字符串题（罗马数字、密码）零容差**，错一个字符就是 0，所以方案里会让模型"逐位拆 → 拼接 → 回读验证"。

---

## 4. 解决方案解析

> 本节按执行管线的顺序，介绍本方案 `solution/training.py` 实现的**两阶段微调 Train → Nudge**，重点放在"为什么这么设计"。

### 4.1 顶层心智模型：轨迹蒸馏 SFT

在钻进代码前，先抓住整套方案在干什么：

> **拿一份"已经带好解题思维链（CoT）"的数据，用 SFT 把这套思维链 + 最终答案灌进一个 LoRA adapter，让基座模型在推理时能复现出来。**

训练数据用的是公开数据集 `dgxchen/nemotron-cot-tong`，字段为 `type, prompt, answer, generated_cot`，其中 `generated_cot` 已为每道题备好了解题思维链，这就是蒸馏的原料。整套方案要做的，就是把"题面 → 思维链 + 答案"组织成 SFT 样本，微调出一个能在推理时复现解题过程的 LoRA。

### 4.2 方案流程概览

> 直接用浏览器打开 `方案流程图.svg` 查看高清版本。

### 4.3 各阶段详解

#### 阶段 0：配置与环境补丁

先看几行核心配置：

```python
HARD_TYPES = {"cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_guess"}
LORA_RANK = 32 ; LORA_ALPHA = 32 ; LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj",
                  "in_proj","out_proj","up_proj","down_proj"]  # lm_head 被注释掉
```

- **`HARD_TYPES` 是整套方案的核心判断。** 它与数据集结论一致：决定排名的是 cryptarithm（两个子类都难）+ equation_numeric 的 guess 子类。注意 `equation_numeric_deduce` 被划进 easy（它能从上下文直接抄出规律，不算难）。这个集合决定了**哪些题在两阶段都全量训、哪些只在第一阶段训**。
- **`rank=32`** 顶到比赛上限，容量越大越能"背下"要记的东西（签名表、规律）。
- **`dropout=0`** 关掉 dropout：SFT 蒸馏要的是"精确复刻轨迹"，宁可过拟合也要学准。
- **`target_modules` 里出现 `in_proj` / `out_proj`** 是和架构挂钩的关键细节：它们是 **Mamba-2（SSM）块的输入输出投影**，普通 Llama 的 LoRA 配方里没有。这里把 LoRA 同时挂在 Attention 和 Mamba 两种层上，覆盖更全。

环境补丁（3 个 cell）是纯工程踩坑：离线装 Triton wheel、给 **Blackwell GPU 打 ptxas 补丁**（甚至 monkey-patch 让它谎报版本号 `'12.0'`）、离线装 `mamba_ssm` / `causal_conv1d`。它的存在本身就是一条信息：**新模型（2025-12 才发布）+ 新硬件（Blackwell）+ 不能联网 = 谁先把环境跑通谁就赢在起跑线。**

#### 阶段 1：数据切分（切成两个不重叠的集合）

```python
n = int(hard_df["type"].value_counts().min())   # 三个 hard 类型里最稀缺那个的题数
# 从每个 easy 类型随机抽 n 条留给 Nudge；其余全部 + 全部 hard 给 Epoch1
```

| | **Epoch1 集（Phase 1 用）** | **Nudge 集（Phase 2 用）** |
|---|---|---|
| Hard 题 | ✅ **全部** | ✅ **全部** |
| Easy 题 | ✅ 除被抽走的 `n×易题类数` 条外的剩余全部 | ✅ 每个易题类型**恰好 `n` 条** |

设计意图：（1）**难题在两阶段都全量出现**，反复夯实；（2）易题拆成互不重叠两半，**Phase 2 见到的是 Phase 1 没训过的新鲜易题**（防遗忘又不重复过拟合）；（3）Nudge 集**类型均衡**，适合做小步、均衡的收尾。

#### 阶段 2：构造 SFT 样本 `build_records`（全文最该逐字看的函数）

格式错一个字符，推理时答案抽取就可能 0 分。这个函数确立了**训练目标的"格式契约"**：

```python
cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()        # ② 剥掉原 CoT 自带的 \boxed
user_content = str(row["prompt"]) + PROMPT_SUFFIX                   # ③ 题面 + 强制后缀(对齐评测)
asst_content = cot_cleaned + f"\n</think>\n\\boxed{{{row['answer']}}}"  # ④ 用官方答案重拼 \boxed
```

一条完整训练目标长这样：

```
<think>                         ← chat template 自动加 (enable_thinking=True)
（上游思维链文本，已剥掉它自带的 \boxed）
</think>                        ← build_records 手动加
\boxed{官方答案}                 ← build_records 手动加(权威答案，不用上游可能写错的)
```

精髓在于把**思维过程**（用上游的）和**最终答案**（用官方的）解耦，并让这个结构**逐字等于评测协议**（评测从 `</think>` 后的 `\boxed{}` 抠答案）。后缀、`<think>/</think>` 配对、`\boxed{}` 三者构成"训练分布 = 评测分布"的契约，复盘改方案时**首先别动坏它**。

#### 阶段 3：分层采样器，让每个 batch "七类齐全"

`batch=1` + `grad_accum=8`，等效 batch=8。若随机洗牌，某个等效 batch 可能**全是同一类题**，梯度忽左忽右、对稀缺难题不利。解法是用**取模轮转（round-robin）**把每个类型的样本均匀撒到所有 batch 桶里（直观理解：发牌，一种花色一张一张轮流发给 N 个 batch）。

工程上刻意保持克制：不去 hack TRL 的内部洗牌逻辑，而是**预先算好一个固定样本顺序**，再用一个"照单全收"的 `PrecomputedOrderSampler` 喂给 DataLoader，`StratifiedSFTTrainer` 只重写 `get_train_dataloader` 一个方法。干净、可复现、改动面极小。

#### 阶段 4：Phase 1 训练（Train），大力出奇迹

```python
SFTConfig(
    learning_rate=2e-4,      # ① 大学习率
    warmup_steps=0,          # ② 不预热
    num_train_epochs=1,      # 只过一遍
    adam_beta2=0.95,         # ③ 比默认 0.999 小，对近期梯度更敏感
    max_grad_norm=1e9,       # ④ 等于关掉梯度裁剪
    neftune_noise_alpha=5.0, # ⑤ NEFTune 噪声
    packing=False,           # ⑥ 不打包(避免跨样本注意力泄漏)
)
```

Phase 1 的定位是"**狠狠地、快速地把轨迹砸进 LoRA**"：大 lr + 不预热 + **不裁剪梯度**，要的就是激进地把分布拉过去。代价是训练可能毛糙、不稳，这正是 Phase 2 要收拾的。训练完**立刻单独存档** `phase1_adapter`，既是保险，也作为"只训一阶段"的 A/B 对照候选。

#### 阶段 5：Phase 2 Nudge，小步轻推（整套方案最精妙处）

Phase 2 在 Phase 1 训完的**同一个 model 上继续训**（LoRA 权重接续，不重载），数据用 Nudge 集。两阶段对比：

| 参数 | Phase 1（Train） | Phase 2（Nudge） | 含义 |
|------|----------------|------------------|------|
| 学习率 | **2e-4** | **5e-6** | **差 40 倍**，Phase 2 极小步 |
| 调度器 | linear | **cosine** | Phase 2 平滑收尾 |
| warmup | 0 | **10** | Phase 2 先热身再动，更稳 |
| `max_grad_norm` | **1e9（不裁）** | **1.0（标准裁剪）** | Phase 2 收住梯度，防破坏 |
| 数据 | 难题全量 + 多数易题 | 难题全量 + 少量均衡易题 | Phase 2 聚焦难题 |
| NEFTune | 5.0 | 5.0 | 一致 |

**为什么分两阶段**：Phase 1 用大刀阔斧把所有题型"学个八九不离十"，但毛糙、在难题上差口气；Phase 2 用 **1/40 的学习率**在难题为主的均衡小集合上"小心翼翼再抠几分"，同时**掺入少量新鲜易题当锚防止灾难性遗忘**。一个管覆盖和速度，一个管精度和稳定。这比"单阶段一把梭"更能在决定排名的难题上压出边际分数，又不至于训崩或顾此失彼。

#### 阶段 6：打包提交 + 清理

存最终 adapter 后，**patch `adapter_config.json`** 是个不改就直接挂掉的坑：

```python
cfg["base_model_name_or_path"] = BASE_MODEL_NAME   # ① 训练时被写成本地缓存路径，必须改回官方名
cfg["inference_mode"] = True                        # ② 标记推理模式
cfg["lora_dropout"]   = 0.0                          # ③ 推理时 dropout 关死，防引入随机性
```

然后只把规则要求的两个文件 `adapter_config.json` + `adapter_model.safetensors` 打进 `submission.zip`（不含基座权重、不含 tokenizer，评测端自带）。

### 4.4 关键技术点

把整套方案的核心技术决策拎出来：

1. **轨迹蒸馏 SFT**：把"解题过程 + 答案"作为 SFT 目标灌进 LoRA，让模型在评测端（不能跑代码）自己复现解题。这是本场所有主流方案的共同范式。
2. **"训练分布 = 评测分布"的格式契约**：`PROMPT_SUFFIX`、`<think>/</think>` 配对、用官方答案重拼 `\boxed{}`，把训练目标逐字对齐评测协议。**这是最不能改错的地方。**
3. **两阶段 Train → Nudge**：lr 差 40 倍、裁剪从关到开、数据从全量收窄到聚焦难题。先猛后精，优于一把梭。
4. **针对架构的 LoRA target**：覆盖 Mamba 的 `in_proj/out_proj`，而不只是 Attention 的 `q/k/v/o_proj`，这是混合架构模型微调的关键细节。
5. **分层采样 + 类型均衡 Nudge 集**：确保稀缺的难题类型在训练信号里始终有稳定的存在感。
6. **NEFTune + dropout=0 的"矛盾"组合**：
   - **NEFTune（Noisy Embedding Fine-Tuning）**：训练时给 embedding 加均匀噪声（`X_noisy = X + (α/√(L·d))·ε`，本方案 α=5.0 偏保守），**只在训练时加、推理不加**。论文中它几乎零成本把 LLaMA-2-7B 的 AlpacaEval 胜率从 ~29% 拉到 ~64%。
   - **为什么本场特别适合**：题面塞满对抗性花絮（"Alice's Wonderland"、乱编故事、被替换的算子符号），这些是噪声不是信号。NEFTune 逼模型**透过花絮抓真正的规律**，而不是死记题面字串。
   - **看似矛盾实则各管一摊**：`dropout=0` 鼓励"记"（要背抽象规律），NEFTune 抑制"记"（要忘具体花絮），**要背的是抽象规律，要忘的是具体花絮。**
7. **不量化、纯 BF16 训练**：30B MoE 很吃显存，但量化会损精度，"要精确复刻轨迹"的任务选择不量化。

### 4.5 两类难题的解题逻辑：cryptarithm 与 bit manipulation

> cryptarithm 和 bit manipulation 是全场真正拉开分数的两类，0.86 → 0.89 几乎全部来自这里。下面把这两类的解题逻辑讲清楚。

- **bit manipulation（位运算）**，三个核心洞察：
  - （1）**逐 bit 串行**：逼模型把"8 位运算"拆成 8 个"单 bit 子问题"写出来，否则准确率暴跌到 ~9%；
  - （2）**列视角 + 验证**：把每个输出 bit 在所有示例上的取值看成一"竖列"，找能复现这列的布尔函数，再用 query 验证（防止偶然匹配）；
  - （3）**HEX 压缩省 token**：把 `01101001` 压成 `69`，砍掉约 28% token，省出预算搜更复杂的门（MAJ 等）。

- **cryptarithm（密码算术）**，
  - 天花板级难：朴素搜索空间约 `10! × 24³ ≈ 5×10¹⁰`，绝不可能在 7680 token 里逐步写完。
  - 破局点叫 **signature（签名）**：把等式中"哪些符号重复、出现在输出哪个位置"抽象成模式（如 `ABCCCDD`），**预先把每个签名对应的候选数字组合算好做成一张「签名目录」让模型背下来**；推理时模型不从头搜，而是"回忆"出候选再用 DFS 验证一致性。

- **这正是整场比赛的胜负手**：**"什么该让模型背下来（memorize）、什么该让它在 trace 里现算（compute）"**，把最贵的第一步从"暴力搜索"变成"查表记忆"。即便冠军方案，cryptarithm 也只有 ~30%(deduce)/~10%(guess) 可解率，**这点差距就是 0.86 与 0.89 之间那 3 分的主要来源。**

---

## 5. 简历竞赛经历模板

> 下面给一个可直接套用的模板，方括号 `[ ]` 处按你的真实排名 / 分数替换。建议放在简历"竞赛 / 项目经历"栏。

---

**NVIDIA Nemotron Model Reasoning Challenge（NVIDIA 推理挑战赛 · Kaggle）**
*2026.03 – 2026.06ㅤ|ㅤ参赛 4163 支队伍ㅤ|ㅤ[排名 Top x% / 铜·银·金牌]，Private LB Accuracy [0.8x]*

- **赛题背景 / 挑战**：在 NVIDIA 指定的混合架构 MoE 推理模型 **Nemotron-3-Nano-30B-A3B**（30B 总参 / 3.5B 激活，Mamba-Transformer 混合）之上微调 **LoRA adapter**，提升其在程序生成推理谜题上的解题准确率。核心难点：评测端**不能跑代码**（解题算法必须蒸馏进模型）、**单题生成上限 7680 token**、难题（密码算术 / 位运算）搜索空间高达 `~5×10¹⁰`。

- **方案设计与实现**：
  - **轨迹蒸馏 SFT**：将带解题思维链（CoT）的样本构造成「`<think>` 推理 + `</think>` + `\boxed{官方答案}`」格式，**逐字对齐评测协议**，用 SFT 把解题过程灌进 LoRA。
  - **两阶段微调 Train → Nudge**：Phase 1 用大学习率（2e-4）+ 关闭梯度裁剪激进灌入全部题型；Phase 2 用 1/40 学习率（5e-6）+ cosine 退火 + 标准裁剪，在**难题为主、类型均衡**的小集合上精修，并掺入新鲜易题**防止灾难性遗忘**。
  - **架构感知的 LoRA**：针对 Mamba-2 块挂上 `in_proj/out_proj`（而非仅 Attention 投影），rank 顶到合规上限 32。
  - **训练工程优化**：自定义**分层采样器**（round-robin 保证每个 batch 各类型均衡）、引入 **NEFTune** 噪声正则对抗题面对抗性花絮、纯 BF16 不量化以精确复刻轨迹；并攻克新模型 + Blackwell GPU 的离线环境依赖（Triton / ptxas / mamba_ssm 补丁）。

- **量化结果**：最终 Private LB Accuracy **[0.8x]**，在 4163 支队伍中取得 **[Top x% / 奖牌]**；通过本地分层 CV 对标 Private LB，规避了评分噪声导致的 Public LB 过拟合。

- **技术栈**：PyTorch · Unsloth · TRL（SFTTrainer）· PEFT-LoRA · NEFTune · vLLM（评测）· Mamba/MoE 混合架构 · 多阶段微调 · 分层采样。
