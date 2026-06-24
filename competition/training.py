# %% [markdown]
# ## 两阶段微调(Two-Phase Fine-Tuning):Train → Nudge
#
# 先在完整数据集(扣掉留作 nudge 的那部分 easy 样本)上训练一个 LoRA adapter,
# 再用一个"以 hard 为主"的混合集去 nudge(轻推)它,在不遗忘 easy 题的前提下,
# 把难题型上的准确率往上压。
#
# **Phase 1 — Train:** 全部 hard 题 + 没有留给 nudge 的 easy 题。
# **Phase 2 — Nudge:** 全部 hard 题 + 每个 easy 类型各 n 条(n = 最少的那个 hard 类型的题数)。

# %%
# ════════════════════════════════════════════════════════════════════════════
#  全局配置(整套方案的"价值判断"都摊在这里)        详见 解决方案讲解.md §1
# ════════════════════════════════════════════════════════════════════════════
import os, sys

# Kaggle 日志默认编码可能不是 utf-8,思维链里有 ⌈⌉、希腊字母等非 ASCII 字符,
# 不强制 utf-8 会在打印时报 UnicodeEncodeError。这几行是纯防御,和算法无关。
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="strict")

SEED              = 42                                            # 全程固定随机种子,保证可复现
BASE_MODEL_NAME   = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"  # 比赛指定基座(提交时要写回 config,见末尾)
# ⚠️ 数据不是官方 train.csv(只有 id/prompt/answer),而是第三方数据集,
#    它多了一列 generated_cot —— 别人已生成好的解题思维链。本脚本只做"学生端"蒸馏。
DATASET_PATH      = "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv"

# 整套方案的核心赌注:认定这三类是"难题",要在两个阶段都重点训。   详见 §1.1
#   - cryptarithm_deduce / cryptarithm_guess:字母算术,两子类都难
#   - equation_numeric_guess:数字方程里只有 guess 子类算难(deduce 不在内)
HARD_TYPES        = {"cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_guess"}

# 评测时主办方会自动在题面后附加"把答案放进 \boxed{}"的指令。训练也拼上同样后缀,
# 让"训练输入 ≈ 评测输入"。输入分布对不齐,推理就可能不老实输出 \boxed{}。  详见 §1.2
PROMPT_SUFFIX     = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# ── LoRA 配置 ──────────────────────────────────────────────────────────────
LORA_RANK    = 32        # 比赛规则硬上限 rank ≤ 32,直接顶满:容量越大越能塞下要背的规律
LORA_ALPHA   = 32        # alpha/rank = 1.0,缩放系数为 1,不做放大
LORA_DROPOUT = 0.0       # 训练关 dropout:蒸馏要精确复刻轨迹,宁可过拟合也要学得准   详见 §1.3
# 注意 in_proj/out_proj —— 这是 Mamba-2(SSM)块的投影,普通 Llama LoRA 配方没有!
# q/k/v/o_proj 是 Attention 层,up/down_proj 是 MoE/MLP 层。三种层全覆盖,呼应混合架构。
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "in_proj", "out_proj", "up_proj", "down_proj"] # lm_head 被排除:省 rank 预算 + 改输出层风险大

# ── Phase 1「Train」:大力糙训,管覆盖和速度 ───────────────────────────────  详见 §7
TRAIN_LR            = 2e-4    # 大学习率,狠狠把轨迹砸进 LoRA
TRAIN_BATCH         = 1       # 单卡单条(30B 模型显存吃紧)
TRAIN_GRAD_ACCUM    = 8       # 梯度累积 8 步 → 等效 batch = 1×8 = 8
TRAIN_EPOCHS        = 1       # 只过一遍数据

# ── Phase 2「Nudge」:小步精修,管难题精度和稳定 ──────────────────────────  详见 §8
NUDGE_LR            = 5e-6    # 极小学习率,是 Phase 1 的 1/40,只"轻推"不推翻
NUDGE_BATCH         = 1
NUDGE_GRAD_ACCUM    = 8
NUDGE_EPOCHS        = 1

print("Config loaded.")
print(f"  Hard types: {HARD_TYPES}")
print(f"  Phase 1: lr={TRAIN_LR}, batch={TRAIN_BATCH}, accum={TRAIN_GRAD_ACCUM}")
print(f"  Phase 2: lr={NUDGE_LR},  batch={NUDGE_BATCH}, accum={NUDGE_GRAD_ACCUM}")


# %% [markdown]
# ## 环境准备(Setup)

# %%
# ════════════════════════════════════════════════════════════════════════════
#  环境补丁(纯工程踩坑:新模型 + 新硬件 Blackwell + 不能联网)   详见 解决方案讲解.md §2
#  这几段"丑"是合理的,反映的是比赛真实门槛:谁先把环境跑通谁就领先。
# ════════════════════════════════════════════════════════════════════════════
import os, glob, sys, subprocess, site

# Kaggle 评测环境不能联网,所有依赖必须从挂载的离线数据集里找 .whl 离线安装。
candidates = glob.glob("/kaggle/input/**/*triton*.whl", recursive=True)  # 在挂载目录里找 Triton 轮子
print("Found Triton wheels:", candidates)
if not candidates:
    raise FileNotFoundError("No Triton wheel found under /kaggle/input")
wheel = candidates[0]

# 装到独立目录再插进 sys.path,避免污染 / 覆盖系统自带的包版本。
target = "/kaggle/working/pydeps"
os.makedirs(target, exist_ok=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "--no-deps", "--target", target, "--upgrade", "--ignore-installed", wheel],  # --no-deps:只装它自己,不连带改依赖
    check=True,
)
if target not in sys.path:
    sys.path.insert(0, target)     # 让本次会话优先用这个目录里的 Triton
site.addsitedir(target)
print("Triton installed.")


# %%
# ── Blackwell GPU 的 ptxas 补丁 ────────────────────────────────────────────
#  Triton 自带的 ptxas(把 PTX 汇编编成 GPU 机器码的工具)还没适配 Blackwell 新架构,
#  这里换上一个能跑 Blackwell 的 ptxas 二进制,并骗过 Triton 的版本检查。
import sys, os, shutil, stat

sys.path.insert(0, '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script')  # 主办方提供的工具脚本目录

# 把 Blackwell 版 ptxas 拷到 /tmp 并加可执行权限(数据集挂载目录通常只读,不能直接 chmod)
ptxas_src = '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell'
ptxas_dst = '/tmp/ptxas-blackwell'
if os.path.exists(ptxas_src) and not os.path.exists(ptxas_dst):
    shutil.copy2(ptxas_src, ptxas_dst)
    os.chmod(ptxas_dst, os.stat(ptxas_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)  # +x 可执行
    # 整个 bin 目录也拷一份并全部加可执行权限
    src_bin = os.path.dirname(ptxas_src)
    dst_bin = '/tmp/triton_nvidia_bin'
    shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
    for f in os.listdir(dst_bin):
        fp = os.path.join(dst_bin, f)
        if os.path.isfile(fp):
            os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_dst
    import triton.backends.nvidia as nv_backend
    nv_backend.__file__ = os.path.join(dst_bin, '..', '__init__.py')   # 把 Triton 的后端指向新 bin 目录
    os.environ["TRITON_PTXAS_PATH"] = ptxas_dst

# 猴补丁(monkey-patch):让 Triton 谎报 ptxas 版本为 12.0,跳过它对真实版本的校验。
import triton.backends.nvidia.compiler as nv_compiler
nv_compiler.get_ptxas_version = lambda arch: '12.0'
print("Training environment fixes applied.")


# %%
# ── 离线安装训练依赖 ───────────────────────────────────────────────────────
import glob, os, subprocess, sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("GPU runtime required.")   # 没 GPU 直接停,别白跑

# 同样从离线数据集装核心训练栈(--no-index 完全禁用联网,--find-links 指向本地轮子目录)
packages_dir = "/kaggle/input/datasets/mayukh18/nemotron-packages/packages"
if not os.path.isdir(packages_dir):
    raise FileNotFoundError(f"Offline wheel directory not found: {packages_dir}")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "--no-index", "--find-links", packages_dir,
     "unsloth", "trl", "peft", "transformers", "datasets", "accelerate", "bitsandbytes"],  # 训练全家桶
    check=True,
)

# Mamba 专用 CUDA 内核要单独装(混合架构必须的,否则 SSM 层跑不起来)。
# 有多个版本时取排序最后一个(通常是最新)。
def _pick_last(pattern):
    wheels = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return wheels[-1] if wheels else None

for wheel in [_pick_last("causal*conv1d*.whl"), _pick_last("mamba_ssm-*.whl")]:  # Mamba 两个关键内核
    if wheel:
        print("Installing:", wheel)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", wheel], check=True)
    else:
        raise FileNotFoundError("Required wheel not found.")

print("All packages installed.")


# %% [markdown]
# ## 载入基座模型(Load Base Model)

# %%
# ════════════════════════════════════════════════════════════════════════════
#  载入基座模型 + 挂 LoRA                            详见 解决方案讲解.md §3
# ════════════════════════════════════════════════════════════════════════════
import torch
import kagglehub
from unsloth import FastLanguageModel   # Unsloth:对显存/速度做过优化的 HF 封装

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")  # 从 Kaggle 缓存拉基座
print(f"Model path: {MODEL_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=8192,        # 和评测 max_model_len=8192 对齐;模型原生支持 1M,这里主动砍到 8192
    load_in_4bit=False,         # 不量化 …
    load_in_8bit=False,         # … 纯 BF16 训练:量化会损精度,而本任务要精确复刻轨迹
    full_finetuning=False,      # 不全参微调,只训 LoRA
    trust_remote_code=True,     # Nemotron 用了自定义建模代码(混合架构),必须允许
    unsloth_force_compile=False,
    attn_implementation="eager",# 用最朴素的 attention:新模型+新硬件下,FlashAttention 等优化实现还不稳
    dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # 模型没定义 pad,拿 eos 顶上(标准做法)
print("Base model loaded.")


# %%
from unsloth import FastLanguageModel

# 把 §1.3 那套 LoRA 配置真正挂到基座上,得到一个只训练 LoRA 增量的 PEFT 模型。
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",                              # 不训练 bias 项
    use_gradient_checkpointing="unsloth",     # Unsloth 的省显存梯度检查点,这么大模型基本必开
    random_state=SEED,
)
model.print_trainable_parameters()            # 打印可训练参数量:你会看到只占总参数极小一部分


# %% [markdown]
# ## 数据切分(Dataset Split)
#
# 切出两个互不重叠的训练集:
# - **Epoch 1 set**:全部 hard 题 + 没有被抽去 nudge 的 easy 题
# - **Nudge set**:全部 hard 题 + 每个 easy 类型各 n 条(n = 最少的那个 hard 类型的题数)

# %%
# ════════════════════════════════════════════════════════════════════════════
#  数据切分:切成两份"不重叠"的训练集                详见 解决方案讲解.md §4
#    Epoch1 集(Phase 1 用)= 全部难题 + 大部分易题
#    Nudge 集 (Phase 2 用)= 全部难题 + 每个易题类型各 n 条新鲜样本(类型均衡)
# ════════════════════════════════════════════════════════════════════════════
import pandas as pd, random, re
from collections import defaultdict

df       = pd.read_csv(DATASET_PATH)
train_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)   # 整体打乱,固定种子可复现
print(f"Total dataset: {len(train_df)} rows")
print("Full type distribution:")
print(train_df["type"].value_counts().to_string())

# 按 HARD_TYPES 把数据劈成"难"和"易"两摊
hard_mask = train_df["type"].isin(HARD_TYPES)
hard_df   = train_df[hard_mask].reset_index(drop=True)
easy_df   = train_df[~hard_mask].copy()  # 注意:保留 train_df 的原始行号,后面靠它定位

# n = 三个难题类型里"最稀缺"那类的题数。它是给每个易题类型的采样配额,
#     目的是让 Nudge 集里 easy 和 hard 的量级拉到同一档,谁也不碾压谁。
n = int(hard_df["type"].value_counts().min())
print(f"\nHard type counts:")
print(hard_df["type"].value_counts().to_string())
print(f"\nn (samples per easy type for nudge set) = {n}")

# 从每个易题类型里随机抽 n 条,记下它们在 train_df 里的原始行号(只抽 easy,hard 一条不抽)
rng = random.Random(SEED + 1)            # 用 SEED+1 另起一个随机源,和上面的打乱解耦
nudge_easy_idx = set()                    # 被"留给 Nudge"的易题行号集合
for etype, egroup in easy_df.groupby("type"):
    sampled = rng.sample(list(egroup.index), min(n, len(egroup)))   # 该类型抽 n 条(不足 n 就全要)
    nudge_easy_idx.update(sampled)
    print(f"  Easy '{etype}': sampled {len(sampled)}/{len(egroup)} for nudge set")

# ── Epoch1 集:全部数据 减去 被留给 Nudge 的那些易题 ───────────────────────
#    → 等于:全部难题 + (全部易题 - 留给 Nudge 的易题)。难题一条不少。
epoch1_df = train_df[~train_df.index.isin(nudge_easy_idx)].reset_index(drop=True)

# ── Nudge 集:全部难题 + 刚抽出来的那批易题 ────────────────────────────────
nudge_easy_df = train_df[train_df.index.isin(nudge_easy_idx)]
nudge_df      = pd.concat([hard_df, nudge_easy_df]).reset_index(drop=True)

print(f"\nEpoch 1 set : {len(epoch1_df)} rows")
print("  Type distribution:", dict(sorted(epoch1_df["type"].value_counts().to_dict().items())))
print(f"\nNudge set   : {len(nudge_df)} rows")
print("  Type distribution:", dict(sorted(nudge_df["type"].value_counts().to_dict().items())))

# 守关键不变量:难题必须"全量"出现在两个集合里(这正是两阶段都重点训难题的基础)
assert len(epoch1_df[epoch1_df["type"].isin(HARD_TYPES)]) == len(hard_df), "Hard problems missing from epoch1 set"
assert len(nudge_df[nudge_df["type"].isin(HARD_TYPES)])   == len(hard_df), "Hard problems missing from nudge set"
print("\nSanity checks passed.")


# %% [markdown]
# ## 训练基础设施(Training Infrastructure)

# %%
# ════════════════════════════════════════════════════════════════════════════
#  训练基础设施:样本构造 + 对话模板 + 分层采样器     详见 解决方案讲解.md §5、§6
# ════════════════════════════════════════════════════════════════════════════
import math, re
from collections import defaultdict
from datasets import Dataset as HFDataset
from trl import SFTTrainer, SFTConfig
from torch.utils.data import DataLoader, Sampler


def build_records(source_df):
    """把 dataframe 转成 SFT 样本,并返回等长的 type 标签列表(给分层采样器用)。
    本函数是全文最不能改错的地方:格式错一个字符,推理时答案抽取就可能 0 分。  详见 §5
    """
    records, types = [], []
    for _, row in source_df.iterrows():
        cot = str(row["generated_cot"])
        # ① 数据清洗:上游生成的思维链可能是空 / "nan" / 几乎为空,跳过坏样本
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            continue
        # ② 剥掉思维链原文里自带的 \boxed{}:最终答案要用官方答案重拼,不信上游写的
        #    (等于把"推理过程"和"最终答案"解耦:过程用上游的,答案用官方的)
        cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
        # ③ user 端 = 题面 + 强制后缀,对齐评测输入
        user_content  = str(row["prompt"]) + PROMPT_SUFFIX
        # ④ assistant 端 = 清洗后的思维链 + 收尾 </think> + 官方答案的 \boxed{}
        #    注意:这里只补"收尾" </think>,开头的 <think> 由下面的 chat template 自动加!
        #    最终目标结构 = <think>(模板加) … 思维链 … </think>(这里加) \boxed{答案}(这里加)
        #    这个结构必须逐字等于评测协议,模型才会在推理时乖乖产出同样的格式。
        asst_content  = cot_cleaned + f"\n</think>\n\\boxed{{{str(row['answer'])}}}"
        records.append({"messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ]})
        types.append(str(row["type"]))    # ⑤ 平行记录每条样本的类型,喂给分层采样器
    return records, types


def formatting_prompts_func(example):
    """把 messages 套上 Nemotron 官方对话模板,转成训练用的纯文本。"""
    messages = example["messages"]
    # 兼容单条 / 批量两种传入形态,统一成"对话列表"
    if messages and isinstance(messages[0], dict):
        conversations = [messages]
    else:
        conversations = messages
    texts = []
    for conversation in conversations:
        try:
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False,
                add_generation_prompt=False,   # 这是训练:要完整对话(含答案)当目标,不是停下等续写
                enable_thinking=True)          # 打开思维模式:模板会在 assistant 前自动插入 <think>
        except TypeError:
            # 兜底:老版本 tokenizer 不认识 enable_thinking 参数
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return texts


def build_stratified_index_order(labels, batch_size, seed):
    """预先算好一个"类型均衡"的样本顺序:让每个等效 batch 内尽量各类型都摊到一点,
    避免某个 batch 全是同一类题导致梯度方向抖动。核心手法 = 发牌(round-robin)。  详见 §6
    """
    # 1) 按类型把样本下标分桶
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    # 2) 每个桶内部先打乱(吃 seed,可复现)
    rng = random.Random(seed)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)
    # 3) 准备 n_batches 个空 batch 桶,并把"桶的填充顺序"也打乱
    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches   = [[] for _ in range(n_batches)]
    b_order   = list(range(n_batches))
    rng.shuffle(b_order)
    # 4) 发牌:把每个类型的样本一张张轮流(取模)撒进各个 batch 桶
    #    → 像发扑克,一种花色逐张发给 N 家,发完再发下一种 → 每桶各花色都有
    assigned  = 0
    for label in sorted(by_label.keys()):
        for idx in by_label[label]:
            batches[b_order[assigned % n_batches]].append(idx)
            assigned += 1
    # 5) 把所有 batch 桶拍平成一维顺序返回
    order = [idx for batch in batches for idx in batch]
    if len(order) != len(labels):
        raise ValueError("Stratified order size mismatch")
    return order


class PrecomputedOrderSampler(Sampler):
    """一个"照单全收"的采样器:严格按给定的固定顺序吐样本下标,不做任何洗牌。"""
    def __init__(self, order): self.order = list(order)
    def __iter__(self):        return iter(self.order)
    def __len__(self):         return len(self.order)


class StratifiedSFTTrainer(SFTTrainer):
    """只重写 get_train_dataloader 的 SFTTrainer:把上面预算好的均衡顺序塞进 DataLoader,
    其余全部继承 TRL 的 SFTTrainer(改动面极小,不去 hack TRL 内部洗牌逻辑)。
    """
    def __init__(self, *args, stratified_order=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratified_order = stratified_order

    def get_train_dataloader(self):
        if self.stratified_order is None:
            return super().get_train_dataloader()     # 没给顺序就走 TRL 默认行为
        if len(self.stratified_order) != len(self.train_dataset):
            raise ValueError("Stratified order length does not match train dataset")
        # 自己拼一个 DataLoader,关键就是 sampler 用上面那个"照单全收"的固定顺序采样器
        dataloader_kwargs = {
            "batch_size":         self.args.per_device_train_batch_size,
            "sampler":            PrecomputedOrderSampler(self.stratified_order),  # ← 均衡顺序在此生效
            "collate_fn":         self.data_collator,
            "num_workers":        self.args.dataloader_num_workers,
            "pin_memory":         self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last":          self.args.dataloader_drop_last,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(self.train_dataset, **dataloader_kwargs)


print("Training infrastructure ready.")


# %% [markdown]
# ## Phase 1:Train
#
# 在"全部 hard 题 + 没有留给 nudge 的 easy 题"上训练。

# %%
# ════════════════════════════════════════════════════════════════════════════
#  Phase 1「Train」:大学习率 + 关梯度裁剪,大力糙训管覆盖   详见 解决方案讲解.md §7
# ════════════════════════════════════════════════════════════════════════════
import gc, time

# 用 Epoch1 集构造样本(§5),并转成 HF Dataset
epoch1_records, epoch1_types = build_records(epoch1_df)
epoch1_dataset = HFDataset.from_list(epoch1_records)
print(f"Epoch 1 records: {len(epoch1_records)}")
print("Type distribution:", dict(sorted(pd.Series(epoch1_types).value_counts().to_dict().items())))

training_args = SFTConfig(
    output_dir="/kaggle/working/phase1_output",
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH,
    gradient_accumulation_steps=TRAIN_GRAD_ACCUM,
    learning_rate=TRAIN_LR,             # 2e-4 大学习率
    lr_scheduler_type="linear",         # 线性衰减
    warmup_steps=0,                     # 不预热:只训 1 epoch,一上来就全速
    max_length=8192,                    # 超长样本截断到 8192,匹配推理窗口
    adam_beta1=0.9,
    adam_beta2=0.95,                    # 比默认 0.999 小:动量记忆更短、对近期梯度更敏感(LLM 常用)
    adam_epsilon=1e-8,
    weight_decay=0.0,                   # 不加权重衰减
    max_grad_norm=1e9,                  # ★ 10 亿 = 实际关掉梯度裁剪:Phase 1 不怕步子大,要的就是激进  §7④
    logging_steps=50,
    save_strategy="no",                 # 训练中不存档(末尾手动存)
    bf16=True,
    gradient_checkpointing=True,        # 省显存(用时间换显存)
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    remove_unused_columns=False,        # 保留自定义列(messages),别被 TRL 自动删掉
    seed=SEED,
    report_to="none",                   # 不上报 wandb 等
    packing=False,                      # ★ 不打包:防止不同题 token 跨样本串扰,保干净的样本边界  §7⑥
    neftune_noise_alpha=5.0,            # ★ NEFTune:embedding 加噪正则,抗死记硬背花絮  详见 §7 名词展开
)

# 等效 batch = 1×8 = 8,据此算出类型均衡的样本顺序(§6)
eff_batch_1   = TRAIN_BATCH * TRAIN_GRAD_ACCUM
strat_order_1 = build_stratified_index_order(epoch1_types, eff_batch_1, SEED)
print(f"Effective batch size: {eff_batch_1}")

trainer_phase1 = StratifiedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=epoch1_dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    stratified_order=strat_order_1,     # 把均衡顺序塞进自定义 Trainer
)

print("\nStarting Phase 1 training ...")
t0 = time.time()
trainer_phase1.train()
elapsed1 = time.time() - t0
print(f"Phase 1 done in {elapsed1/60:.1f} min")

# 训完清显存,为 Phase 2 腾地方
gc.collect()
import torch; torch.cuda.empty_cache()


# %%
# 单独存一次 Phase 1 的 adapter。两个用途:① 保险(Phase 2 万一训崩可退回);
# ② 把"只有 Phase 1"当成一个可提交的对照候选,方便 A/B 单阶段 vs 两阶段。  详见 §7 末
PHASE1_ADAPTER_DIR = "/kaggle/working/phase1_adapter"
model.save_pretrained(PHASE1_ADAPTER_DIR)
tokenizer.save_pretrained(PHASE1_ADAPTER_DIR)
print(f"Phase 1 adapter saved to {PHASE1_ADAPTER_DIR}")
for fname in os.listdir(PHASE1_ADAPTER_DIR):
    fpath = os.path.join(PHASE1_ADAPTER_DIR, fname)
    print(f"  {fname}: {os.path.getsize(fpath)/1024/1024:.1f} MB")

# %% [markdown]
# ## Phase 2:Nudge
#
# 用超低 LR,在"全部 hard 题 + 每个类型各 n 条 easy 题"上 nudge(轻推)已训好的 adapter。

# %%
# ⚠️ 整段被三引号注释掉 = 作者试过但最终放弃的实验。  详见 解决方案讲解.md §8.4
#    想法:Phase 2(精修)时把 LoRA dropout 从 0 提到 0.1,加点正则防过拟合。
#    最终没启用 → Phase 2 的正则只靠 NEFTune + 小 lr,不加 dropout。
#    复盘价值:这是个现成的、可重启的对照实验。
"""# ── Set dropout to 0.1 for nudge phase ───────────────────────────────────────
import torch.nn as nn

_updated = 0
for _name, _module in model.named_modules():
    if hasattr(_module, 'lora_dropout') and isinstance(_module.lora_dropout, nn.ModuleDict):
        for _adapter_name in _module.lora_dropout.keys():
            _module.lora_dropout[_adapter_name] = nn.Dropout(p=0.1)
            _updated += 1

if _updated == 0:
    raise RuntimeError("No LoRA dropout layers found — check model structure before proceeding.")

print(f"LoRA dropout set to 0.1 on {_updated} layers.")"""

# %%
# ════════════════════════════════════════════════════════════════════════════
#  Phase 2「Nudge」:极小 lr + 开梯度裁剪,在难题上精修同时防遗忘易题  详见 §8
#  注意:在 Phase 1 训完的"同一个 model"上继续训,LoRA 权重接着 Phase 1,不重载。
# ════════════════════════════════════════════════════════════════════════════
import gc, time

# 用 Nudge 集构造样本(§4:全部难题 + 每类 n 条新鲜易题,类型均衡)
nudge_records, nudge_types = build_records(nudge_df)
nudge_dataset = HFDataset.from_list(nudge_records)
print(f"Nudge records: {len(nudge_records)}")
print("Type distribution:", dict(sorted(pd.Series(nudge_types).value_counts().to_dict().items())))

# 和 Phase 1 对比,差异全在"求稳 + 精修"(见 §8 对比表):
nudge_args = SFTConfig(
    output_dir="/kaggle/working/phase2_output",
    num_train_epochs=NUDGE_EPOCHS,
    per_device_train_batch_size=NUDGE_BATCH,
    gradient_accumulation_steps=NUDGE_GRAD_ACCUM,
    learning_rate=NUDGE_LR,             # ★ 5e-6,Phase 1 的 1/40:只微调、不推翻已学到的东西
    lr_scheduler_type="cosine",         # ★ 余弦衰减,平滑收尾(典型的"冷却段")
    warmup_steps=10,                    # ★ 加一点预热,精修阶段更稳
    max_length=8192,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    weight_decay=0.0,
    max_grad_norm=1.0,                  # ★ 重新打开梯度裁剪(对比 Phase 1 的 1e9):收住梯度防破坏
    logging_steps=5,                    # 盯得更细(数据量小)
    save_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    remove_unused_columns=False,
    seed=SEED,
    report_to="none",
    packing=False,
    neftune_noise_alpha=5.0,            # NEFTune 两阶段都开,保持一致
)

eff_batch_2   = NUDGE_BATCH * NUDGE_GRAD_ACCUM
strat_order_2 = build_stratified_index_order(nudge_types, eff_batch_2, SEED)
print(f"Effective batch size: {eff_batch_2}")

trainer_phase2 = StratifiedSFTTrainer(
    model=model,                        # ← 同一个 model,接着 Phase 1 继续训
    args=nudge_args,
    train_dataset=nudge_dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    stratified_order=strat_order_2,
)

print("\nStarting Phase 2 nudge ...")
t0 = time.time()
trainer_phase2.train()
elapsed2 = time.time() - t0
print(f"Phase 2 done in {elapsed2/60:.1f} min")

gc.collect()
import torch; torch.cuda.empty_cache()


# %% [markdown]
# ## 保存与打包(Save & Package)

# %%
# ════════════════════════════════════════════════════════════════════════════
#  保存最终 adapter + 打补丁 + 打包 submission.zip    详见 解决方案讲解.md §9
# ════════════════════════════════════════════════════════════════════════════
import json, os, shutil, zipfile

ADAPTER_DIR = "/kaggle/working/final_adapter"
model.save_pretrained(ADAPTER_DIR)        # 存的是 Phase 1+2 之后的最终 LoRA
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved to {ADAPTER_DIR}")

# ── patch adapter_config.json:不改这里,评测端会直接挂掉 ──────────────────
cfg_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
# ① 训练时这个字段可能被写成本地缓存路径,评测机上不存在 → 必须写回官方基座名,
#    vLLM 才能正确把 adapter 挂到主办方提供的基座上。(最关键的一个坑)
cfg["base_model_name_or_path"] = BASE_MODEL_NAME
cfg["inference_mode"] = True              # ② 标记为推理模式
cfg["lora_dropout"]   = 0.0              # ③ 推理 dropout 必须为 0,防止任何残留随机性
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

# ── 打包:只装规则要求的两个文件,不含基座权重(基座评测端自带) ──────────
zip_path = "/kaggle/working/submission.zip"
required  = ["adapter_config.json", "adapter_model.safetensors"]  # 提交物就这两个
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in required:
        fpath = os.path.join(ADAPTER_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing: {fpath}")    # 缺文件早失败,别提交一个坏包
        zf.write(fpath, fname)
        print(f"  Added {fname} ({os.path.getsize(fpath)/1024/1024:.1f} MB)")

print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print("Done! Ready to submit.")


# %%
# ── 清理工作目录 ───────────────────────────────────────────────────────────
#  Kaggle 对 /kaggle/working 的产物大小有上限,删掉中间产物只留要用的。
import os, shutil, glob

KEEP = {
    "/kaggle/working/submission.zip",     # 最终提交包
    "/kaggle/working/phase1_adapter",     # 保留 Phase 1 adapter 作对照候选(见 §7 末)
}

for path in glob.glob("/kaggle/working/*"):
    if path in KEEP:
        continue
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"Deleted: {path}")
    except Exception as e:
        print(f"Could not delete {path}: {e}")    # 删不掉就跳过,不让清理步骤中断流程

print("\nRemaining files:")
for path in glob.glob("/kaggle/working/*"):
    print(f"  {path} ({os.path.getsize(path):,} bytes)")

# %%



