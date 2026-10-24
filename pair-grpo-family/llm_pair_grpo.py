import torch
import transformers
from datasets import load_dataset
from trl import DPOTrainer, SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)

# ====================== 配置 ======================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "google/gemma-2-9b-it"  # 直接切换

DATASET = "trl-lib/ultrafeedback_binarized"
DEVICE = "cuda"
BATCH_SIZE = 2
LR = 5e-6
EPOCHS = 3
SEED = 42

# ====================== 加载模型 & 分词器 ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE)

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "right"

# ====================== 加载数据集 ======================
dataset = load_dataset(DATASET, split="test[:10%]")  # 小样本快速验证

def format_prompt(sample):
    return {
        "prompt": sample["prompt"],
        "chosen": sample["chosen"][-1]["content"],
        "rejected": sample["rejected"][-1]["content"]
    }

dataset = dataset.map(format_prompt)

# ====================== 训练参数 ======================
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_steps=50,
    fp16=False,
    bf16=True,
    gradient_accumulation_steps=4,
    seed=SEED
)

# ==============================================================================
# 1. 训练 DPO
# ==============================================================================
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,
)
print("=== Start Training DPO ===")
dpo_trainer.train()

# ==============================================================================
# 2. 训练 Standard GRPO
# ==============================================================================
from grpo_trainer import GRPOTrainer
grpo_trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,
)
print("=== Start Training Standard GRPO ===")
grpo_trainer.train()

# ==============================================================================
# 3. 训练 Soft Pair-GRPO
# ==============================================================================
from soft_pair_grpo_trainer import SoftPairGRPOTrainer
soft_trainer = SoftPairGRPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,
)
print("=== Start Training Soft Pair-GRPO ===")
soft_trainer.train()

# ==============================================================================
# 4. 训练 Hard Pair-GRPO
# ==============================================================================
from hard_pair_grpo_trainer import HardPairGRPOTrainer
hard_trainer = HardPairGRPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,
)
print("=== Start Training Hard Pair-GRPO ===")
hard_trainer.train()
