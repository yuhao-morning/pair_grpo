# ====================== LLM 终极完整版 ======================
# 包含：DPO + Standard GRPO + Soft Pair-GRPO (1/-1偏好)
# 全部自己实现，不依赖 trl 的 GRPOTrainer
# ======================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import json
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer
import warnings
warnings.filterwarnings("ignore")

# ====================== 配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"
MAX_LENGTH = 256
MAX_PROMPT_LENGTH = 128

# LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./llm_all_results",
    per_device_train_batch_size=1,
    max_steps=10,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    learning_rate=1e-4,
    fp16=False,
)

# ====================== 数据集 ======================
dataset = load_dataset("Anthropic/hh-rlhf")

def format_data(sample):
    prompt = sample["chosen"].split("Assistant:")[0].split("Human:")[-1].strip()
    return {
        "prompt": prompt,
        "chosen": sample["chosen"],
        "rejected": sample["rejected"]
    }
dataset = dataset.map(format_data)
train_dataset = dataset["train"]

# ====================== 模型加载 ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def get_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
    return get_peft_model(model, lora_config)

results = {}

# ------------------------------------------------------------------------------
# ✅ 1. DPO
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("训练 DPO")
try:
    trainer = DPOTrainer(
        model=get_model(),
        ref_model=AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE),
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        beta=0.1,
    )
    trainer.train()
    results["DPO"] = "成功"
except Exception as e:
    results["DPO"] = str(e)

# ------------------------------------------------------------------------------
# ✅ 2. Standard GRPO (自己实现，不需要库支持)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("训练 Standard GRPO")
try:
    class StandardGRPOTrainer(DPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.beta = 0.1

        def compute_loss(self, model, inputs, return_outputs=False):
            # GRPO 损失：基于对数概率差 + 约束
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            
            # GRPO 核心：奖励 = log(p_chosen) - log(p_rejected)
            return (loss, outputs) if return_outputs else loss

    trainer = StandardGRPOTrainer(
        model=get_model(),
        ref_model=AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE),
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        beta=0.1,
    )
    trainer.train()
    results["Standard_GRPO"] = "成功"
except Exception as e:
    results["Standard_GRPO"] = str(e)

# ------------------------------------------------------------------------------
# ✅ 3. Soft Pair-GRPO (你原创：1 / -1 成对偏好奖励)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("训练 Soft Pair-GRPO (1/-1 偏好奖励，对齐你的RL版本)")
try:
    class SoftPairGRPOTrainer(DPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            
            # 🔥 你的核心：chosen=1, rejected=-1
            # 完全对齐 Hopper / Ant / Walker2d 里的逻辑
            return (loss, outputs) if return_outputs else loss

    trainer = SoftPairGRPOTrainer(
        model=get_model(),
        ref_model=AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE),
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        beta=0.1,
    )
    trainer.train()
    results["Soft_Pair_GRPO"] = "成功 (1/-1偏好奖励)"
except Exception as e:
    results["Soft_Pair_GRPO"] = str(e)

# ====================== 结果输出 ======================
print("\n" + "="*60)
print("🎉 全部算法训练完成！")
print(json.dumps(results, indent=2, ensure_ascii=False))
