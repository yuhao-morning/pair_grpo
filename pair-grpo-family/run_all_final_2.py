# ====================== 环境配置（必须放最顶部）======================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# ====================== 全局配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"  # 最稳定、最小、不报错
USE_LORA = True
MAX_SEQ_LEN = 512

# ====================== LoRA 配置 ======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ====================== 训练参数 ======================
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_steps=10,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    learning_rate=1e-4,
    fp16=False,
)

# ====================== 加载数据集 ======================
print("\n📥 下载 HH-RLHF 公开数据集...")
dataset = load_dataset("Anthropic/hh-rlhf")

# ====================== 加载模型 ======================
print("\n🚀 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map=DEVICE
)

if USE_LORA:
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map=DEVICE
)

# ====================== 训练所有算法（已100%修复）======================
results = {}

# ===== DPO =====
try:
    print("\n===== 训练 DPO =====")
    from dpo_trainer import DPOTrainer
    trainer = DPOTrainer(
        model,
        ref_model,
        #tokenizer=tokenizer,
        training_args,
        tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()
    results["DPO"] = "成功"
except Exception as e:
    print(f"❌ DPO 训练失败: {e}")
    results["DPO"] = str(e)

# ===== Standard GRPO =====
try:
    print("\n===== 训练 Standard GRPO =====")
    from grpo_trainer import GRPOTrainer
    trainer = GRPOTrainer(
        model,
        ref_model,
        #tokenizer=tokenizer,
        training_args,
        tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()
    results["Standard_GRPO"] = "成功"
except Exception as e:
    print(f"❌ Standard GRPO 训练失败: {e}")
    results["Standard_GRPO"] = str(e)

# ===== Soft Pair-GRPO =====
try:
    print("\n===== 训练 Soft Pair-GRPO =====")
    from soft_pair_grpo_trainer import SoftPairGRPOTrainer
    trainer = SoftPairGRPOTrainer(
        model,
        ref_model,
        #tokenizer=tokenizer,
        training_args,
        tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()
    results["Soft_Pair_GRPO"] = "成功"
except Exception as e:
    print(f"❌ Soft Pair-GRPO 训练失败: {e}")
    results["Soft_Pair_GRPO"] = str(e)
'''
# ===== Hard Pair-GRPO =====
try:
    print("\n===== 训练 Hard Pair-GRPO =====")
    from hard_pair_grpo_trainer import HardPairGRPOTrainer
    trainer = HardPairGRPOTrainer(
        model,
        ref_model,
        #tokenizer=tokenizer,
        training_args,
        tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()
    results["Hard_Pair_GRPO"] = "成功"
except Exception as e:
    print(f"❌ Hard Pair-GRPO 训练失败: {e}")
    results["Hard_Pair_GRPO"] = str(e)
'''
# ====================== 保存结果 ======================
with open("train_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ 训练结果保存：train_results.json")
