# ====================== 环境配置 ======================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# ====================== 基础配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"
USE_LORA = True

# ====================== LoRA 配置 ======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ====================== 【老版本专用】超精简训练参数 ======================
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    max_steps=5,
    logging_steps=1,
    save_total_limit=1,
    report_to="none",
    learning_rate=1e-4,
    gradient_accumulation_steps=1,
)

# ====================== 加载数据集 ======================
print("\n📥 加载 HH-RLHF 数据集...")
dataset = load_dataset("Anthropic/hh-rlhf")

# ====================== 加载模型 & 分词器 ======================
print("\n🚀 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)

if USE_LORA:
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# ====================== 【核心修复】所有训练器统一规则 ======================
results = {}

# ---------------------- DPO ----------------------
try:
    print("\n===== 训练 DPO =====")
    from dpo_trainer import DPOTrainer
    trainer = DPOTrainer(
        model, ref_model, training_args, dataset["train"]
    )
    trainer.train()
    results["DPO"] = "成功"
except Exception as e:
    print(f"❌ DPO 训练失败: {str(e)[:80]}...")
    results["DPO"] = str(e)

# ---------------------- GRPO ----------------------
try:
    print("\n===== 训练 Standard GRPO =====")
    from grpo_trainer import GRPOTrainer
    trainer = GRPOTrainer(
        model, ref_model, training_args, dataset["train"]
    )
    trainer.train()
    results["GRPO"] = "成功"
except Exception as e:
    print(f"❌ GRPO 训练失败: {str(e)[:80]}...")
    results["GRPO"] = str(e)

# ---------------------- Soft Pair-GRPO ----------------------
try:
    print("\n===== 训练 Soft Pair-GRPO =====")
    from soft_pair_grpo_trainer import SoftPairGRPOTrainer
    trainer = SoftPairGRPOTrainer(
        model, ref_model, training_args, dataset["train"]
    )
    trainer.train()
    results["Soft_Pair_GRPO"] = "成功"
except Exception as e:
    print(f"❌ Soft Pair-GRPO 训练失败: {str(e)[:80]}...")
    results["Soft_Pair_GRPO"] = str(e)

# ---------------------- Hard Pair-GRPO ----------------------
try:
    print("\n===== 训练 Hard Pair-GRPO =====")
    from hard_pair_grpo_trainer import HardPairGRPOTrainer
    trainer = HardPairGRPOTrainer(
        model, ref_model, training_args, dataset["train"]
    )
    trainer.train()
    results["Hard_Pair_GRPO"] = "成功"
except Exception as e:
    print(f"❌ Hard Pair-GRPO 训练失败: {str(e)[:80]}...")
    results["Hard_Pair_GRPO"] = str(e)

# ====================== 保存结果 ======================
with open("train_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ 全部流程执行完毕！结果已保存！")
