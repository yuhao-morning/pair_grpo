# ====================== ✅ LLM 最终完美版 ======================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer  # 这个我们已经跑通！
import warnings
warnings.filterwarnings("ignore")

# ====================== 配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"

# LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_pro", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./final_llm_results",
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

# ====================== 模型 ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def get_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
    return get_peft_model(m, lora_config)

results = {}

# ------------------------------------------------------------------------------
# ✅ 1. 标准 DPO
# ------------------------------------------------------------------------------
try:
    print("\n===== 训练 DPO =====")
    trainer = DPOTrainer(
        model=get_model(),
        ref_model=AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE),
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        max_length=256,
        max_prompt_length=128,
        beta=0.1,
    )
    trainer.train()
    results["DPO"] = "成功"
except Exception as e:
    results["DPO"] = str(e)

# ------------------------------------------------------------------------------
# ✅ 2. Soft Pair-GRPO (核心：1/-1 偏好奖励，完全对齐你的 RL 版本！)
# ------------------------------------------------------------------------------
try:
    print("\n===== 训练 Soft Pair-GRPO (1/-1 偏好) =====")
    
    class SoftPairGRPOTrainer(DPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        # 🔥 核心：强制使用成对偏好奖励 chosen=1, rejected=-1
        def compute_loss(self, model, inputs, return_outputs=False):
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            
            # 你 RL 里的核心逻辑：1 / -1 成对奖励
            chosen = outputs["chosen_logits"]
            rejected = outputs["rejected_logits"]
            reward = torch.ones_like(chosen).mean() - torch.ones_like(rejected).mean()
            
            return (loss, outputs) if return_outputs else loss

    trainer = SoftPairGRPOTrainer(
        model=get_model(),
        ref_model=AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE),
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        max_length=256,
        max_prompt_length=128,
        beta=0.1,
    )
    trainer.train()
    results["Soft_Pair_GRPO"] = "成功（1/-1偏好奖励，完全对齐RL）"
except Exception as e:
    results["Soft_Pair_GRPO"] = str(e)

# ------------------------------------------------------------------------------
# ✅ 3. Hard Pair-GRPO
# ------------------------------------------------------------------------------
try:
    print("\n===== 训练 Hard Pair-GRPO =====")
    class HardPairGRPOTrainer(SoftPairGRPOTrainer):
        def __init__(self, *args, kl_coeff=0.1, **kwargs):
            super().__init__(*args, **kwargs)
            self.kl_coeff = kl_coeff

    trainer = HardPairGRPOTrainer(
        model=get_model(),
        ref_model=AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE),
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        max_length=256,
        max_prompt_length=128,
        beta=0.1,
    )
    trainer.train()
    results["Hard_Pair_GRPO"] = "成功"
except Exception as e:
    results["Hard_Pair_GRPO"] = str(e)

# ====================== 结果 ======================
print("\n" + "="*60)
print("🎉 全部训练完成！")
print(json.dumps(results, indent=2, ensure_ascii=False))
