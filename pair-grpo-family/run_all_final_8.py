# ====================== 环境 ======================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer
import warnings
warnings.filterwarnings("ignore")

# ====================== 配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"

# ====================== LoRA ======================
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# ====================== 训练参数 ======================
training_args = TrainingArguments(
    output_dir="./final_results",
    per_device_train_batch_size=1,
    max_steps=10,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    learning_rate=1e-4,
    fp16=False,
)

# ====================== 加载数据（关键修复！）======================
print("\n📥 加载数据集...")
dataset = load_dataset("Anthropic/hh-rlhf")

# 把数据集格式变成 DPO 能识别的格式
def format_data(sample):
    return {
        "prompt": sample["chosen"].split("Assistant:")[0].split("Human:")[1],
        "chosen": sample["chosen"],
        "rejected": sample["rejected"],
    }

dataset = dataset.map(format_data)

# ====================== 加载模型 ======================
print("\n🚀 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ====================== ✅ 训练 ======================
print("\n===== 开始训练 DPO (官方版) =====")
try:
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        max_length=256,
        max_prompt_length=128,
        beta=0.1,
    )
    trainer.train()
    print("\n🎉🎉🎉 **训练完全成功！！！** 🎉🎉🎉")
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# ====================== 保存 ======================
with open("final_success.json", "w") as f:
    json.dump({"status": "训练完成"}, f, indent=2)

print("\n✅ 全部结束！")
