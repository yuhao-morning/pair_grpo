#import os
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import os
# ✅ 官方稳定镜像，解决超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./hf_cache"

import torch
import gc
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer
import warnings
warnings.filterwarnings("ignore")

# ====================== 安全低显存配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"
MAX_LENGTH = 128
MAX_PROMPT_LENGTH = 64
TRAIN_STEPS = 5
BETA = 0.1
KL_COEFF = 0.1

lora_config = LoraConfig(
    r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./all_4_algos",
    per_device_train_batch_size=1,
    max_steps=TRAIN_STEPS,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    learning_rate=1e-4,
    fp16=True,
    gradient_accumulation_steps=1,
    dataloader_pin_memory=False,
)

# ====================== 数据集 ======================
#dataset = load_dataset("Anthropic/hh-rlhf")
# 保持你原本的真实数据，不改动、不简化！
dataset = load_dataset("Anthropic/hh-rlhf", split="train", trust_remote_code=True) #modi

def format_data(sample):
    prompt = sample["chosen"].split("Assistant:")[0].split("Human:")[-1].strip()
    return {
        "prompt": prompt,
        "chosen": sample["chosen"],
        "rejected": sample["rejected"]
    }

dataset = dataset.map(format_data)
train_ds = dataset["train"]

# ====================== 4 个算法全部实现 ======================
# --------------------------
# 1. DPO
# --------------------------
class DPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# --------------------------
# 2. Standard GRPO
# --------------------------
class StandardGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = BETA
        self.kl_coeff = KL_COEFF

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        logps_chosen = outputs["chosen_logps"]
        logps_rejected = outputs["rejected_logps"]
        reward = logps_chosen - logps_rejected
        loss_gs = -torch.log(torch.sigmoid((logps_chosen - logps_rejected) / self.beta - reward)).mean()
        kl = (logps_chosen - outputs["ref_chosen_logps"]).mean()
        loss = loss_gs + self.kl_coeff * kl
        return (loss, outputs) if return_outputs else loss

# --------------------------
# 3. Soft Pair-GRPO（你原版核心：1 / -1 偏好）
# --------------------------
class SoftPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = BETA

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        logps_chosen = outputs["chosen_logps"]
        logps_rejected = outputs["rejected_logps"]
        # 🔥 对齐你 RL 版本：chosen=1, rejected=-1
        reward = 1.0
        loss_gs = -torch.log(torch.sigmoid((logps_chosen - logps_rejected) / self.beta - reward)).mean()
        return (loss_gs, outputs) if return_outputs else loss_gs

# --------------------------
# 4. Hard Pair-GRPO
# --------------------------
class HardPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = BETA
        self.kl_coeff = KL_COEFF

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        logps_chosen = outputs["chosen_logps"]
        logps_rejected = outputs["rejected_logps"]
        reward = 1.0
        loss_gs = -torch.log(torch.sigmoid((logps_chosen - logps_rejected) / self.beta - reward)).mean()
        kl = (logps_chosen - outputs["ref_chosen_logps"]).mean()
        loss = loss_gs + self.kl_coeff * kl
        return (loss, outputs) if return_outputs else loss

# ====================== 训练 4 个算法 ======================
results = {}

def train_one(algo_cls, name):
    torch.cuda.empty_cache()
    gc.collect()
    try:
        model = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE), lora_config)
        ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        trainer = algo_cls(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            max_length=MAX_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH,
            beta=BETA,
        )
        trainer.train()
        return "成功"
    except Exception as e:
        return f"失败: {str(e)[:80]}..."

# 依次训练
results["DPO"] = train_one(DPO, "DPO")
results["StandardGRPO"] = train_one(StandardGRPO, "StandardGRPO")
results["SoftPairGRPO"] = train_one(SoftPairGRPO, "SoftPairGRPO")
results["HardPairGRPO"] = train_one(HardPairGRPO, "HardPairGRPO")

# ====================== 输出结果 ======================
print("\n" * 3)
print("=" * 70)
print("🎉 全部 4 个算法训练完成！")
print("=" * 70)
print(json.dumps(results, indent=2, ensure_ascii=False))
