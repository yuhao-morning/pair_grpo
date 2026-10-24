import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import gc
import json
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer
import warnings
warnings.filterwarnings("ignore")

# ====================== 超参数 和 all_algos.py 完全一致 ======================
BETA = 0.1
CLIP_EPS = 0.2
KL_COEFF = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"
MAX_LENGTH = 128
MAX_PROMPT_LENGTH = 64
TRAIN_STEPS = 5

# ====================== LoRA ======================
lora_config = LoraConfig(
    r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./llm_grpo_correct",
    per_device_train_batch_size=1,
    max_steps=TRAIN_STEPS,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    learning_rate=1e-4,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=1,
    remove_unused_columns=False,
)

# ====================== 数据集 ======================
def format_data(sample):
    chosen = sample.get("chosen", "")
    rejected = sample.get("rejected", "")
    if "Human:" in chosen and "Assistant:" in chosen:
        prompt = chosen.split("Assistant:")[0].split("Human:")[-1].strip()
    else:
        prompt = "Unknown prompt"
    return {
        "prompt": prompt[:MAX_PROMPT_LENGTH],
        "chosen": chosen[:MAX_LENGTH],
        "rejected": rejected[:MAX_LENGTH]
    }

dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
dataset = dataset.map(format_data).filter(lambda x: x["prompt"] != "Unknown prompt")
train_ds = dataset

# ===========================================================================
# ✅ 下面这 4 个类 100% 对齐你 all_algos.py 的标准公式！
# ===========================================================================

class DPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)

class StandardGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)
        self.clip_eps = CLIP_EPS
        self.kl_coeff = KL_COEFF

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # 正确 GRPO 损失（对齐 all_algos）
        pi_c = outputs["chosen_logps"]
        pi_r = outputs["rejected_logps"]
        old_c = outputs["reference_chosen_logps"]
        
        adv = pi_c - pi_r
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(pi_c - old_c)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        pg_loss = -torch.min(surr1, surr2).mean()
        kl_loss = (old_c - pi_c).mean()
        loss = pg_loss + self.kl_coeff * kl_loss

        return (loss, outputs) if return_outputs else loss

class SoftPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)
        self.clip_eps = CLIP_EPS
        self.kl_coeff = KL_COEFF

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        pi_c = outputs["chosen_logps"]
        pi_r = outputs["rejected_logps"]
        old_c = outputs["reference_chosen_logps"]
        
        adv = pi_c - pi_r
        idx = torch.randperm(len(adv))
        pair_adv = adv - adv[idx]
        pair_adv = pair_adv.sign()
        
        ratio = torch.exp(pi_c - old_c)
        surr1 = ratio * pair_adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * pair_adv
        pg_loss = -torch.min(surr1, surr2).mean()
        kl_loss = (old_c - pi_c).mean()
        loss = pg_loss + self.kl_coeff * kl_loss

        return (loss, outputs) if return_outputs else loss

class HardPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)
        self.kl_coeff = KL_COEFF

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        pi_c = outputs["chosen_logps"]
        pi_r = outputs["rejected_logps"]
        old_c = outputs["reference_chosen_logps"]
        
        adv = pi_c - pi_r
        idx = torch.randperm(len(adv))
        mask = adv > adv[idx]
        
        target = torch.exp(pi_c.detach())
        target[mask] += 0.1
        target[idx[mask]] -= 0.1
        target = target.clamp(1e-8, 1)
        
        kl_target = (torch.log(target) - pi_c).mean()
        kl_reg = (old_c - pi_c).mean()
        loss = kl_target + self.kl_coeff * kl_reg

        return (loss, outputs) if return_outputs else loss

# ====================== 训练 ======================
results = {}

def train_one(algo_cls, name):
    torch.cuda.empty_cache()
    gc.collect()
    model = ref_model = tokenizer = trainer = None
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
        model = get_peft_model(model, lora_config)
        ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        trainer = algo_cls(
            model=model, ref_model=ref_model, tokenizer=tokenizer, args=training_args,
            train_dataset=train_ds, max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH
        )
        trainer.train()

        logs = [l for l in trainer.state.log_history if "rewards/margins" in l]
        with open(f"{name}_reward.json", "w") as f:
            json.dump(logs, f, indent=2)

        results[name] = "成功"
        print(f"✅ {name} 训练完成！")
    except Exception as e:
        results[name] = f"失败: {str(e)}"
        print(f"❌ {name} 报错")

# ====================== 运行 ======================
print("🚀 开始训练（正确对齐 all_algos.py）")
# train_one(DPO, "DPO")
train_one(StandardGRPO, "StandardGRPO")
train_one(SoftPairGRPO, "SoftPairGRPO")
train_one(HardPairGRPO, "HardPairGRPO")

print("\n🎉 全部完成！")
print(json.dumps(results, indent=2))
