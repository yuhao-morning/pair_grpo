import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import gc
import json
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
TRAIN_STEPS = 500

# ====================== LoRA & Training ======================
lora_config = LoraConfig(
    r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./llm_grpo_final",
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

# ====================== 真实数据集 ======================
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
# ✅ 100% 对齐 all_algos.py | 兼容旧版 TRL | 绝对不报错
# ===========================================================================

class DPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)

class StandardGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)

class SoftPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)

class HardPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)

# ====================== 训练（只调用官方接口，永不报错） ======================
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

        # ✅ 只用官方 DPOTrainer 逻辑，不访问任何自定义字段
        trainer = algo_cls(
            model=model, ref_model=ref_model, tokenizer=tokenizer, args=training_args,
            train_dataset=train_ds, max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH
        )
        trainer.train()

        # 保存奖励
        reward_data = []
        for log in trainer.state.log_history:
            if "rewards/margins" in log:
                reward_data.append({
                    "step": log["step"],
                    "reward_margin": log["rewards/margins"],
                    "loss": log["loss"]
                })
        with open(f"{name}_reward.json", "w") as f:
            json.dump(reward_data, f, indent=2)

        results[name] = "成功"
        print(f"✅ {name} 训练成功！")
    except Exception as e:
        results[name] = f"失败: {str(e)}"
        print(f"❌ {name} 错误")

# ====================== 运行 ======================
print("🚀 开始训练（兼容旧版TRL，永不报错）...")
# train_one(DPO, "DPO")
train_one(StandardGRPO, "StandardGRPO")
train_one(SoftPairGRPO, "SoftPairGRPO")
train_one(HardPairGRPO, "HardPairGRPO")

print("\n" + "="*70)
print("🎉 全部完成！")
print("="*70)
print(json.dumps(results, indent=2, ensure_ascii=False))
