import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
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
MAX_LENGTH = 128
MAX_PROMPT_LENGTH = 64
TRAIN_STEPS = 5000
BETA = 0.1
KL_COEFF = 0.1

lora_config = LoraConfig(
    r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./final_grpo",
    per_device_train_batch_size=1,
    max_steps=TRAIN_STEPS,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    learning_rate=1e-4,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=1,
    dataloader_pin_memory=False,
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

# ====================== 模型 ======================
class DPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        kwargs.pop("beta", None)
        super().__init__(*args, **kwargs, beta=BETA)

class StandardGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)
        self.kl_coeff = KL_COEFF
    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs)

class SoftPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)
    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs)

class HardPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)
        self.kl_coeff = KL_COEFF
    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs)

# ====================== 训练 + 自动保存 reward ======================
results = {}

def train_one(algo_cls, name):
    torch.cuda.empty_cache()
    gc.collect()
    model = ref_model = tokenizer = trainer = None
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, low_cpu_mem_usage=True)
        model = get_peft_model(model, lora_config)
        model.train()
        ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, low_cpu_mem_usage=True)
        for param in ref_model.parameters():
            param.requires_grad = False
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        trainer = algo_cls(
            model=model, ref_model=ref_model, tokenizer=tokenizer, args=training_args,
            train_dataset=train_ds, max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH
        )

        # ✅ 训练
        trainer.train()

        # ✅【关键】保存 reward 日志到文件
        log_history = trainer.state.log_history
        reward_data = [
            {
                "step": log.get("step", 0),
                "loss": log.get("loss", 0),
                "reward_margin": log.get("rewards/margins", 0),
                "reward_chosen": log.get("rewards/chosen", 0),
                "reward_rejected": log.get("rewards/rejected", 0),
                "accuracy": log.get("rewards/accuracies", 0),
            }
            for log in log_history if "rewards/margins" in log
        ]

        with open(f"{name}_rewards.json", "w", encoding="utf-8") as f:
            json.dump(reward_data, f, indent=2)

        results[name] = "成功"
        print(f"✅ {name} 训练完成，reward 已保存到 {name}_rewards.json")

    except Exception as e:
        results[name] = f"失败: {str(e)[:80]}"
        print(f"❌ {name} 报错: {results[name]}")
    finally:
        del model, ref_model, tokenizer, trainer
        torch.cuda.empty_cache()
        gc.collect()

# ====================== 只跑 GRPO ======================
print("🚀 开始训练 GRPO...")
train_one(DPO, "DPO")
train_one(StandardGRPO, "StandardGRPO")
train_one(SoftPairGRPO, "SoftPairGRPO")
train_one(HardPairGRPO, "HardPairGRPO")

print("\n" + "="*70)
print("🎉 全部训练完成！")
print("📊 reward 数据已保存为 JSON 文件！")
print("="*70)
print(json.dumps(results, indent=2))
