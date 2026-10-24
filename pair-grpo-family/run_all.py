import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# ====================== 配置（论文标准）======================
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # 可换 google/gemma-7b-it
USE_LORA = True
BATCH_SIZE = 4
LR = 5e-5
EPOCHS = 2
BETA = 0.1
CLIP = 0.2
DEVICE = "auto"

# LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# ====================== 1. 自动下载公开数据集 ======================
print("\n📥 下载 HH-RLHF 公开数据集...")
ds = load_dataset("Anthropic/hh-rlhf")
train_data = ds["train"].shuffle(seed=42).select(range(5000))  # 5k快速跑，论文用full
test_data = ds["test"].select(range(500))

# 格式化
def format_turns(example):
    prompt = example["chosen"].rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:"
    chosen = example["chosen"].split("\n\nAssistant:")[-1].strip()
    rejected = example["rejected"].split("\n\nAssistant:")[-1].strip()
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

train_data = train_data.map(format_turns)
test_data = test_data.map(format_turns)

# ====================== 2. 加载模型 ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
if USE_LORA:
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)

# ====================== 训练参数 ======================
training_args = TrainingArguments(
    output_dir="./train_out", per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4, learning_rate=LR, num_train_epochs=EPOCHS,
    logging_steps=10, fp16=True, save_strategy="no", report_to="none"
)

# ====================== 导入4个Trainer ======================
from dpo_trainer import DPOTrainer
from grpo_trainer import GRPOTrainer
from soft_pair_grpo_trainer import SoftPairGRPOTrainer
from hard_pair_grpo_trainer import HardPairGRPOTrainer

TRAINERS = [
    ("DPO", DPOTrainer),
    ("Standard GRPO", GRPOTrainer),
    ("Soft Pair-GRPO", SoftPairGRPOTrainer),
    ("Hard Pair-GRPO", HardPairGRPOTrainer),
]

results = {}

# ====================== 3. 训练全部4个算法 ======================
for name, trainer_cls in TRAINERS:
    print(f"\n🚀 训练 {name}...")
    if name in ["Standard GRPO", "Soft Pair-GRPO"]:
        trainer = trainer_cls(
            model=model, ref_model=ref_model, args=training_args, tokenizer=tokenizer,
            train_dataset=train_data, beta=BETA, clip_range=CLIP,
            max_length=1024, max_prompt_length=512
        )
    else:
        trainer = trainer_cls(
            model=model, ref_model=ref_model, args=training_args, tokenizer=tokenizer,
            train_dataset=train_data, beta=BETA,
            max_length=1024, max_prompt_length=512
        )
    trainer.train()
    results[name] = trainer.reward_history
    trainer.save_model(f"./{name}_final")

# ====================== 4. 绘制论文奖励对比图 ======================
plt.figure(figsize=(12, 6))
colors = ["blue", "orange", "green", "red"]
for i, (name, rewards) in enumerate(results.items()):
    df = pd.DataFrame({"step": range(len(rewards)), "reward": rewards})
    sns.lineplot(x="step", y="reward", data=df, label=name, linewidth=2.5, color=colors[i])

plt.title("LLM Alignment Reward Comparison (DPO vs GRPO vs Pair-GRPO)", fontsize=14)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Reward Score (Higher = Better)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reward_comparison.png", dpi=300)
print("\n✅ 对比图已保存：reward_comparison.png")

# ====================== 5. 保存结果JSON ======================
with open("train_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ 训练结果保存：train_results.json")
