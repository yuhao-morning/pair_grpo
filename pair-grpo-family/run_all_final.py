
#import os
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
#os.environ["HF_DATASETS_OFFLINE"] = "1"
# run_all_final.py 第一行就加这个

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
torch.cuda.set_per_process_memory_fraction(0.85, device=0)



#import os
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# 然后再导入
#from transformers import AutoTokenizer

import torch
torch.cuda.set_per_process_memory_fraction(20 / 23.64, device=0)
torch.cuda.set_per_process_memory_fraction(0.85, 0) 
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ====================== ✅ 修复：公开模型（无权限报错，直接用）======================
MODEL_ID = "google/gemma-2b-it"        # 小模型，超快
#MODEL_ID = "deepseek-ai/deepseek-llm-7b-chat"
#MODEL_ID = "facebook/opt-125m"
MODEL_ID = "Qwen/Qwen2.5-0.5B"#"Qwen/Qwen-7B-Chat" #"google/gemma-7b-it"         # 论文级 7B，效果同 LLaMA-7B

USE_LORA = True
BATCH_SIZE = 2 #4
LR = 5e-5
EPOCHS = 2
BETA = 0.1
CLIP = 0.2
DEVICE = "auto"

# LoRA 配置（Gemma / LLaMA 通用）
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Gemma 通用
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# ====================== 1. 自动下载公开数据集（HH-RLHF，100%公开）======================
print("\n📥 下载 HH-RLHF 公开数据集...")
ds = load_dataset("Anthropic/hh-rlhf")
train_data = ds["train"].shuffle(seed=42).select(range(5000))  # 5k快速训练
test_data = ds["test"].select(range(500))

# 格式化数据（自动转成 prompt/chosen/rejected）
def format_turns(example):
    try:
        prompt = example["chosen"].rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:"
        chosen = example["chosen"].split("\n\nAssistant:")[-1].strip()
        rejected = example["rejected"].split("\n\nAssistant:")[-1].strip()
    except:
        prompt = example["chosen"]
        chosen = example["chosen"]
        rejected = example["rejected"]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

train_data = train_data.map(format_turns)
test_data = test_data.map(format_turns)

# ====================== 2. 加载模型（✅ 无权限报错，公开模型）======================
print("\n🚀 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

#tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
)
if USE_LORA:
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
)

# ====================== 训练参数 ======================
'''
training_args = TrainingArguments(
    output_dir="./train_out",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    fp16=True,
    save_strategy="no",
    report_to="none",
    optim="paged_adamw_8bit",
)
'''
from trl import DPOConfig

training_args = DPOConfig(
        max_length = 256,#1024,
        beta = BETA
        #batch_size = BATCH_SIZE
)
# ====================== 导入4个Trainer（无修改）======================
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

# ====================== 3. 训练4种算法 ======================
for name, trainer_cls in TRAINERS:
    print(f"\n===== 训练 {name} =====")
    try:
        if name in ["Standard GRPO", "Soft Pair-GRPO"]:
            trainer = trainer_cls(
                model=model, ref_model=ref_model, args=training_args, #tokenizer=tokenizer,
                train_dataset=train_data, beta=BETA, #clip_range=CLIP,
                #max_length=1024, max_prompt_length=512
            )
        else:
            trainer = trainer_cls(
                model=model, ref_model=ref_model, args=training_args, #tokenizer=tokenizer,
                train_dataset=train_data, beta=BETA,
                #max_length=1024, max_prompt_length=512
            )
        trainer.train()
        results[name] = trainer.reward_history
        trainer.save_model(f"./{name}_model")
        print(f"✅ {name} 训练完成！")
    except Exception as e:
        print(f"❌ {name} 训练失败: {e}")
        traceback.print_exc()

# ====================== 4. 绘制论文奖励对比图 ======================
if len(results) > 0:
    plt.figure(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (alg, rewards) in enumerate(results.items()):
        df = pd.DataFrame({
            "step": range(len(rewards)),
            "reward": pd.Series(rewards).rolling(window=5).mean()  # 平滑曲线
        })
        sns.lineplot(x="step", y="reward", data=df, label=alg, linewidth=3, color=colors[i])

    plt.title("Reward Comparison (DPO vs GRPO vs Pair-GRPO)", fontsize=15)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Reward (Higher = Better)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("reward_comparison.png", dpi=300)
    print("\n🎉 论文对比图已保存：reward_comparison.png")

# ====================== 5. 保存结果 ======================
with open("train_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("✅ 训练结果保存：train_results.json")
