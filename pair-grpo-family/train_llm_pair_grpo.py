import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model

# ====================== 论文实验配置（LLaMA/Gemma 通用）======================
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # 可换：google/gemma-7b-it
TRAIN_DATA_PATH = "pair_data.json"             # 必须是 (prompt, chosen, rejected) 格式
BETA = 0.1                                     # 论文固定超参
CLIP_RANGE = 0.2                               # 论文固定超参
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4

# LoRA 大模型高效训练（论文标配）
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# 加载模型 & 分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model = get_peft_model(model, lora_config)
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results", per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION, learning_rate=LEARNING_RATE,
    num_train_epochs=3, logging_steps=10, fp16=True, report_to="none"
)

# ====================== 导入 4 个 Trainer ======================
from grpo_trainer import GRPOTrainer
from soft_pair_grpo_trainer import SoftPairGRPOTrainer
from hard_pair_grpo_trainer import HardPairGRPOTrainer

# 数据集（必须是 (prompt, chosen, rejected) 配对格式）
from datasets import load_dataset
dataset = load_dataset("json", data_files=TRAIN_DATA_PATH)["train"]

# ====================== 运行 4 种算法（论文完整对比）======================
TRAINERS = [
    ("DPO", DPOTrainer),
    ("Standard GRPO", GRPOTrainer),
    ("Soft Pair-GRPO", SoftPairGRPOTrainer),
    ("Hard Pair-GRPO", HardPairGRPOTrainer),
]

for name, trainer_cls in TRAINERS:
    print(f"\n===== 训练 {name} (论文实验) =====")
    trainer = trainer_cls(
        model=model, ref_model=ref_model, args=training_args,
        beta=BETA, clip_range=CLIP_RANGE, tokenizer=tokenizer,
        train_dataset=dataset, max_length=1024, max_prompt_length=512
    )
    trainer.train()
    trainer.save_model(f"./{name}_model")
