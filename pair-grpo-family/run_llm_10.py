import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer
from trl.trainer.dpo_trainer import DPOBatch
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

# ====================== 核心配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"
MAX_LENGTH = 128
MAX_PROMPT_LENGTH = 64
TRAIN_STEPS = 5000
BETA = 0.1          # DPO beta
CLIP_EPS = 0.2      # PPO/GRPO Clip 系数
KL_COEFF = 0.1      # GRPO KL 正则系数
GAMMA = 0.99        # 折扣因子（对齐 all_algos）
LAM = 0.95          # GAE 系数（对齐 all_algos）

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

# ====================== 数据集处理（对齐偏好对逻辑） ======================
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

# ====================== 核心算法实现（严格对齐 all_algos.py） ======================
class DPO(DPOTrainer):
    """
    对齐 all_algos.py 中 DPO 逻辑：
    loss = -log(1 + exp(-log_ratio * delta))
    delta = ret1 - ret2（偏好对收益差）
    """
    def __init__(self, *args, **kwargs):
        kwargs.pop("beta", None)
        super().__init__(*args, **kwargs, beta=BETA)

    def compute_loss(self, model, inputs, return_outputs=False):
        batch = DPOBatch(**inputs)
        # 1. 获取 DPO 基础输出（chosen/rejected log prob）
        outputs = self.get_batch_loss_metrics(model, batch)
        chosen_logps = outputs["chosen_logps"]
        rejected_logps = outputs["rejected_logps"]
        
        # 2. 对齐 all_algos 中 DPO 核心损失
        log_ratio = chosen_logps - rejected_logps
        delta = 1.0  # LLM 场景下 chosen 收益 > rejected，固定 delta=1（对应 all_algos 中 ret1-ret2）
        loss = -torch.log(1 + torch.exp(-log_ratio * delta)).mean()
        
        if return_outputs:
            return loss, outputs
        return loss

class StandardGRPO(DPOTrainer):
    """
    对齐 all_algos.py 中 StandardGRPO 逻辑：
    1. PPO Clip 策略（surr1/surr2）
    2. KL 正则（old_logps - new_logps）
    3. Critic 损失（MSE(rets, vals)）
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)
        self.kl_coeff = KL_COEFF
        self.clip_eps = CLIP_EPS

    def compute_loss(self, model, inputs, return_outputs=False):
        batch = DPOBatch(**inputs)
        outputs = self.get_batch_loss_metrics(model, batch)
        
        # 1. 提取基础变量（对齐 all_algos 中 rollout 数据）
        chosen_logps = outputs["chosen_logps"]  # new_logps
        rejected_logps = outputs["rejected_logps"]
        old_logps = outputs["reference_chosen_logps"]  # 参考模型 logp（对应 all_algos 中 old_logps）
        
        # 2. 计算优势函数（对齐 all_algos 中 advs）
        advs = chosen_logps - rejected_logps
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        # 3. PPO Clip 核心（对齐 all_algos 中 surr1/surr2）
        ratio = torch.exp(chosen_logps - old_logps)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advs
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 4. KL 正则（对齐 all_algos 中 kl = mean(old_logps - new_logps)）
        kl = torch.mean(old_logps - chosen_logps)
        actor_loss += self.kl_coeff * kl
        
        # 5. Critic 损失（简化版：用 chosen 收益作为 ret，模型输出作为 val）
        vals = model(**batch.chosen_input_ids).logits.mean(-1).squeeze()
        rets = torch.ones_like(vals) * advs.mean()  # 简化收益标签
        critic_loss = F.mse_loss(vals, rets)
        
        # 总损失（对齐 all_algos：actor_loss + 0.5*critic_loss）
        loss = actor_loss + 0.5 * critic_loss

        if return_outputs:
            return loss, outputs
        return loss

class SoftPairGRPO(DPOTrainer):
    """
    对齐 all_algos.py 中 SoftPairGRPO 逻辑：
    1. 构造软偏好对（pair_adv = advs - advs[idx]）
    2. 二值化 pair_adv（>0→1，<0→-1）
    3. Clip + KL 正则
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)
        self.clip_eps = CLIP_EPS
        self.kl_coeff = KL_COEFF

    def compute_loss(self, model, inputs, return_outputs=False):
        batch = DPOBatch(**inputs)
        outputs = self.get_batch_loss_metrics(model, batch)
        
        # 1. 提取基础变量
        chosen_logps = outputs["chosen_logps"]
        old_logps = outputs["reference_chosen_logps"]
        
        # 2. 构造软偏好对（对齐 all_algos 中 idx = randperm）
        advs = chosen_logps - outputs["rejected_logps"]
        idx = torch.randperm(len(advs))
        pair_adv = advs - advs[idx]
        pair_adv = torch.where(pair_adv > 0, 1.0, -1.0)  # 软偏好二值化
        
        # 3. PPO Clip + KL 正则
        ratio = torch.exp(chosen_logps - old_logps)
        surr1 = ratio * pair_adv
        surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * pair_adv
        actor_loss = -torch.min(surr1, surr2).mean()
        
        kl = torch.mean(old_logps - chosen_logps)
        actor_loss += self.kl_coeff * kl
        
        # 4. Critic 损失
        vals = model(**batch.chosen_input_ids).logits.mean(-1).squeeze()
        rets = torch.ones_like(vals) * pair_adv.mean()
        critic_loss = F.mse_loss(vals, rets)
        
        loss = actor_loss + 0.5 * critic_loss

        if return_outputs:
            return loss, outputs
        return loss

class HardPairGRPO(DPOTrainer):
    """
    对齐 all_algos.py 中 HardPairGRPO 逻辑：
    1. 构造硬偏好对（preferred_mask = advs > advs[idx]）
    2. 调整目标分布（提高好动作概率，降低坏动作概率）
    3. KL 散度逼近目标分布 + 正则
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)
        self.kl_coeff = KL_COEFF
        self.delta = 0.1  # 对齐 all_algos 中 delta=0.1

    def compute_loss(self, model, inputs, return_outputs=False):
        batch = DPOBatch(**inputs)
        outputs = self.get_batch_loss_metrics(model, batch)
        
        # 1. 提取基础变量
        chosen_logps = outputs["chosen_logps"]
        old_logps = outputs["reference_chosen_logps"]
        advs = chosen_logps - outputs["rejected_logps"]
        
        # 2. 构造硬偏好对（对齐 all_algos 中 preferred_mask）
        idx = torch.randperm(len(advs))
        preferred_mask = advs > advs[idx]
        
        # 3. 调整目标分布（对齐 all_algos 中 target_ps 逻辑）
        target_ps = torch.exp(chosen_logps.detach())
        delta = self.delta
        
        # 3.1 安全调整 delta（避免概率越界）
        delta = torch.min(
            torch.tensor(delta, device=target_ps.device),
            1 - target_ps[preferred_mask] - 1e-6,
            target_ps[idx][preferred_mask] - 1e-6
        )
        
        # 3.2 硬偏好更新：提高优选动作概率，降低非优选
        target_ps = target_ps.clone()
        target_ps[preferred_mask] += delta
        target_ps[idx][preferred_mask] -= delta
        target_logps = torch.log(target_ps)
        
        # 4. KL 散度逼近目标（对齐 all_algos 中 kl_target + kl_reg）
        kl_target = torch.mean(target_logps - chosen_logps)  # KL(π_target || π_new)
        kl_reg = torch.mean(old_logps - chosen_logps)        # KL(π_old || π_new)
        actor_loss = kl_target + self.kl_coeff * kl_reg
        
        # 5. Critic 损失
        vals = model(**batch.chosen_input_ids).logits.mean(-1).squeeze()
        rets = torch.where(preferred_mask, 1.0, -1.0)  # 硬偏好标签
        critic_loss = F.mse_loss(vals, rets)
        
        loss = actor_loss + 0.5 * critic_loss

        if return_outputs:
            return loss, outputs
        return loss

# ====================== 训练逻辑（复用原有框架） ======================
results = {}

def train_one(algo_cls, name):
    torch.cuda.empty_cache()
    gc.collect()
    model = ref_model = tokenizer = trainer = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map=DEVICE, low_cpu_mem_usage=True
        )
        model = get_peft_model(model, lora_config)
        model.train()
        
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map=DEVICE, low_cpu_mem_usage=True
        )
        for param in ref_model.parameters():
            param.requires_grad = False
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        trainer = algo_cls(
            model=model, ref_model=ref_model, tokenizer=tokenizer, args=training_args,
            train_dataset=train_ds, max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH
        )

        # 训练
        trainer.train()

        # 保存 reward 日志
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

# ====================== 执行训练 ======================
print("🚀 开始训练 GRPO 系列算法（对齐 all_algos.py 逻辑）...")
train_one(DPO, "DPO")
train_one(StandardGRPO, "StandardGRPO")
train_one(SoftPairGRPO, "SoftPairGRPO")
train_one(HardPairGRPO, "HardPairGRPO")

print("\n" + "="*70)
print("🎉 全部训练完成！")
print("📊 reward 数据已保存为 JSON 文件！")
print("="*70)
print(json.dumps(results, indent=2))
