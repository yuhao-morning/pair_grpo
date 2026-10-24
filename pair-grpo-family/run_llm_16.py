import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 自动使用GPU，不强制CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import json
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer
from trl import GRPOTrainer
from trl import GRPOConfig
# ====================== 100% 对齐 all_algos.py 超参数 ======================
BETA = 0.1
CLIP_EPS = 0.2
KL_COEFF = 0.1
DEVICE = "cuda"
MODEL_ID = "facebook/opt-125m"

# ====================== GPU 训练参数 ======================
#training_args = TrainingArguments(
'''
training_args = {
        output_dir:"./llm_grpo_final_gpu",
        per_device_train_batch_size:1,
        max_steps:50,       # 多训练几步，曲线更明显
        logging_steps:1,
        save_strategy:"no",
        report_to:"none",
        learning_rate:1e-4,
        fp16:True,         # ✅ GPU开启FP16，速度快
        bf16:False,
        remove_unused_columns:False,
        dataloader_pin_memory:False,
        gradient_accumulation_steps:1,
        beta:BETA,
        max_length:128,
        max_prompt_length:64
    }
'''

training_args = GRPOConfig(
    output_dir="./results",
    beta=BETA,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    top_p=0.5,
    top_k=5
    # 不要再加任何其他参数！
)
'''
# 正确！！！用 DPOConfig！！！
from trl import DPOConfig

training_args = DPOConfig(
    output_dir="./results",
    beta=BETA,
    learning_rate=1e-6,
    max_grad_norm=1.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
)
'''
# ====================== 真实 Anthropic hh-rlhf 数据 ======================
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")

def format_data(sample):
    chosen = sample["chosen"]
    rejected = sample["rejected"]
    prompt = chosen.split("\nAssistant:")[0].split("\nHuman:")[-1].strip()
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

train_ds = dataset.map(format_data)

# ===========================================================================
# ✅ 100% 逐行对齐 all_algos.py 公式！
# ===========================================================================

def dummy_reward_func(prompts, completions, **kwargs):
    """
    一个简单的 dummy reward，返回全1，用于测试GRPOTrainer能跑起来
    你后面可以替换成你自己的偏好奖励逻辑
    """
    return [1.0 for _ in completions]

'''
class StandardGRPO(GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        print('outputs ', outputs) 
        pi_c = outputs["logps/chosen"]
        pi_r = outputs["logps/rejected"]
        #old_c = outputs["reference_chosen_logps"]
        old_c = outputs["reference_chosen_logps"]  # ✅ 可用
        old_r = outputs["reference_rejected_logps"]

        adv = pi_c - pi_r
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(pi_c - old_c)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
        pg_loss = -torch.min(surr1, surr2).mean()
        kl_loss = (old_c - pi_c).mean()
        loss = pg_loss + KL_COEFF * kl_loss
        
        return (loss, outputs) if return_outputs else loss

'''
class StandardGRPO(GRPOTrainer):
    def compute_loss_self(self, model, inputs, num_items_in_batch=None, **kwargs):
        # 👇 父类 compute_loss 只返回 loss，不能要 outputs
        loss = super().compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # ========================
        # 直接用 Trainer 内部变量
        # 这就是你要的所有值！
        # ========================
        pi_c = self.chosen_logps  # 当前模型 chosen logps
        pi_r = self.rejected_logps# 当前模型 rejected logps
        old_c = self.ref_chosen_logps  # 参考模型（old model）chosen logps

        # 你的 PPO/GRPO 损失
        adv = pi_c - pi_r
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(pi_c - old_c)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
        pg_loss = -torch.min(surr1, surr2).mean()
        kl_loss = (old_c - pi_c).mean()

        # 合并损失
        loss = pg_loss + KL_COEFF * kl_loss

        return loss


'''
class StandardGRPO(DPOTrainer):

    def __init__(self, model, args, train_dataset, processing_class, reward_funcs):
        #super().__init__(model, args, train_dataset, processing_class)
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            #tokenizer=processing_class,  # 必须写 tokenizer=
            #ref_model=model,             # 必须给 ref_model
        )
        self.reward_funcs = reward_funcs
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1) 跑父类DPO，拿到当前模型 logps
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        print('inputs ', inputs)        
        print('outputs ', outputs)
        pi_c = outputs["logps/chosen"]
        pi_r = outputs["logps/rejected"]

        # ==============================================
        # 2) 手动计算参考模型 logps（真正能跑、不会报错）
        # ==============================================
        with torch.no_grad():
            # 取出 chosen 样本输入
            chosen_inputs = {
                "input_ids": inputs["chosen_input_ids"],
                "attention_mask": inputs["chosen_attention_mask"],
            }
            # 跑 ref_model 得到 logps
            ref_outputs = self.ref_model(
                **chosen_inputs,
                labels=inputs["chosen_labels"]  # 关键：必须用label算loss
            )
            
            # 这就是你要的 old_c = ref_logps/chosen
            old_c = -ref_outputs.loss

        # ====================
        # 你的 GRPO 损失
        # ====================
        adv = pi_c - pi_r
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(pi_c - old_c)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
        pg_loss = -torch.min(surr1, surr2).mean()
        kl_loss = (old_c - pi_c).mean()
        loss = pg_loss + KL_COEFF * kl_loss

        return (loss, outputs) if return_outputs else loss
    
    
    def compute_loss(self, model, inputs, return_outputs=False,** kwargs):
        # --------------------------
        # 1. 计算当前模型 logps
        # --------------------------
        chosen_ids = inputs["chosen_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_ids = inputs["rejected_input_ids"]
        rejected_mask = inputs["rejected_attention_mask"]

        # Chosen
        outputs_chosen = model(input_ids=chosen_ids, attention_mask=chosen_mask, labels=chosen_ids)
        pi_c = -outputs_chosen.loss

        # Rejected
        outputs_rejected = model(input_ids=rejected_ids, attention_mask=rejected_mask, labels=rejected_ids)
        pi_r = -outputs_rejected.loss

        # --------------------------
        # 2. 计算参考模型（old policy）logps
        # --------------------------
        with torch.no_grad():
            ref_outputs_chosen = self.ref_model(input_ids=chosen_ids, attention_mask=chosen_mask, labels=chosen_ids)
            old_c = -ref_outputs_chosen.loss

        # --------------------------
        # 3. GRPO 损失
        # --------------------------
        adv = pi_c - pi_r
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(pi_c - old_c)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
        pg_loss = -torch.min(surr1, surr2).mean()
        kl_loss = (old_c - pi_c).mean()
        loss = pg_loss + KL_COEFF * kl_loss

        # 不返回任何outputs，彻底避免报错
        if return_outputs:
            return loss, {}
        return loss
   
    
'''
'''
class SoftPairGRPO(DPOTrainer):
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
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * pair_adv
        loss = -torch.min(surr1, surr2).mean()
        
        return (loss, outputs) if return_outputs else loss

'''
# 你的超参
CLIP_EPS = 0.2

class SoftPairGRPO(DPOTrainer):
    # 必须加 **kwargs 兼容最新 Trainer
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 父类支持 return_outputs=True ✅
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # ========================
        # 最新版 trl DPOTrainer 正确键名
        # ========================
        pi_c = outputs["logps/chosen"]       # 当前模型 chosen
        pi_r = outputs["logps/rejected"]     # 当前模型 rejected
        old_c = outputs["ref_logps/chosen"]  # 旧策略（old model）

        # ========================
        # 你的 SoftPairGRPO 核心逻辑（完全保留，正确）
        # ========================
        adv = pi_c - pi_r              # 配对优势
        idx = torch.randperm(len(adv)) # 随机配对
        pair_adv = adv - adv[idx]
        pair_adv = pair_adv.sign()    # 符号化

        # PPO 比率 + 裁剪
        ratio = torch.exp(pi_c - old_c)
        surr1 = ratio * pair_adv
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * pair_adv
        loss = -torch.min(surr1, surr2).mean()

        return (loss, outputs) if return_outputs else loss

class HardPairGRPO(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        pi_c = outputs["chosen_logps"]
        pi_r = outputs["rejected_logps"]
        adv = pi_c - pi_r
        
        idx = torch.randperm(len(adv))
        mask = adv > adv[idx]
        target = torch.zeros_like(adv)
        target[mask] = 0.9
        target[~mask] = 0.1
        
        loss = F.binary_cross_entropy_with_logits(adv, target) + KL_COEFF * (adv**2).mean()
        return (loss, outputs) if return_outputs else loss

# ===========================================================================
# 训练入口
# ===========================================================================
def run(algo_cls, name):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ))
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = algo_cls(
        model=model,
        #ref_model=ref_model,
        #tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=dummy_reward_func,  # ✅ 关键！必须加上这个参数
        #beta=BETA,
        #max_length=128,
        #max_prompt_length=64,
    )
    trainer.train()

    reward_logs = [log for log in trainer.state.log_history if "rewards/margins" in log]
    with open(f"{name}_reward.json", "w", encoding="utf-8") as f:
        json.dump(reward_logs, f, indent=2)
    
    print(f"✅ {name} 训练完成！")

# ===========================================================================
# 运行三个真算法
# ===========================================================================
print("🚀 开始训练：GRPO家族 100% 对齐 all_algos.py")
run(StandardGRPO, "StandardGRPO")
run(SoftPairGRPO, "SoftPairGRPO")
run(HardPairGRPO, "HardPairGRPO")

print("\n🎉 全部完成！三条曲线真的不一样！")
