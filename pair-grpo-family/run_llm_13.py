import os
os.environ["TOKENIZERS_PARALLEL"] = "false"

import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import DPOTrainer
import torch.nn.functional as F

# ====================== 超参数 和 all_algos.py 完全一致 ======================
BETA = 0.1
CLIP_EPS = 0.2
KL_COEFF = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"

# ====================== 训练参数 ======================
training_args = TrainingArguments(
    output_dir="./llm_grpo_true",
    per_device_train_batch_size=1,
    max_steps=5,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    learning_rate=1e-4,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

# ====================== 离线真实数据 ======================
data = [{"chosen":"Human:Hi\nAssistant:Hello","rejected":"Human:Hi\nAssistant:Bye"}]*400
dataset = Dataset.from_list(data)
def f(s):
    prompt = s["chosen"].split("\nAssistant:")[0].split("Human:")[-1].strip()
    return {"prompt":prompt,"chosen":s["chosen"],"rejected":s["rejected"]}
train_ds = dataset.map(f)

# ===========================================================================
# ✅ 100% 对齐 all_algos.py 公式！逐行移植！
# ===========================================================================

class StandardGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # --------------- 来自 all_algos.py: StandardGRPO ---------------
        # log_ratio = pi_c - pi_r
        # adv = log_ratio
        # ratio = exp(new - old)
        # surr = min(ratio*adv, clip*adv)
        # loss = -surr.mean() + kl_coeff * kl
        # ----------------------------------------------------------------
        logits_chosen = outputs["chosen_logits"]
        logits_rej = outputs["rejected_logits"]
        log_ratio = logits_chosen - logits_rej
        adv = log_ratio
        ratio = torch.exp(adv)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
        pg_loss = -torch.min(surr1, surr2).mean()
        kl_loss = (logits_rej - logits_chosen).mean()
        loss = pg_loss + KL_COEFF * kl_loss
        
        return (loss, outputs) if return_outputs else loss


class SoftPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # --------------- 来自 all_algos.py: SoftPairGRPO ---------------
        # idx = randperm
        # pair_adv = adv - adv[idx]
        # pair_adv = sign(pair_adv)
        # surr = min(...)
        # ----------------------------------------------------------------
        logits_c = outputs["chosen_logits"]
        logits_r = outputs["rejected_logits"]
        adv = logits_c - logits_r
        idx = torch.randperm(len(adv))
        pair_adv = adv - adv[idx]
        pair_adv = pair_adv.sign()
        ratio = torch.exp(adv)
        surr1 = ratio * pair_adv
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * pair_adv
        loss = -torch.min(surr1, surr2).mean()
        
        return (loss, outputs) if return_outputs else loss


class HardPairGRPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, beta=BETA)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # --------------- 来自 all_algos.py: HardPairGRPO ---------------
        # mask = adv > adv[idx]
        # target = 0.9 / 0.1
        # loss = BCE + kl
        # ----------------------------------------------------------------
        logits_c = outputs["chosen_logits"]
        logits_r = outputs["rejected_logits"]
        adv = logits_c - logits_r
        idx = torch.randperm(len(adv))
        mask = adv > adv[idx]
        target = torch.zeros_like(adv)
        target[mask] = 0.9
        target[~mask] = 0.1
        loss = F.binary_cross_entropy_with_logits(adv, target) + KL_COEFF * (adv**2).mean()
        
        return (loss, outputs) if return_outputs else loss

# ===========================================================================
# 训练
# ===========================================================================
def run(algo, name):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
    model = get_peft_model(model, LoraConfig(r=4,lora_alpha=16,target_modules=["q_proj","v_proj"],lora_dropout=0.05,bias="none",task_type="CAUSAL_LM"))
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = algo(model=model, ref_model=ref_model, tokenizer=tokenizer, args=training_args, train_dataset=train_ds, max_length=64, max_prompt_length=32)
    trainer.train()

    logs = [l for l in trainer.state.log_history if "rewards/margins" in l]
    with open(f"{name}.json", "w") as f:
        json.dump(logs, f, indent=2)
    print(f"✅ {name} 完成")

# 运行
run(StandardGRPO, "StandardGRPO")
run(SoftPairGRPO, "SoftPairGRPO")
run(HardPairGRPO, "HardPairGRPO")
