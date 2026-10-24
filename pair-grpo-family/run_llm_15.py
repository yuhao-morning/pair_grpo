import torch
import json
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer

# ====================== 完全和 all_algos.py 一致 ======================
BETA = 0.1
CLIP_EPS = 0.2
KL_COEFF = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/opt-125m"

training_args = TrainingArguments(
    output_dir="./final",
    per_device_train_batch_size=1,
    max_steps=5,
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    report_to="none",
)

# ====================== 真实数据 ======================
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
def format_fn(s):
    chosen = s["chosen"]
    rejected = s["rejected"]
    prompt = chosen.split("\n\nAssistant:")[0].split("\n\nHuman:")[-1]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
train_ds = dataset.map(format_fn)

# ===========================================================================
# ✅ 100% 对齐 all_algos.py 公式！
# ===========================================================================

class StandardGRPO(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        pi_c = outputs["chosen_logps"]
        pi_r = outputs["rejected_logps"]
        old_c = outputs["reference_chosen_logps"]
        adv = pi_c - pi_r
        ratio = torch.exp(pi_c - old_c)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
        pg_loss = -torch.min(surr1, surr2).mean()
        kl_loss = (old_c - pi_c).mean()
        loss = pg_loss + KL_COEFF * kl_loss
        return (loss, outputs) if return_outputs else loss

class SoftPairGRPO(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        pi_c = outputs["chosen_logps"]
        pi_r = outputs["rejected_logps"]
        adv = pi_c - pi_r
        idx = torch.randperm(len(adv))
        pair_adv = adv - adv[idx]
        pair_adv = pair_adv.sign()
        ratio = torch.exp(pi_c - outputs["reference_chosen_logps"])
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
# 训练
# ===========================================================================
def run(algo, name):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = algo(
        model=model, ref_model=ref_model, tokenizer=tokenizer,
        args=training_args, train_dataset=train_ds, beta=BETA,
        max_length=256, max_prompt_length=128
    )
    trainer.train()

    logs = [l for l in trainer.state.log_history if "rewards/margins" in l]
    with open(f"{name}.json", "w") as f:
        json.dump(logs, f)

# 运行
run(StandardGRPO, "StandardGRPO")
run(SoftPairGRPO, "SoftPairGRPO")
run(HardPairGRPO, "HardPairGRPO")
