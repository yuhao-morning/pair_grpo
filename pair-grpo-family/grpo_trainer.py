import torch
import torch.nn.functional as F
from trl import DPOTrainer

class GRPOTrainer(DPOTrainer):
    def __init__(self, model, ref_model, args, tokenizer, beta=0.1, clip_range=0.2, **kwargs):
        super().__init__(model=model, ref_model=ref_model, args=args,  **kwargs)
        self.clip_range = clip_range

    def compute_loss(self, model, inputs, return_outputs=False):
        print('inputs ', inputs)
        chosen_inputs = {k: v for k, v in inputs.items() if "chosen" in k}
        rejected_inputs = {k: v for k, v in inputs.items() if "rejected" in k}

        c_logits = model(**chosen_inputs).logits
        r_logits = model(**rejected_inputs).logits
        ref_c_logits = self.ref_model(**chosen_inputs).logits
        ref_r_logits = self.ref_model(**rejected_inputs).logits

        c_lp = self._get_logps(c_logits, chosen_inputs["attention_mask"])
        r_lp = self._get_logps(r_logits, rejected_inputs["attention_mask"])
        ref_c_lp = self._get_logps(ref_c_logits, chosen_inputs["attention_mask"])
        ref_r_lp = self._get_logps(ref_r_logits, rejected_inputs["attention_mask"])

        log_r = (c_lp - ref_c_lp) - (r_lp - ref_r_lp)
        adv = torch.exp(self.beta * log_r) - 1
        adv_clip = torch.clamp(adv, -self.clip_range, self.clip_range)
        loss = -adv_clip.mean()

        return (loss, (c_logits, r_logits)) if return_outputs else loss

    @staticmethod
    def _get_logps(logits, mask):
        lp = F.log_softmax(logits, dim=-1)
        tok_lp = torch.gather(lp[:, :-1], -1, logits[:, 1:].argmax(-1).unsqueeze(-1)).squeeze(-1)
        return (tok_lp * mask[:, 1:]).sum(-1)
