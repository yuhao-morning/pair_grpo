import torch
import torch.nn.functional as F
from trl import DPOTrainer

class SoftPairGRPOTrainer(DPOTrainer):
    def __init__(self, model, ref_model, args, tokenizer, beta=0.1, clip_range=0.2, **kwargs):
        super().__init__(model=model, ref_model=ref_model, args=args,  **kwargs)
        self.clip_range = clip_range

    def compute_loss(self, model, inputs, return_outputs=False):
        c_inputs = {k:v for k,v in inputs.items() if "chosen" in k}
        r_inputs = {k:v for k,v in inputs.items() if "rejected" in k}

        c_lg = model(**c_inputs).logits
        r_lg = model(**r_inputs).logits
        c_lp = self._get_logps(c_lg, c_inputs["attention_mask"])
        r_lp = self._get_logps(r_lg, r_inputs["attention_mask"])

        adv = c_lp - r_lp
        adv_clip = torch.clamp(adv, -self.clip_range, self.clip_range)
        loss = -adv_clip.mean()

        return (loss, (c_lg, r_lg)) if return_outputs else loss

    @staticmethod
    def _get_logps(logits, mask):
        lp = F.log_softmax(logits, -1)
        tok_lp = torch.gather(lp[:, :-1], -1, logits[:, 1:].argmax(-1).unsqueeze(-1)).squeeze(-1)
        return (tok_lp * mask[:,1:]).sum(-1)
