import torch
import torch.nn.functional as F
from trl import DPOTrainer

class DPOTrainer(DPOTrainer):
    def __init__(self, model, ref_model, args, beta=0.1, **kwargs):
        super().__init__(model=model, ref_model=ref_model, args=args, **kwargs)
        self.reward_history = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        chosen_inputs = {k:v for k,v in inputs.items() if "chosen" in k}
        rejected_inputs = {k:v for k,v in inputs.items() if "rejected" in k}
        print('outputs in dpo trainer ', outputs)
        c_lp = self._get_batch_logps(outputs["chosen_logits"], chosen_inputs["attention_mask"])
        r_lp = self._get_batch_logps(outputs["rejected_logits"], rejected_inputs["attention_mask"])
        reward = (c_lp - r_lp).sigmoid().mean().item()
        self.reward_history.append(reward)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _get_batch_logps(logits, mask):
        lp = F.log_softmax(logits, -1)
        tok_lp = torch.gather(lp[:, :-1], -1, logits[:,1:].argmax(-1).unsqueeze(-1)).squeeze(-1)
        return (tok_lp * mask[:,1:]).sum(-1)
