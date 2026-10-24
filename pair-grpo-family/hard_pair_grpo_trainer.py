import torch
import torch.nn.functional as F
from trl import DPOTrainer

class HardPairGRPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        c_inputs = {k:v for k,v in inputs.items() if "chosen" in k}
        r_inputs = {k:v for k,v in inputs.items() if "rejected" in k}

        m_c = model(**c_inputs).logits
        m_r = model(**r_inputs).logits
        ref_c = self.ref_model(**c_inputs).logits
        ref_r = self.ref_model(**r_inputs).logits

        m_c_lp = self._get_logps(m_c, c_inputs["attention_mask"])
        m_r_lp = self._get_logps(m_r, r_inputs["attention_mask"])

        kl = F.kl_div(F.log_softmax(m_c, -1), F.softmax(ref_c, -1), reduction="none").sum(-1).mean()
        adv = (m_c_lp - m_r_lp) - self.beta * kl
        loss = -adv.mean()

        return (loss, (m_c, m_r)) if return_outputs else loss

    @staticmethod
    def _get_logps(logits, mask):
        lp = torch.log_softmax(logits, -1)
        tok_lp = torch.gather(lp[:, :-1], -1, logits[:,1:].argmax(-1).unsqueeze(-1)).squeeze(-1)
        return (tok_lp * mask[:,1:]).sum(-1)
