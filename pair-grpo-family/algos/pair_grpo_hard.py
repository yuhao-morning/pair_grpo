from algos.pair_grpo_soft import SoftPairGRPO, Actor, Critic
import torch
import torch.nn as nn

class HardPairGRPO(SoftPairGRPO):
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        super().__init__(obs_dim, act_dim, cfg, continuous)

    def update(self, rollout):
        obs = torch.tensor(rollout["obs"], dtype=torch.float32)
        acts = torch.tensor(rollout["acts"], dtype=torch.float32 if self.continuous else torch.int64)
        old_logps = torch.tensor(rollout["logps"], dtype=torch.float32)
        advs = torch.tensor(rollout["advs"], dtype=torch.float32)
        rets = torch.tensor(rollout["rets"], dtype=torch.float32)

        idx = torch.randperm(len(advs))
        pair_adv = advs - advs[idx]
        pair_adv = (pair_adv - pair_adv.mean()) / (pair_adv.std() + 1e-8)

        kl_coeff = self.cfg["kl_coeff"]
        for _ in range(self.cfg["update_epochs"]):
            dist = self.actor.get_dist(obs)
            new_logps = dist.log_prob(acts)
            if self.continuous:
                new_logps = new_logps.sum(-1)
            vals = self.critic(obs)

            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * pair_adv
            surr2 = torch.clamp(ratio, 1-self.cfg["clip_eps"], 1+self.cfg["clip_eps"]) * pair_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            kl_div = torch.mean(old_logps - new_logps)
            actor_loss += kl_coeff * kl_div

            critic_loss = nn.MSELoss()(vals, rets)
            total_loss = actor_loss + 0.5 * critic_loss

            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()
        return total_loss.item()
