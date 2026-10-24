import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from utils.gae import compute_gae

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, continuous=False):
        super().__init__()
        self.continuous = continuous
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        if continuous:
            self.mu = nn.Linear(128, act_dim)
            self.log_std = nn.Parameter(torch.zeros(1, act_dim))
        else:
            self.out = nn.Linear(128, act_dim)

    def get_dist(self, obs):
        x = self.fc(obs)
        if self.continuous:
            mu = self.mu(x)
            std = self.log_std.exp()
            return Normal(mu, std)
        else:
            logits = self.out(x)
            return Categorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, obs):
        return self.net(obs).squeeze()

class SoftPairGRPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(cfg["lr"])
        )

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        act = dist.sample()

        #act = act.squeeze(0)  # modified

        logp = dist.log_prob(act)
        val = self.critic(obs)
        if self.continuous:
            return act.detach().cpu().numpy(), logp.sum().item(), val.item()
        else:
            return act.item(), logp.item(), val.item()
    '''
    def select_action_2(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        act = dist.sample()
    
        # ✅ 必须先算 log_prob，再 squeeze！
        logp = dist.log_prob(act)
        val = self.critic(obs)
    
        # ✅ 现在再挤维度
        act = act.squeeze(0)

        if self.continuous:
            return act.detach().numpy(), logp.sum().item(), val.item()
        else:
            return act.item(), logp.item(), val.item()
    '''
    def update(self, rollout):
        obs = torch.tensor(rollout["obs"], dtype=torch.float32)
        acts = torch.tensor(rollout["acts"], dtype=torch.float32 if self.continuous else torch.int64)
        old_logps = torch.tensor(rollout["logps"], dtype=torch.float32)
        advs = torch.tensor(rollout["advs"], dtype=torch.float32)
        rets = torch.tensor(rollout["rets"], dtype=torch.float32)

        idx = torch.randperm(len(advs))
        pair_adv = advs - advs[idx]
        pair_adv = (pair_adv - pair_adv.mean()) / (pair_adv.std() + 1e-8)

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
            critic_loss = nn.MSELoss()(vals, rets)
            total_loss = actor_loss + 0.5 * critic_loss

            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()
        return total_loss.item()
