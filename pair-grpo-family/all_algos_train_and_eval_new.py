import argparse
import yaml
import os
import json
from tqdm import tqdm
from envs.make_env import make_env
from utils.logger import Logger
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from datetime import datetime

# ==========================
# 1. Actor + Critic 网络
# ==========================
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

# ==========================
# 2. 算法实现（完全保留你原来的）
# ==========================
class DPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=float(cfg["lr"]))
        self.clip_eps = float(cfg["clip_eps"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:
            act = dist.mean if self.continuous else dist.logits.argmax()
        else:
            act = dist.sample()
        logp = dist.log_prob(act)
        val = self.critic(obs)
        if self.continuous:
            return act.detach().numpy().squeeze(0), logp.sum().item(), val.item()
        else:
            return act.item(), logp.item(), val.item()

    def update(self, rollout):
        obs = torch.tensor(np.array(rollout["obs"]), dtype=torch.float32)
        acts = torch.tensor(np.array(rollout["acts"]), dtype=torch.float32 if self.continuous else torch.long)
        old_logps = torch.tensor(np.array(rollout["logps"]), dtype=torch.float32)
        rets = torch.tensor(np.array(rollout["rets"]), dtype=torch.float32)
        idxs = torch.randperm(len(obs))
        obs1, obs2 = obs, obs[idxs]
        act1, act2 = acts, acts[idxs]
        ret1, ret2 = rets, rets[idxs]
        old_logp1, old_logp2 = old_logps, old_logps[idxs]
        dist1 = self.actor.get_dist(obs1)
        dist2 = self.actor.get_dist(obs2)
        new_logp1 = dist1.log_prob(act1)
        new_logp2 = dist2.log_prob(act2)
        if self.continuous:
            new_logp1 = new_logp1.sum(-1)
            new_logp2 = new_logp2.sum(-1)
        log_ratio = new_logp1 - new_logp2 - (old_logp1 - old_logp2)
        delta = ret1 - ret2
        loss = -torch.log(1 + torch.exp(-log_ratio * delta)).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def save(self, path):
        torch.save({"actor":self.actor.state_dict(),"critic":self.critic.state_dict()}, path)

class PPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=float(cfg["lr"]))
        self.clip_eps = float(cfg["clip_eps"])
        self.kl_coeff = float(cfg["kl_coeff"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:
            act = dist.mean if self.continuous else dist.logits.argmax()
        else:
            act = dist.sample()
        logp = dist.log_prob(act)
        val = self.critic(obs)
        if self.continuous:
            return act.detach().numpy().squeeze(0), logp.sum().item(), val.item()
        else:
            return act.item(), logp.item(), val.item()

    def update(self, rollout):
        obs = torch.tensor(np.array(rollout["obs"]), dtype=torch.float32)
        acts = torch.tensor(np.array(rollout["acts"]), dtype=torch.float32 if self.continuous else torch.long)
        old_logps = torch.tensor(np.array(rollout["logps"]), dtype=torch.float32)
        advs = torch.tensor(np.array(rollout["advs"]), dtype=torch.float32)
        rets = torch.tensor(np.array(rollout["rets"]), dtype=torch.float32)
        old_logps = old_logps.detach()
        for _ in range(self.cfg["update_epochs"]):
            dist = self.actor.get_dist(obs)
            new_logps = dist.log_prob(acts)
            if self.continuous:
                new_logps = new_logps.sum(-1)
            vals = self.critic(obs)
            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advs
            actor_loss = -torch.min(surr1, surr2).mean()
            kl = torch.mean(old_logps - new_logps)
            actor_loss += self.kl_coeff * kl
            critic_loss = nn.MSELoss()(vals, rets)
            loss = actor_loss + 0.5 * critic_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()

    def save(self, path):
        torch.save({"actor":self.actor.state_dict(),"critic":self.critic.state_dict()}, path)

class StandardGRPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=float(cfg["lr"]))
        self.clip_eps = float(cfg["clip_eps"])
        self.kl_coeff = float(cfg["kl_coeff"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:
            act = dist.mean if self.continuous else dist.logits.argmax()
        else:
            act = dist.sample()
        logp = dist.log_prob(act)
        val = self.critic(obs)
        if self.continuous:
            return act.detach().numpy().squeeze(0), logp.sum().item(), val.item()
        else:
            return act.item(), logp.item(), val.item()

    def update(self, rollout):
        obs = torch.tensor(np.array(rollout["obs"]), dtype=torch.float32)
        acts = torch.tensor(np.array(rollout["acts"]), dtype=torch.float32 if self.continuous else torch.long)
        old_logps = torch.tensor(np.array(rollout["logps"]), dtype=torch.float32)
        advs = torch.tensor(np.array(rollout["advs"]), dtype=torch.float32)
        rets = torch.tensor(np.array(rollout["rets"]), dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        old_logps = old_logps.detach()
        for _ in range(self.cfg["update_epochs"]):
            dist = self.actor.get_dist(obs)
            new_logps = dist.log_prob(acts)
            if self.continuous:
                new_logps = new_logps.sum(-1)
            vals = self.critic(obs)
            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advs
            actor_loss = -torch.min(surr1, surr2).mean()
            kl = torch.mean(old_logps - new_logps)
            actor_loss += self.kl_coeff * kl
            critic_loss = nn.MSELoss()(vals, rets)
            loss = actor_loss #+ 0.5 * critic_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()

    def save(self, path):
        torch.save({"actor":self.actor.state_dict(),"critic":self.critic.state_dict()}, path)

class SoftPairGRPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=float(cfg["lr"]))
        self.clip_eps = float(cfg["clip_eps"])
        self.kl_coeff = float(cfg["kl_coeff"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:
            act = dist.mean if self.continuous else dist.logits.argmax()
        else:
            act = dist.sample()
        logp = dist.log_prob(act)
        val = self.critic(obs)
        if self.continuous:
            return act.detach().numpy().squeeze(0), logp.sum().item(), val.item()
        else:
            return act.item(), logp.item(), val.item()

    def update(self, rollout):
        obs = torch.tensor(np.array(rollout["obs"]), dtype=torch.float32)
        acts = torch.tensor(np.array(rollout["acts"]), dtype=torch.float32 if self.continuous else torch.long)
        old_logps = torch.tensor(np.array(rollout["logps"]), dtype=torch.float32)
        advs = torch.tensor(np.array(rollout["advs"]), dtype=torch.float32)
        rets = torch.tensor(np.array(rollout["rets"]), dtype=torch.float32)
        idx = torch.randperm(len(advs))
        pair_adv = advs - advs[idx]
        pair_adv = torch.where(pair_adv > 0, 1.0, -1.0)
        pair_adv = (pair_adv - pair_adv.mean()) / (pair_adv.std() + 1e-8)
        old_logps = old_logps.detach()
        for _ in range(self.cfg["update_epochs"]):
            dist = self.actor.get_dist(obs)
            new_logps = dist.log_prob(acts)
            if self.continuous:
                new_logps = new_logps.sum(-1)
            vals = self.critic(obs)
            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * pair_adv
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * pair_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            kl = torch.mean(old_logps - new_logps)
            actor_loss += self.kl_coeff * kl
            critic_loss = nn.MSELoss()(vals, rets)
            loss = actor_loss #+ 0.5 * critic_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()

    def save(self, path):
        torch.save({"actor":self.actor.state_dict(),"critic":self.critic.state_dict()}, path)

class HardPairGRPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=float(cfg["lr"]))
        self.clip_eps = float(cfg["clip_eps"])
        self.kl_coeff = float(cfg["kl_coeff"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:
            act = dist.mean if self.continuous else dist.logits.argmax()
        else:
            act = dist.sample()
        logp = dist.log_prob(act)
        val = self.critic(obs)
        if self.continuous:
            return act.detach().numpy().squeeze(0), logp.sum().item(), val.item()
        else:
            return act.item(), logp.item(), val.item()

    def update(self, rollout):
        obs = torch.tensor(np.array(rollout["obs"]), dtype=torch.float32)
        acts = torch.tensor(rollout["acts"], dtype=torch.long)
        old_logps = torch.tensor(rollout["logps"], dtype=torch.float32)
        advs = torch.tensor(rollout["advs"], dtype=torch.float32)
        rets = torch.tensor(rollout["rets"], dtype=torch.float32)
        idx = torch.randperm(len(advs))
        preferred_mask = advs > advs[idx]
        target_logps = old_logps.clone()
        delta = 0.1
        target_ps = torch.exp(target_logps)
        Delta = np.zeros(len(advs))
        Num = np.zeros(len(advs))
        for i in range(len(advs)):
            if preferred_mask[i]:
                delta = min([delta, 1 - target_ps[i] - 1e-6, target_ps[idx[i]] - 1e-6])
                Delta[i] += delta
                Num[i] += 1
                Delta[idx[i]] -= delta
                Num[idx[i]] += 1
            else:
                delta = min([delta, 1 - target_ps[i] - 1e-6, target_ps[idx[i]] - 1e-6])
                Delta[i] -= delta
                Num[i] += 1
                Delta[idx[i]] += delta
                Num[idx[i]] += 1
        target_ps += Delta
        target_logps = torch.log(target_ps)
        old_logps = old_logps.detach()
        target_logps = target_logps.detach()
        for _ in range(self.cfg["update_epochs"]):
            dist = self.actor.get_dist(obs)
            new_logps = dist.log_prob(acts)
            if self.continuous:
                new_logps = new_logps.sum(-1)
            kl_target = torch.mean(target_logps - new_logps)
            kl_reg = torch.mean(old_logps - new_logps)
            actor_loss = kl_target + self.kl_coeff * kl_reg
            vals = self.critic(obs)
            critic_loss = nn.MSELoss()(vals, rets)
            loss = actor_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()

    def save(self, path):
        torch.save({"actor":self.actor.state_dict(),"critic":self.critic.state_dict()}, path)

# ==========================
# 3. 工具函数
# ==========================
def compute_gae(rewards, dones, values, next_val, gamma, lam):
    advantages = []
    last_adv = 0.0
    for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
        delta = r + gamma * next_val * (1 - d) - v
        adv = delta + gamma * lam * (1 - d) * last_adv
        advantages.insert(0, adv)
        last_adv = adv
        next_val = v
    advantages = np.array(advantages)
    returns = advantages + np.array(values)
    return advantages, returns

def evaluate(agent, env, episodes=10):
    total = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ret = 0
        while not (done or trunc):
            a, _, _ = agent.select_action(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(a)
            ret += r
        total.append(ret)
    return np.mean(total), np.std(total)

# ==========================
# 4. 主函数（你原版训练循环 + 验证 + 保存最优）
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--algo", nargs='+', type=str, default="soft_pair_grpo",
                        choices=["dpo", "standard_grpo", "soft_pair_grpo", "hard_pair_grpo", "ppo"])
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for algo in args.algo:

        env = make_env(cfg["env_name"], cfg["seed"])
        eval_env = make_env(cfg["env_name"], cfg["seed"] + 100)

        obs_dim = env.observation_space.shape[0]
        continuous = isinstance(env.action_space, gymnasium.spaces.Box)
        act_dim = env.action_space.shape[0] if continuous else env.action_space.n

        if algo == "dpo":
            agent = DPO(obs_dim, act_dim, cfg, continuous)
        elif algo == "ppo":
            agent = PPO(obs_dim, act_dim, cfg, continuous)
        elif algo == "standard_grpo":
            agent = StandardGRPO(obs_dim, act_dim, cfg, continuous)
        elif algo == "soft_pair_grpo":
            agent = SoftPairGRPO(obs_dim, act_dim, cfg, continuous)
        elif algo == "hard_pair_grpo":
            agent = HardPairGRPO(obs_dim, act_dim, cfg, continuous)
        else:
            raise ValueError("algo not supported")

        log_dir = os.path.join("logs", cfg["env_name"], algo)
        os.makedirs(log_dir, exist_ok=True)
        logger = Logger(log_dir)
        rollout_size = cfg["rollout_size"]
        total_steps = cfg["total_timesteps"]
        global_step = 0
        pbar = tqdm(total=total_steps)

        best_eval = -np.inf
        eval_interval = cfg.get("eval_interval", 2048 * 5)
        eval_episodes = cfg.get("eval_episodes", 10)

        while global_step < total_steps:
            # ========================
            # 🎯 你原来的训练代码（完全不变！）
            # ========================
            obs, _ = env.reset()
            done = False
            trunc = False
            ep_ret = 0
            rollout = {"obs": [], "acts": [], "logps": [], "vals": [], "rews": [], "dones": []}

            for _ in range(rollout_size):
                act, logp, val = agent.select_action(obs)
                next_obs, rew, done, trunc, info = env.step(act)
                rollout["obs"].append(obs)
                rollout["acts"].append(act)
                rollout["logps"].append(logp)
                rollout["vals"].append(val)
                rollout["rews"].append(rew)
                rollout["dones"].append(done or trunc)

                obs = next_obs
                ep_ret += rew
                global_step += 1
                pbar.update(1)

                if done or trunc:
                    logger.add_scalar("reward/episode", ep_ret, global_step)
                    obs, _ = env.reset()
                    ep_ret = 0
                    done = False
                    trunc = False

            # GAE & Update
            _, _, next_val = agent.select_action(obs)
            advs, rets = compute_gae(rollout["rews"], rollout["dones"], rollout["vals"], next_val, cfg["gamma"], cfg["gae_lambda"])
            rollout["advs"] = advs
            rollout["rets"] = rets
            agent.update(rollout)

            # ========================
            # 🆕 我加的：验证 + 保存最优模型
            # ========================
            if global_step % eval_interval < 2048:
                mean_r, std_r = evaluate(agent, eval_env, eval_episodes)
                logger.add_scalar("eval/mean", mean_r, global_step)
                logger.add_scalar("eval/std", std_r, global_step)
                print(f"[{algo}] step {global_step} | eval {mean_r:.1f} ± {std_r:.1f}")

                if mean_r > best_eval:
                    best_eval = mean_r
                    save_path = os.path.join(log_dir, "best_model.pth")
                    agent.save(save_path)
                    print(f"✅ Best model saved: {best_eval:.1f}")

        pbar.close()
        logger.close()
        print(f"🏆 {algo} best eval = {best_eval}")
        print("✅ Done")

if __name__ == "__main__":
    main()
