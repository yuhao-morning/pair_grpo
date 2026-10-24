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
import matplotlib.pyplot as plt

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
# 2. 四大算法实现
# ==========================

# -----------------
# A. DPO
# -----------------
class DPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(cfg["lr"])
        )
        self.clip_eps = float(cfg["clip_eps"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:  # 验证时使用确定性策略
            if self.continuous:
                act = dist.mean
            else:
                act = dist.logits.argmax()
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
    
    def save_model(self, path):
        """保存策略模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'cfg': self.cfg,
            'continuous': self.continuous
        }, path)

# -----------------
# B. 标准 PPO
# -----------------
class PPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(cfg["lr"])
        )
        self.clip_eps = float(cfg["clip_eps"])
        self.kl_coeff = float(cfg["kl_coeff"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:  # 验证时使用确定性策略
            if self.continuous:
                act = dist.mean
            else:
                act = dist.logits.argmax()
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
    
    def save_model(self, path):
        """保存策略模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'cfg': self.cfg,
            'continuous': self.continuous
        }, path)

# -----------------
# B. 标准 GRPO
# -----------------
class StandardGRPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(cfg["lr"])
        )
        self.clip_eps = float(cfg["clip_eps"])
        self.kl_coeff = float(cfg["kl_coeff"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:  # 验证时使用确定性策略
            if self.continuous:
                act = dist.mean
            else:
                act = dist.logits.argmax()
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
            loss = actor_loss + 0.5 * critic_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()
    
    def save_model(self, path):
        """保存策略模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'cfg': self.cfg,
            'continuous': self.continuous
        }, path)

# -----------------
# C. Soft Pair GRPO
# -----------------
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
        self.clip_eps = float(cfg["clip_eps"])
        self.kl_coeff = float(cfg["kl_coeff"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:  # 验证时使用确定性策略
            if self.continuous:
                act = dist.mean
            else:
                act = dist.logits.argmax()
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
            loss = actor_loss + 0.5 * critic_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()
    
    def save_model(self, path):
        """保存策略模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'cfg': self.cfg,
            'continuous': self.continuous
        }, path)

class HardPairGRPO:
    def __init__(self, obs_dim, act_dim, cfg, continuous=False):
        self.cfg = cfg
        self.continuous = continuous
        self.actor = Actor(obs_dim, act_dim, continuous)
        self.critic = Critic(obs_dim)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(cfg["lr"])
        )
        self.clip_eps = float(cfg["clip_eps"])
        self.kl_coeff = float(cfg["kl_coeff"])

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        if deterministic:  # 验证时使用确定性策略
            if self.continuous:
                act = dist.mean
            else:
                act = dist.logits.argmax()
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
        acts = torch.tensor(np.array(rollout["acts"]), dtype=torch.long)
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
        
        for i in range(self.cfg["update_epochs"]):
            dist = self.actor.get_dist(obs)
            new_logps = dist.log_prob(acts)
            if self.continuous:
                new_logps = new_logps.sum(-1)
            
            kl_target = torch.mean(target_logps - new_logps)
            kl_reg = torch.mean(old_logps - new_logps)

            actor_loss = kl_target 
            actor_loss += self.kl_coeff * kl_reg
            vals = self.critic(obs)
            critic_loss = nn.MSELoss()(vals, rets)
            loss = actor_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return loss.item()
    
    def save_model(self, path):
        """保存策略模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'cfg': self.cfg,
            'continuous': self.continuous
        }, path)

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

def evaluate_agent(agent, env, eval_episodes=10):
    """
    验证智能体性能
    :param agent: 待验证的智能体
    :param env: 验证环境
    :param eval_episodes: 验证的回合数
    :return: 平均奖励、奖励标准差
    """
    total_rewards = []
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ep_ret = 0
        while not (done or trunc):
            # 验证时使用确定性策略
            act, _, _ = agent.select_action(obs, deterministic=True)
            obs, rew, done, trunc, _ = env.step(act)
            ep_ret += rew
        total_rewards.append(ep_ret)
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    return mean_reward, std_reward

def plot_results(results, save_path):
    """
    绘制不同算法的训练/验证奖励对比图
    :param results: 字典，key=算法名，value={train_rewards: [], eval_rewards: [], steps: []}
    :param save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制训练奖励
    plt.subplot(1, 2, 1)
    for algo_name, data in results.items():
        plt.plot(data["steps"], data["train_rewards"], label=algo_name)
    plt.xlabel("Steps")
    plt.ylabel("Training Reward")
    plt.title("Training Reward Comparison")
    plt.legend()
    plt.grid(True)
    
    # 绘制验证奖励
    plt.subplot(1, 2, 2)
    for algo_name, data in results.items():
        plt.plot(data["steps"], data["eval_rewards"], label=algo_name)
    plt.xlabel("Steps")
    plt.ylabel("Evaluation Reward (Mean)")
    plt.title("Evaluation Reward Comparison")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_results_summary(results, save_path):
    """
    保存结果汇总（包括最终奖励、最佳奖励等）
    :param results: 字典，key=算法名，value={train_rewards: [], eval_rewards: [], steps: []}
    :param save_path: 保存路径
    """
    summary = {}
    for algo_name, data in results.items():
        final_train_reward = data["train_rewards"][-1] if data["train_rewards"] else 0
        final_eval_reward = data["eval_rewards"][-1] if data["eval_rewards"] else 0
        best_eval_reward = max(data["eval_rewards"]) if data["eval_rewards"] else 0
        summary[algo_name] = {
            "final_train_reward": final_train_reward,
            "final_eval_reward": final_eval_reward,
            "best_eval_reward": best_eval_reward,
            "mean_eval_reward": np.mean(data["eval_rewards"]) if data["eval_rewards"] else 0
        }
    
    # 找出最优算法
    best_algo = max(summary.keys(), key=lambda x: summary[x]["best_eval_reward"])
    summary["best_algorithm"] = best_algo
    
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n=== 结果汇总 ===")
    for algo_name, stats in summary.items():
        if algo_name == "best_algorithm":
            print(f"最优算法: {stats}")
        else:
            print(f"\n{algo_name}:")
            print(f"  最终训练奖励: {stats['final_train_reward']:.2f}")
            print(f"  最终验证奖励: {stats['final_eval_reward']:.2f}")
            print(f"  最佳验证奖励: {stats['best_eval_reward']:.2f}")
            print(f"  平均验证奖励: {stats['mean_eval_reward']:.2f}")

# ==========================
# 4. 单算法训练函数
# ==========================
def train_algorithm(algo_name, cfg, env_name, seed):
    """训练单个算法并返回结果"""
    # 创建环境
    env = make_env(env_name, seed)
    eval_env = make_env(env_name, seed + 100)  # 验证环境（不同seed）
    
    # 获取维度信息
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gymnasium.spaces.Box):
        continuous = True
        act_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, gymnasium.spaces.Discrete):
        continuous = False
        act_dim = env.action_space.n
    else:
        raise ValueError("不支持的动作空间")

    # 选择算法
    if algo_name == "dpo":
        agent = DPO(obs_dim, act_dim, cfg, continuous)
    elif algo_name == "ppo":
        agent = PPO(obs_dim, act_dim, cfg, continuous)
    elif algo_name == "standard_grpo":
        agent = StandardGRPO(obs_dim, act_dim, cfg, continuous)
    elif algo_name == "soft_pair_grpo":
        agent = SoftPairGRPO(obs_dim, act_dim, cfg, continuous)
    elif algo_name == "hard_pair_grpo":
        agent = HardPairGRPO(obs_dim, act_dim, cfg, continuous)
    else:
        raise ValueError("算法不存在")

    # 初始化日志
    log_dir = os.path.join("logs", env_name, algo_name)
    logger = Logger(log_dir)
    rollout_size = cfg["rollout_size"]
    total_steps = cfg["total_timesteps"]
    eval_interval = cfg.get("eval_interval", 1000)  # 验证间隔（步数）
    eval_episodes = cfg.get("eval_episodes", 10)     # 每次验证的回合数
    
    global_step = 0
    pbar = tqdm(total=total_steps, desc=f"Training {algo_name}")
    
    # 记录奖励
    train_rewards = []
    eval_rewards = []
    step_records = []
    
    while global_step < total_steps:
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
                logger.log_reward(global_step, ep_ret)
                train_rewards.append(ep_ret)
                obs, _ = env.reset()
                ep_ret = 0
                done = False
                trunc = False

            # 定期验证
            if global_step % eval_interval == 0:
                mean_eval_ret, std_eval_ret = evaluate_agent(agent, eval_env, eval_episodes)
                eval_rewards.append(mean_eval_ret)
                step_records.append(global_step)
                logger.add_scalar("reward/eval_mean", mean_eval_ret, global_step)
                logger.add_scalar("reward/eval_std", std_eval_ret, global_step)
                tqdm.write(f"[{algo_name}] Step {global_step}: Eval Mean Reward = {mean_eval_ret:.2f} (±{std_eval_ret:.2f})")

        _, _, next_val = agent.select_action(obs)
        advs, rets = compute_gae(
            rollout["rews"], rollout["dones"], rollout["vals"],
            next_val, cfg["gamma"], cfg["gae_lambda"]
        )
        rollout["advs"] = advs
        rollout["rets"] = rets
        loss = agent.update(rollout)
        logger.add_scalar("loss/total", loss, global_step)

    # 训练结束后保存模型
    model_path = os.path.join(log_dir, f"{algo_name}_final_model.pth")
    agent.save_model(model_path)
    tqdm.write(f"模型已保存到: {model_path}")

    # 最终验证
    final_mean_eval, final_std_eval = evaluate_agent(agent, eval_env, eval_episodes * 2)
    eval_rewards.append(final_mean_eval)
    step_records.append(total_steps)
    tqdm.write(f"[{algo_name}] Final Eval: {final_mean_eval:.2f} (±{final_std_eval:.2f})")

    pbar.close()
    logger.close()
    
    return {
        "train_rewards": train_rewards,
        "eval_rewards": eval_rewards,
        "steps": step_records
    }

# ==========================
# 5. 主函数
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--algos", nargs="+", default=["ppo", "standard_grpo", "soft_pair_grpo", "hard_pair_grpo"],
                        choices=["dpo", "standard_grpo", "soft_pair_grpo", "hard_pair_grpo", "ppo"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    env_name = cfg["env_name"]
    results = {}
    
    # 训练指定的所有算法
    for algo in args.algos:
        print(f"\n========== 开始训练 {algo} ==========")
        results[algo] = train_algorithm(algo, cfg, env_name, args.seed)
    
    # 保存结果并绘制对比图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", env_name, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    # 绘制对比图
    plot_path = os.path.join(result_dir, "reward_comparison.png")
    plot_results(results, plot_path)
    print(f"\n奖励对比图已保存到: {plot_path}")
    
    # 保存结果汇总
    summary_path = os.path.join(result_dir, "results_summary.json")
    save_results_summary(results, summary_path)
    print(f"结果汇总已保存到: {summary_path}")
    
    print(f"\n✅ 所有算法训练完成！结果保存到：{result_dir}")

if __name__ == "__main__":
    main()
