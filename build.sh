#!/bin/bash

PROJ_NAME="pair-grpo-family"
mkdir -p $PROJ_NAME/{configs,algos,utils,envs}

# === requirements.txt ===
cat > $PROJ_NAME/requirements.txt <<EOF
torch==2.3.0
gymnasium[box2d]==0.29.1
mujoco==2.3.7
pyyaml
numpy
matplotlib
tensorboard
tqdm
pandas
EOF

# === .gitignore ===
cat > $PROJ_NAME/.gitignore <<EOF
__pycache__/
logs/
csv_data/
plots/
checkpoints/
*.pyc
*.log
EOF

# === README.md ===
cat > $PROJ_NAME/README.md <<EOF
# Pair-GRPO Family
Official Implementation of Pair-GRPO Family (Soft-Pair GRPO & Hard-Pair GRPO)

## Installation
conda activate pairgrpo
pip install -r requirements.txt

## Run
python run.py --config configs/lunarlander.yaml --version soft
python run.py --config configs/lunarlander.yaml --version hard
EOF

# === configs ===
cat > $PROJ_NAME/configs/lunarlander.yaml <<EOF
env_name: LunarLander-v3
total_timesteps: 1000000
rollout_size: 2048
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_eps: 0.2
kl_coeff: 0.5
update_epochs: 10
mini_batch_size: 64
seed: 0
EOF

cat > $PROJ_NAME/configs/hopper.yaml <<EOF
env_name: Hopper-v4
total_timesteps: 2000000
rollout_size: 2048
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_eps: 0.2
kl_coeff: 0.5
update_epochs: 10
mini_batch_size: 64
seed: 0
EOF

cp $PROJ_NAME/configs/hopper.yaml $PROJ_NAME/configs/walker2d.yaml
sed -i 's/Hopper-v4/Walker2d-v4/' $PROJ_NAME/configs/walker2d.yaml

# === algos __init__ ===
touch $PROJ_NAME/algos/__init__.py

# === utils 三个文件 ===
cat > $PROJ_NAME/utils/logger.py <<EOF
import os
import csv
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(log_dir, "tb"))
        self.csv_path = os.path.join(log_dir, "reward.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "reward"])

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_reward(self, step, rew):
        self.csv_writer.writerow([step, rew])
        self.csv_file.flush()

    def close(self):
        self.writer.close()
        self.csv_file.close()
EOF

cat > $PROJ_NAME/utils/gae.py <<EOF
import numpy as np

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
EOF

cat > $PROJ_NAME/utils/plot.py <<EOF
import csv
import matplotlib.pyplot as plt
import os

def plot_curve(log_dir, save_name="reward_curve.png"):
    csv_path = os.path.join(log_dir, "reward.csv")
    steps = []
    rewards = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            rewards.append(float(row["reward"]))
    plt.figure(figsize=(10,5))
    plt.plot(steps, rewards, label="Episode Reward")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(log_dir, save_name), dpi=300)
    plt.close()
EOF

# === envs/make_env.py ===
cat > $PROJ_NAME/envs/make_env.py <<EOF
import gymnasium as gym

def make_env(env_name, seed=0):
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env
EOF

# === algos/pair_grpo_soft.py ===
cat > $PROJ_NAME/algos/pair_grpo_soft.py <<EOF
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
            lr=cfg["lr"]
        )

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        dist = self.actor.get_dist(obs)
        act = dist.sample()
        logp = dist.log_prob(act)
        val = self.critic(obs)
        if self.continuous:
            return act.detach().numpy(), logp.sum().item(), val.item()
        else:
            return act.item(), logp.item(), val.item()

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
EOF

# === algos/pair_grpo_hard.py ===
cat > $PROJ_NAME/algos/pair_grpo_hard.py <<EOF
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
EOF

# === 修复版 run.py（解决 IndexError 报错）===
cat > $PROJ_NAME/run.py <<EOF
import argparse
import yaml
import os
from tqdm import tqdm
from envs.make_env import make_env
from utils.logger import Logger
from utils.plot import plot_curve
from algos.pair_grpo_soft import SoftPairGRPO
from algos.pair_grpo_hard import HardPairGRPO
import gymnasium

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--version", type=str, default="soft", choices=["soft", "hard"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    env = make_env(cfg["env_name"], cfg["seed"])
    obs_dim = env.observation_space.shape[0]

    # ====================== 修复核心代码 ======================
    # 正确判断连续/离散动作空间，彻底解决 tuple index out of range
    # =========================================================
    if isinstance(env.action_space, gymnasium.spaces.Box):
        continuous = True
        act_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, gymnasium.spaces.Discrete):
        continuous = False
        act_dim = env.action_space.n
    else:
        raise ValueError("Unsupported action space type")

    if args.version == "soft":
        agent = SoftPairGRPO(obs_dim, act_dim, cfg, continuous)
    else:
        agent = HardPairGRPO(obs_dim, act_dim, cfg, continuous)

    log_dir = os.path.join("logs", cfg["env_name"], args.version)
    logger = Logger(log_dir)

    rollout_size = cfg["rollout_size"]
    total_steps = cfg["total_timesteps"]
    global_step = 0

    pbar = tqdm(range(total_steps))
    while global_step < total_steps:
        obs, _ = env.reset()
        done, trunc = False, False
        ep_ret = 0

        rollout = {"obs":[], "acts":[], "logps":[], "vals":[], "rews":[], "dones":[]}

        for _ in range(rollout_size):
            act, logp, val = agent.select_action(obs)
            next_obs, rew, done, trunc, _ = env.step(act)
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
                obs, _ = env.reset()
                ep_ret = 0
                done, trunc = False, False

        _, _, next_val = agent.select_action(obs)
        from utils.gae import compute_gae
        advs, rets = compute_gae(
            rollout["rews"], rollout["dones"], rollout["vals"],
            next_val, cfg["gamma"], cfg["gae_lambda"]
        )
        rollout["advs"] = advs
        rollout["rets"] = rets

        loss = agent.update(rollout)
        logger.add_scalar("loss/total", loss, global_step)

    logger.close()
    plot_curve(log_dir)
    print(f"Done! Log & plot saved to {log_dir}")

if __name__ == "__main__":
    main()
EOF

echo "✅ 完整 Pair-GRPO-Family 工程生成完毕！"
echo "👉 执行：cd pair-grpo-family && conda activate pairgrpo && pip install -r requirements.txt"
echo "👉 运行示例：python run.py --config configs/lunarlander.yaml --version soft"
