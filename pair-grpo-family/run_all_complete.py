import argparse
import yaml
import os
import subprocess
import sys
from multiprocessing import Pool
from all_algos import DPO, StandardGRPO, SoftPairGRPO, HardPairGRPO  # 导入算法类
from envs.make_env import make_env
from utils.logger import Logger
import gymnasium
import numpy as np
import torch
from tqdm import tqdm

# ==========================
# 1. 复用核心训练逻辑（封装为函数）
# ==========================
def train_algo(cfg_path, algo_name, env_name, seed):
    """
    单算法单环境训练函数（复用all_algos.py的main逻辑）
    """
    # 加载配置
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = seed  # 覆盖seed确保可复现
    cfg["env_name"] = env_name

    # 创建环境
    env = make_env(cfg["env_name"], cfg["seed"])
    obs_dim = env.observation_space.shape[0]
    
    # 判断动作空间类型
    if isinstance(env.action_space, gymnasium.spaces.Box):
        continuous = True
        act_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, gymnasium.spaces.Discrete):
        continuous = False
        act_dim = env.action_space.n
    else:
        raise ValueError(f"不支持的动作空间: {type(env.action_space)}")

    # 初始化算法
    if algo_name == "dpo":
        agent = DPO(obs_dim, act_dim, cfg, continuous)
    elif algo_name == "standard_grpo":
        agent = StandardGRPO(obs_dim, act_dim, cfg, continuous)
    elif algo_name == "soft_pair_grpo":
        agent = SoftPairGRPO(obs_dim, act_dim, cfg, continuous)
    elif algo_name == "hard_pair_grpo":
        agent = HardPairGRPO(obs_dim, act_dim, cfg, continuous)
    else:
        raise ValueError(f"未知算法: {algo_name}")

    # 初始化日志
    log_dir = os.path.join("logs", cfg["env_name"], algo_name, f"seed_{seed}")
    logger = Logger(log_dir)
    rollout_size = cfg["rollout_size"]
    total_steps = cfg["total_timesteps"]
    global_step = 0
    pbar = tqdm(total=total_steps, desc=f"{env_name}-{algo_name}-seed{seed}")

    # 核心训练循环
    while global_step < total_steps:
        obs, _ = env.reset()
        done = False
        trunc = False
        ep_ret = 0
        rollout = {"obs": [], "acts": [], "logps": [], "vals": [], "rews": [], "dones": []}

        for _ in range(rollout_size):
            act, logp, val = agent.select_action(obs)
            next_obs, rew, done, trunc, info = env.step(act)
            
            # 收集rollout数据
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

            # 回合结束处理
            if done or trunc:
                logger.add_scalar("reward/episode", ep_ret, global_step)
                logger.log_reward(global_step, ep_ret)
                obs, _ = env.reset()
                ep_ret = 0
                done = False
                trunc = False

        # 计算GAE优势函数和回报
        _, _, next_val = agent.select_action(obs)
        advs, rets = compute_gae(
            rollout["rews"], rollout["dones"], rollout["vals"],
            next_val, cfg["gamma"], cfg["gae_lambda"]
        )
        rollout["advs"] = advs
        rollout["rets"] = rets
        
        # 算法更新
        loss = agent.update(rollout)
        logger.add_scalar("loss/total", loss, global_step)

    # 收尾
    pbar.close()
    logger.close()
    print(f"✅ 训练完成: {env_name}-{algo_name}-seed{seed} | 日志路径: {log_dir}")

# ==========================
# 2. 复用GAE计算函数（来自all_algos.py）
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

# ==========================
# 3. 批量运行配置
# ==========================
def main():
    parser = argparse.ArgumentParser(description="批量运行GRPO系列算法")
    parser.add_argument("--config_dir", type=str, default="configs", help="配置文件目录")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="实验种子列表")
    parser.add_argument("--num_workers", type=int, default=1, help="并行进程数")
    args = parser.parse_args()

    # 定义要运行的环境和算法列表
    EXPERIMENTS = [
        # (环境名, 配置文件名, 算法名)
        ("Walker2d-v4", "walk2d.yaml", "standard_grpo"),
        ("Walker2d-v4", "walk2d.yaml", "soft_pair_grpo"),
        ("Walker2d-v4", "walk2d.yaml", "hard_pair_grpo"),
        ("Walker2d-v4", "walk2d.yaml", "dpo"),
        ("Ant-v4", "ant.yaml", "standard_grpo"),
        ("Ant-v4", "ant.yaml", "soft_pair_grpo"),
        ("Ant-v4", "ant.yaml", "hard_pair_grpo"),
        ("Hopper-v4", "hopper.yaml", "standard_grpo"),
        ("Hopper-v4", "hopper.yaml", "soft_pair_grpo"),
        ("Hopper-v4", "hopper.yaml", "hard_pair_grpo"),
    ]

    # 生成所有任务
    tasks = []
    for env_name, cfg_file, algo_name in EXPERIMENTS:
        cfg_path = os.path.join(args.config_dir, cfg_file)
        if not os.path.exists(cfg_path):
            print(f"⚠️ 配置文件不存在: {cfg_path}，跳过该实验")
            continue
        for seed in args.seeds:
            tasks.append((cfg_path, algo_name, env_name, seed))

    # 执行任务（串行/并行）
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            # 多进程执行
            pool.starmap(train_algo, tasks)
    else:
        # 串行执行（方便调试）
        for task in tasks:
            train_algo(*task)

    print("\n🎉 所有实验运行完成！")

if __name__ == "__main__":
    main()
