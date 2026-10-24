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
