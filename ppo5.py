import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

# ====================== 训练部分 ======================
log_dir = "./ppo_log/"
os.makedirs(log_dir, exist_ok=True)

# 你可以随便换环境：LunarLander-v3 / HalfCheetah-v4 / Hopper-v4
env = gym.make("HalfCheetah-v4")
env = Monitor(env, log_dir)  # 自动记录奖励

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    tensorboard_log=None
)

model.learn(total_timesteps=200000)

# ====================== 自动画奖励曲线（修复版） ======================
def plot_reward_curve(log_folder):
    # 自动找到 monitor 生成的 csv
    csv_files = glob.glob(os.path.join(log_folder, "*.csv"))
    if not csv_files:
        print("没有找到日志文件")
        return
    csv_path = csv_files[0]

    df = pd.read_csv(csv_path, comment="#")  # 关键修复！

    # 自动识别列名（兼容所有版本）
    if "episode_reward" in df.columns:
        reward_col = "episode_reward"
    elif "r" in df.columns:
        reward_col = "r"
    else:
        reward_col = df.columns[0]  # 兜底

    # 平滑
    window = 20
    df["rew_smooth"] = df[reward_col].rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df[reward_col], alpha=0.3, label="Raw Reward")
    plt.plot(df["rew_smooth"], linewidth=2, label=f"Smooth ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reward_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("图表已保存为 reward_curve.png")

plot_reward_curve(log_dir)
