import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import os

# 1. 建日志文件夹
log_dir = "./ppo_log_monitor/"
os.makedirs(log_dir, exist_ok=True)

# 2. 选环境：可换成 Hopper-v4 HalfCheetah-v4 Ant-v4 LunarLander-v3
env = gym.make("LunarLander-v3")
# env = gym.make("HalfCheetah-v4")

# Monitor 会自动记录每episode奖励、长度
env = Monitor(env, log_dir)

# 3. PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    tensorboard_log=None
)

# 4. 训练
model.learn(total_timesteps=500000)

# ====================== 自动画奖励曲线 ======================
def plot_reward_curve(log_folder):
    # 读取 Monitor 生成的 csv
    csv_path = os.path.join(log_folder, os.listdir(log_folder)[0])
    df = pd.read_csv(csv_path)
    
    # 平滑窗口
    window = 20
    df["rew_smooth"] = df["r"].rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10,5))
    plt.plot(df["l"], df["r"], alpha=0.3, label="Raw Reward")
    plt.plot(df["l"], df["rew_smooth"], linewidth=2, label=f"Smooth({window}) Reward")
    plt.xlabel("Episode Length")
    plt.ylabel("Episode Reward")
    plt.title("RL Training Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reward_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_reward_curve(log_dir)
