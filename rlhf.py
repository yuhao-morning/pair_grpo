import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ----------------------
# 1. 环境（可换 MuJoCo）
# ----------------------
env = gym.make("HalfCheetah-v4")  # 换 LunarLander-v3 也行
log_dir = "./rlhf_log/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# ----------------------
# 2. 人工奖励函数（RLHF核心）
# ----------------------
def human_reward(obs, action, env_reward):
    """
    模拟人类偏好：
    - 速度快 → 加分
    - 动作平稳 → 加分
    """
    speed = np.sum(obs[7:10])  # HalfCheetah速度维度
    action_penalty = np.sum(np.square(action))  # 动作剧烈惩罚
    human_score = speed * 2.0 - action_penalty * 0.1
    return human_score

# ----------------------
# 3. 自定义RLHF环境
# ----------------------
class RLHFEnv(gym.Wrapper):
    def step(self, action):
        obs, env_reward, done, truncated, info = self.env.step(action)
        # 混合奖励：环境奖励 + 人工奖励
        h_reward = human_reward(obs, action, env_reward)
        total_reward = env_reward + 0.5 * h_reward  # 权重可调
        return obs, total_reward, done, truncated, info

env = RLHFEnv(env)

# ----------------------
# 4. 训练PPO
# ----------------------
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=500000)

# ----------------------
# 5. 自动画奖励曲线
# ----------------------
def plot_curve():
    df = pd.read_csv(os.path.join(log_dir, os.listdir(log_dir)[0]))
    df["smooth"] = df["r"].rolling(20).mean()
    plt.figure(figsize=(10,5))
    plt.plot(df["r"], alpha=0.3, label="Raw")
    plt.plot(df["smooth"], linewidth=2, label="Smooth")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Env + Human)")
    plt.title("RLHF Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("rlhf_reward.png", dpi=300)
    plt.show()

plot_curve()

# ----------------------
# 6. 测试
# ----------------------
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, truncated, _ = env.step(action)
    print(f"Reward: {r:.2f}")
    if done or truncated:
        obs, _ = env.reset()
