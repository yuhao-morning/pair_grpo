import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===================== 配置区（可直接修改）=====================
ENV_NAME = "HalfCheetah-v4"#"LunarLander-v3"#"Pendulum-v1"#"HalfCheetah-v4"#"LunarLander-v3"#"Pendulum-v1"#"LunarLander-v3"#"CartPole-v1"#"HalfCheetah-v4"#"Hopper-v4"#"CartPole-v1"#"Hopper-v4"#"LunarLander-v3"#"CartPole-v1"#"HalfCheetah-v4"#"FetchReach-v4"#"CartPole-v1"#"LunarLander-v3"#"Pendulum-v1"#"Walker2d-v4" #"HalfCheetah-v4" #"Hopper-v4" #"LunarLander-v3"
LOG_DIR = "logs"
WINDOW_SIZE = 50  # 滑动平均窗口，LunarLander用30-50最合适
FIGSIZE = (10, 6)
DPI = 300
# 算法配置：和你日志文件夹名一一对应
ALGOS = [
    #"dpo",
    #"ppo",
    "standard_grpo",
    "soft_pair_grpo",
    "true_soft_pair_grpo",
    "perf_grpo",
    #"hard_pair_grpo"
]
LABELS = [
    #"DPO",
    #"PPO",
    "Standard GRPO",
    "Soft Pair-GRPO",
    "True Soft Pair-GRPO",
    "Perf GRPO",
    #"Hard Pair-GRPO"
]
COLORS = [
    #"#1f77b4",   # 蓝色 - DPO
    #"#1f77b4",    # 蓝色 - PPO
    "#ff7f0e",   # 橙色 - Standard GRPO
    "#2ca02c",   # 绿色 - Soft Pair-GRPO
    "#1f77b4",  # 蓝色 - True Soft Pair-GRPO
    "#d62728",    # 红色 - Perf-GRPO
    #"#d62728"    # 红色 - Hard Pair-GRPO
]
# =============================================================

def moving_average(data, window):
    """滑动平均，处理边界数据"""
    if len(data) < window:
        return data
    weights = np.repeat(1.0, window) / window
    return np.convolve(data, weights, mode="valid")

plt.figure(figsize=FIGSIZE, dpi=DPI)

for i, (algo, label, color) in enumerate(zip(ALGOS, LABELS, COLORS)):
    csv_path = os.path.join(LOG_DIR, ENV_NAME, algo, "reward.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️  警告：未找到 {csv_path}，跳过该算法")
        continue
    print('csv path ', csv_path)
    df = pd.read_csv(csv_path)
    steps = df["step"].values
    rewards = df["reward"].values
    
    # 滑动平均
    smooth_rewards = moving_average(rewards, WINDOW_SIZE)
    smooth_steps = moving_average(steps, WINDOW_SIZE)
    
    # 绘制平滑曲线（主线条）
    plt.plot(smooth_steps, smooth_rewards, label=label, color=color, linewidth=2.5, alpha=0.9)
    # 可选：绘制原始曲线（淡色背景，展示波动）
    plt.plot(steps, rewards, color=color, linewidth=0.8, alpha=0.2)

# 美化配置
plt.title(f"Training Reward on {ENV_NAME}", fontsize=16, pad=15)
plt.xlabel("Timesteps", fontsize=14)
plt.ylabel("Episode Reward", fontsize=14)
plt.grid(True, alpha=0.3, linestyle="--")
plt.legend(fontsize=12, loc="best")
plt.tight_layout()

# 保存图片
save_path = f"{ENV_NAME}_training_curve_paper.png"
plt.savefig(save_path, bbox_inches="tight", dpi=DPI)
plt.close()

print(f"✅ 论文级对比图已保存到：{save_path}")
print("你可以直接把这张图插入论文，格式、分辨率、配色都符合学术规范")
