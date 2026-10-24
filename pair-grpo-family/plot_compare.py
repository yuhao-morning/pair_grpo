import pandas as pd
import matplotlib.pyplot as plt
import os

env = "Ant-v4"#"LunarLander-v3"
algos = ["dpo", "standard_grpo", "soft_pair_grpo", "hard_pair_grpo"]
labels = ["DPO", "Standard GRPO", "Soft Pair-GRPO", "Hard Pair-GRPO"]
colors = ["blue", "orange", "green", "red"]

plt.figure(figsize=(10, 5))

for algo, label, c in zip(algos, labels, colors):
    path = f"logs/{env}/{algo}/reward.csv"
    df = pd.read_csv(path)
    plt.plot(df["step"], df["reward"], label=label, color=c, alpha=0.8)

plt.title(f"{env} Algorithm Comparison", fontsize=14)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{env}_compare.png", dpi=300)
plt.show()
