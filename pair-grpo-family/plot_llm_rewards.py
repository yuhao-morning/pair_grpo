import json
import numpy as np
import matplotlib.pyplot as plt

# 风格完全和你 RL 环境一样
plt.style.use("default")
plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "figure.figsize": (10, 5),
})

def load_reward(name):
    with open(f"{name}_reward.json", "r") as f:
        data = json.load(f)
    steps = [x["step"] for x in data]
    margins = [x["reward_margin"] for x in data]
    return steps, margins

# 加载四个算法（如果有DPO也可以加）
d = {}
d["StandardGRPO"] = load_reward("StandardGRPO")
d["SoftPairGRPO"] = load_reward("SoftPairGRPO")
#d["HardPairGRPO"] = load_reward("HardPairGRPO")

# 绘图
colors = ["orange", "red"]#, "green"]
labels = ["Standard GRPO", "Soft PairGRPO"]#, "Hard PairGRPO"]

print(d['SoftPairGRPO'])

for (name, color, label) in zip(d.keys(), colors, labels):
    print('name ', name)
    steps, margin = d[name]
    print('margin ', margin)
    plt.plot(steps, margin, 'o-', linewidth=2, label=label, color=color)

plt.xlabel("Training Steps")
plt.ylabel("Reward Margin (chosen - rejected)")
plt.title("LLM Reward Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("llm_reward_curve.png", dpi=300)
plt.show()
