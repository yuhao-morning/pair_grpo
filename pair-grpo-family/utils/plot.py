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
