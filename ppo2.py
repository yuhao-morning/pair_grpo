import gymnasium as gym
from stable_baselines3 import PPO

# --------------------------
# 修复 1：关闭图形渲染（服务器必备）
# --------------------------
#env = gym.make("CartPole-v1")
env = gym.make("LunarLander-v3")
# --------------------------
# 修复 2：强制用 CPU，避免 GPU 警告
# --------------------------
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",        # 关键修复
    tensorboard_log=None # 关键修复：关闭 tensorboard
)

# --------------------------
# 开始训练！
# --------------------------
model.learn(total_timesteps=300000)

# --------------------------
# 训练完测试
# --------------------------
print("训练完成，开始测试...")
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print(obs, reward, done, truncated, info)
    if done or truncated:
        obs, _ = env.reset()

print("测试完成！")
