# 1. 导入库
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# 2. 创建 MuJoCo 机器人环境（任选一个！）
# 你可以随便换：HalfCheetah-v4, Ant-v4, Hopper-v4, Walker2d-v4, Humanoid-v4
env = gym.make("HalfCheetah-v4")  # 猎豹奔跑（最简单、最快收敛）
env = Monitor(env)

# 3. 创建 PPO 算法（专门给 MuJoCo 调过参）
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    device="cpu",      # 服务器用 cpu 更快
    tensorboard_log=None
)

# 4. 开始训练！（MuJoCo 通常需要 100万步 左右收敛）
print("开始训练 MuJoCo 机器人...")
model.learn(total_timesteps=1000000)  # 100万步

# 5. 训练完测试
print("训练完成，开始测试...")
obs, _ = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print(f"状态: {obs[:4]}... 奖励: {reward:.2f}")
    if done or truncated:
        obs, _ = env.reset()
