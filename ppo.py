import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 1. 创建环境（CartPole 最经典）
env = gym.make("CartPole-v1", render_mode="human")

# 2. 创建算法
model = PPO(
    "MlpPolicy",    # 神经网络
    env,            # 环境
    verbose=1,      # 打印训练信息
    tensorboard_log="./ppo_log/"
)

# 3. 训练！
model.learn(total_timesteps=100000)

# 4. 测试效果
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()
