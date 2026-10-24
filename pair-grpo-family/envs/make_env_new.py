import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation, NormalizeObservation, NormalizeReward

gym.register_envs(gymnasium_robotics)

class HERRobotWrapper(gym.Wrapper):
    """HER专用包装器：在观测展平前提取目标"""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # 展平前从原始字典观测取目标
        ag = obs["achieved_goal"]
        dg = obs["desired_goal"]
        return obs, info, ag, dg

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 展平前从原始字典观测取目标
        ag = obs["achieved_goal"]
        dg = obs["desired_goal"]
        return obs, reward, terminated, truncated, info, ag, dg

def make_env(env_name, seed):
    env = gym.make(env_name)
    # 1. 先套HER包装（必须在Flatten之前！）
    env = HERRobotWrapper(env)
    # 2. 再展平观测、归一化
    env = FlattenObservation(env)
    env = NormalizeObservation(env)
    env = NormalizeReward(env, gamma=0.98)
    # 设置随机种子
    env.reset(seed=seed)
    return env

# 统一入口
def make_env_robotics(env_name, seed):
    return make_env(env_name, seed)
