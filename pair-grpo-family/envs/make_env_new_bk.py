import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation, NormalizeObservation, NormalizeReward

gym.register_envs(gymnasium_robotics)

class HERRobotWrapper(gym.Wrapper):
    """HER专用包装器：提取 achieved_goal / desired_goal"""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info, info["achieved_goal"], info["desired_goal"]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info, info["achieved_goal"], info["desired_goal"]

def make_env(env_name, seed):
    env = gym.make(env_name)
    # 展平字典观测
    env = FlattenObservation(env)
    # 观测/奖励归一化（稀疏任务必备）
    env = NormalizeObservation(env)
    env = NormalizeReward(env, gamma=0.98)
    # HER 包装
    env = HERRobotWrapper(env)
    # 设置随机种子
    env.reset(seed=seed)
    return env

# 统一入口，废弃原区分逻辑，全部走 make_env
def make_env_robotics(env_name, seed):
    return make_env(env_name, seed)
