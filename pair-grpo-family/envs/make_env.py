import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
def make_env(env_name, seed=0):
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env

def make_env_robotics(env_name, seed):
    env = gym.make(env_name)
    # 展平字典观测为一维数组
    env = FlattenObservation(env)
    env.reset(seed=seed)
    return env
