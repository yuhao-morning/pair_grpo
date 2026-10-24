import numpy as np
import random
from collections import deque

class HERReplayBuffer:
    def __init__(self, capacity: int, her_ratio: float = 0.8):
        self.capacity = capacity
        self.her_ratio = her_ratio
        self.buffer = deque(maxlen=capacity)

    def add_trajectory(self, traj):
        """存入一条完整轨迹"""
        self.buffer.append(traj)

    def compute_sparse_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """FetchReach 原生稀疏奖励：距离>0.05 则 -1，否则 0"""
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist > 0.05).astype(np.float32)

    def sample(self, batch_size: int):
        """采样 batch，混合原始样本 + HER 重标记样本"""
        batch_obs = []
        batch_acts = []
        batch_rewards = []
        batch_dones = []
        batch_ag = []
        batch_dg = []

        for _ in range(batch_size):
            # 随机选一条轨迹
            traj = random.choice(self.buffer)
            T = len(traj)
            t = random.randint(0, T - 1)
            obs_t, act_t, rew_t, done_t, ag_t, dg_t = traj[t]

            # HER 重标记：用未来到达的位置作为新目标
            if random.random() < self.her_ratio:
                future_t = random.randint(t, T - 1)
                new_dg = traj[future_t][4]
                new_rew = self.compute_sparse_reward(ag_t, new_dg)
                batch_obs.append(obs_t)
                batch_acts.append(act_t)
                batch_rewards.append(new_rew)
                batch_dones.append(done_t)
                batch_ag.append(ag_t)
                batch_dg.append(new_dg)
            else:
                # 使用原始目标与奖励
                batch_obs.append(obs_t)
                batch_acts.append(act_t)
                batch_rewards.append(rew_t)
                batch_dones.append(done_t)
                batch_ag.append(ag_t)
                batch_dg.append(dg_t)

        return (
            np.array(batch_obs),
            np.array(batch_acts),
            np.array(batch_rewards),
            np.array(batch_dones),
            np.array(batch_dg)
        )

    def __len__(self):
        return len(self.buffer)
