"""LunarLander + 一条自然语言要求："降落在着陆台左半边（x < 0）"。

环境回报（reward）是**隐藏的金标准** = 原始奖励 + 结束时的额外惩罚：
    回合以"非坠毁"方式结束（着陆停稳或超时）且最终 x > 0 时，额外 −80。
info["r_hand"] 是原始 LunarLander 奖励，代表"设计者手写的奖励"：它是对称的，不知道左半边这条要求。
用来检验：只用手写奖励的 PPO 会两边都落；用比较（oracle 或 LLM 裁判，裁判看到这条要求）训练的策略应偏向左边。
"""
import gymnasium as gym
import numpy as np


class LunarLeft(gym.Wrapper):
    PENALTY = 80.0

    def __init__(self, path_penalty=0.0):
        super().__init__(gym.make("LunarLander-v3"))
        self.path_penalty = path_penalty   # >0 时为 v2 要求：飞行过程中每有一步 x > 0.1 再扣这么多

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        info = dict(info, r_hand=float(r))
        gold = float(r)
        if self.path_penalty and obs[0] > 0.1:
            gold -= self.path_penalty
        if term or trunc:
            u = self.env.unwrapped
            crashed = term and (getattr(u, "game_over", False) or abs(obs[0]) >= 1.0)
            landed = term and not crashed
            info.update(crashed=bool(crashed), landed=bool(landed), x_end=float(obs[0]))
            if not crashed and obs[0] > 0:
                gold -= self.PENALTY
        return obs, gold, term, trunc, info


gym.register("LunarLeft-v0", entry_point=LunarLeft)
gym.register("LunarLeftPath-v0", entry_point=LunarLeft, kwargs=dict(path_penalty=1.0))
