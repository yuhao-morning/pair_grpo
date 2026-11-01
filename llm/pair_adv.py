"""LLM 智能体实验的比较式优势估计（定义见 prereg.md 第 3 节）。

与 verl-agent 无关的纯 numpy/torch 实现：输入是每个步样本一个数（回合回报 R、折扣回报 G_t 等）和分组键，
输出每个步样本一个优势；接到 verl-agent 时再按回复 token 广播（与 core_gigpo.py 相同）。
原则：比较臂与对应的标量方法用同一个量、同一个分组，只把"组内减均值/除标准差"换成"组内平均符号"。
"""
from collections import defaultdict

import numpy as np
import torch
from torch import nn


def to_hashable(x):
    """观测转成可哈希的键（同 verl-agent gigpo.core_gigpo.to_hashable）。"""
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return tuple(x.flatten())
    if isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    raise TypeError(f"Unsupported type: {type(x)}")


def group_indices(*keys):
    """按若干个等长键数组的组合分组，返回 {键: 行号数组}。"""
    groups = defaultdict(list)
    for i, k in enumerate(zip(*keys)):
        groups[tuple(to_hashable(e) for e in k)].append(i)
    return {k: np.asarray(v) for k, v in groups.items()}


def compare_norm(x, groups):
    """比较归一化：A_i = mean_{j 同组, j≠i} sign(x_i − x_j)；单元素组为 0。groups 来自 group_indices。"""
    x = np.asarray(x, dtype=np.float64)
    A = np.zeros(len(x))
    for idx in groups.values():
        if len(idx) < 2:
            continue
        v = x[idx]
        A[idx] = np.sign(v[:, None] - v[None, :]).sum(1) / (len(idx) - 1)
    return A


def batch_scale(A, eps=1e-8):
    """整批只缩放、不平移（同 phase_b.py）；全零时原样返回。"""
    s = A.std()
    return A / (s + eps) if s > 0 else A


def turn_index(traj_uid):
    """每个步样本在自己轨迹里的回合序号。要求同一轨迹的样本在数组里按时间顺序出现
    （verl-agent 在 compute_step_discounted_returns 里也这样假设），必须在 _balance_batch 打乱之前调用。"""
    seen = defaultdict(int)
    t = np.zeros(len(traj_uid), dtype=np.int64)
    for i, u in enumerate(traj_uid):
        t[i] = seen[u]
        seen[u] += 1
    return t


def pair_grpo(R, uid):
    """GRPO 的比较版：同一指令组内所有步样本（verl-agent 的"跨步"计算）上比较 R。"""
    return compare_norm(R, group_indices(uid))


def pair_gigpo(R, G, uid, anchor_obs, w=1.0):
    """GiGPO 的比较版：回合项在组内比较 R，步项在锚组（组内观测完全相同）里比较 G_t。"""
    return compare_norm(R, group_indices(uid)) + w * compare_norm(G, group_indices(uid, anchor_obs))


def pair_step(G, uid, t):
    """论文的逐步比较：同一组、同一回合序号（即 t 时刻仍在运行）的样本之间比较 G_t，没有回合项。"""
    return compare_norm(G, group_indices(uid, t))


def sig2(x):
    """2σ(x) − 1：critic 预测的比较期望。"""
    return np.tanh(np.asarray(x, dtype=np.float64) / 2)


def pcrit_residual(G, uid, t, f):
    """比较 critic 的残差优势：A_i = mean_j [c_ij − (2σ(f_i − f_j) − 1)]，j 取同一 (组, t) 的其他样本。
    f 是训练前的 critic 在每个样本状态上的值。另外返回配对 (ri, rj, c) 供训练 critic。"""
    G, f = np.asarray(G, dtype=np.float64), np.asarray(f, dtype=np.float64)
    A = np.zeros(len(G))
    RI, RJ, C = [], [], []
    for idx in group_indices(uid, t).values():
        n = len(idx)
        if n < 2:
            continue
        a, b = np.triu_indices(n, 1)
        ri, rj = idx[a], idx[b]
        c = np.sign(G[ri] - G[rj])
        resid = c - sig2(f[ri] - f[rj])
        np.add.at(A, ri, resid); np.add.at(A, rj, -resid)
        A[idx] /= n - 1
        RI.append(ri); RJ.append(rj); C.append(c)
    if not RI:
        return A, (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    return A, (np.concatenate(RI), np.concatenate(RJ), np.concatenate(C))


def win_rates(rank):
    """裁判的组内名次（0 最好，并列同名次）→ 胜率 w_i = (赢的个数 + ½ 平的个数) / (n − 1)。"""
    r = np.asarray(rank, dtype=np.float64)
    if len(r) < 2:
        return np.full(len(r), 0.5)
    s = np.sign(r[None, :] - r[:, None])   # i 名次更小（更好）时为 +1
    return ((s.sum(1) / (len(r) - 1)) + 1) / 2


class RunningNorm:
    """特征的滑动均值和方差（同 phase_b.py 的做法）。"""

    def __init__(self, dim):
        self.n, self.mean, self.m2 = 1e-4, np.zeros(dim), np.ones(dim)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        b = len(x)
        if b == 0:
            return
        bm, bv = x.mean(0), x.var(0)
        tot = self.n + b
        d = bm - self.mean
        self.mean = self.mean + d * b / tot
        self.m2 = self.m2 + bv * b + d ** 2 * self.n * b / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.m2 / self.n + 1e-8), -10, 10)


class PairCritic(nn.Module):
    """V(s)：2 层 128 单元 tanh MLP，作用在冻结特征上（同 phase_b.py 的 Critic）。"""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_critic(critic, opt, X, pairs, rng, epochs=4, max_pairs=40000, batch=1024, clip=0.5):
    """BT 损失训练 critic；平局目标 0.5。X: [N, dim] 已标准化的特征；pairs = (ri, rj, c)。返回最后一个 epoch 的平均损失。"""
    ri, rj, c = pairs
    if len(c) == 0:
        return float("nan")
    sel = rng.choice(len(c), min(len(c), max_pairs), replace=False)
    X = torch.as_tensor(X, dtype=torch.float32)
    RI, RJ = torch.as_tensor(ri[sel]), torch.as_tensor(rj[sel])
    Y = torch.as_tensor((c[sel] + 1) / 2, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(sel))
        losses = []
        for s in range(0, len(sel), batch):
            mb = perm[s:s + batch]
            logit = critic(X[RI[mb]]) - critic(X[RJ[mb]])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, Y[mb])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), clip)
            opt.step()
            losses.append(loss.item())
    return float(np.mean(losses))
