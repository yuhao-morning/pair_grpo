"""Phase B/C：不用 critic、不用标量奖励的 pair 方法，和同分组的 GRPO / 标准 PPO 对比。

每次更新：若干组，每组 K 条轨迹从同一个初始状态出发（同一个 reset seed）。
优势（广播到轨迹的每一步，或按时间步 t 计算）：
  ppo        : critic + GAE，batch 内 z-score（参照，用标量奖励）
  grpo_traj  : 组内对总回报 z-score
  pair_traj  : 组内 mean_j sign(G_i - G_j)
  grpo_step  : 每个 t，对还活着的轨迹的 reward-to-go G_t 做 z-score
  pair_step  : 每个 t，mean_{j 活着} sign(G_t^i - G_t^j)
  soft_step  : 每个 t，mean_j tanh((G_t^i - G_t^j) / (tau * s))，sign 与线性之间的插值
  softz_step : 同上，但尺度用每个 t 组内的 std σ_t；tau→∞ 趋于 grpo_step
  tie_step   : 三值比较，|ΔG_t| < tau * σ_t 判"一样"(0)，其余 ±1
  gigpo      : 标量；轨迹级 z-score + omega * 完全相同状态组内的 z-score（Feng et al. 2025）
  pcrit_res  : 只用比较；BT 学 V(s)，优势 = mean_j [c_ij − (2σ(V_i−V_j)−1)]
  pcrit_q    : 只用比较；BT 学 Q(s,a)，优势 = Q(s,a) − mean_m Q(s,a' ~ π)
  pcrit_resq : 只用比较；同 pcrit_q 的 Q，优势 = o_i − b_i（真实比较 − Q 推出的状态基线），消融用
  rm_ppo     : 用同样的比较以 BT 学奖励模型 r̂(s,a)，再跑 PPO + critic + GAE（Christiano 式基线）
  spo        : Swamy et al. 2024：与最近 B 条轨迹的队列比较总回报，得分平均分摊到每一步，再跑 PPO
  pcrit_dr   : 双重稳健；采到的动作用 o_i − m_i(a_i)，另采 M 个动作用模型优势 m_i(a') − b_i，权重 1/M
pair_* 只用比较结果；比较预算用 --pairs_per_t / --every 控制（见 sample_pairs）；比较可以加噪声（--noise 选相关噪声 tie/pos/tiepos，见 noisy_labels）：
  --flip eps   每个比较以概率 eps 翻转
  --tnoise sig 比较/打分前给回报加重尾噪声 sig * s * t(1.5)，标量方法看到同样的噪声回报
所有非 ppo 方法最后只除以 batch std（不再中心化），保证步长可比，只有"形状"不同。
"""
import argparse
import csv
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

torch.set_num_threads(1)

ALGOS = ["ppo", "grpo_traj", "pair_traj", "grpo_step", "pair_step", "soft_step", "softz_step", "tie_step", "gigpo", "pcrit_res", "pcrit_q", "pcrit_resq", "pcrit_dr", "rm_ppo", "spo"]
CRITIC_ALGOS = ("ppo", "rm_ppo", "spo")   # 用 critic + GAE 的方法


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, continuous):
        super().__init__()
        self.continuous = continuous
        self.fc = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        if continuous:
            self.mu = nn.Linear(128, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            self.out = nn.Linear(128, act_dim)

    def dist(self, obs):
        x = self.fc(obs)
        if self.continuous:
            return Normal(self.mu(x), self.log_std.exp())
        return Categorical(logits=self.out(x))

    def log_prob(self, obs, act):
        lp = self.dist(obs).log_prob(act)
        return lp.sum(-1) if self.continuous else lp


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class RunningNorm:
    def __init__(self, dim):
        self.n, self.mean, self.m2 = 1e-4, np.zeros(dim), np.ones(dim)

    def update(self, x):
        b_n, b_mean, b_var = len(x), x.mean(0), x.var(0)
        d, tot = b_mean - self.mean, self.n + len(x)
        self.mean = self.mean + d * b_n / tot
        self.m2 = self.m2 + b_var * b_n + d ** 2 * self.n * b_n / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.m2 / self.n + 1e-8), -10, 10)


def discounted_to_go(r, gamma):
    g, out = 0.0, np.zeros(len(r))
    for t in range(len(r) - 1, -1, -1):
        g = r[t] + gamma * g
        out[t] = g
    return out


def pair_matrix(u, alive, flip, rng, soft_tau=None, scale=1.0, tie=None):
    """u: [K, T] 观测到的效用；alive: [K, T] bool。返回 [K, T] 的 mean_{j≠i, alive} cmp(i, j)。
    tie: 差距小于 tie*scale 时判"一样"(0)，仍然只用三值比较。"""
    d = u[:, None, :] - u[None, :, :]                    # [K, K, T], d[i, j] = u_i - u_j
    c = np.tanh(d / (soft_tau * scale + 1e-8)) if soft_tau else np.sign(d)
    if tie:
        c = np.where(np.abs(d) < tie * scale, 0.0, c)
    if flip > 0:                                          # 每个无序对翻转一次，保持反对称
        K, _, T = c.shape
        m = np.triu(rng.random((K, K, T)) < flip, 1)
        m = m | m.transpose(1, 0, 2)
        c = np.where(m, -c, c)
    valid = alive[:, None, :] & alive[None, :, :]
    valid &= ~np.eye(len(u), dtype=bool)[:, :, None]
    cnt = valid.sum(1)
    return np.where(cnt > 0, (c * valid).sum(1) / np.maximum(cnt, 1), 0.0)


def noisy_labels(d, s, kind, rng, beta=0.44, tau_n=0.85):
    """相关噪声下的比较标签。d = U_i − U_j（i 是配对里先呈现的那个），s = 该时间步组内 std。
    tie   : 以概率 1 − σ(|z|/tau_n) 翻转，z = d/s；越接近平局越容易错（平均翻转率 ≈ 25%）
    pos   : 以概率 beta 直接选先呈现的 i（+1），否则如实；beta=0.44 对应 72% 选第一个
    tiepos: 以概率 2(1 − σ(|z|/tau_n)) "拿不准"，拿不准就选 i；翻转率与 tie 相同，但方向总偏向 i
    """
    c = np.sign(d)
    z = np.abs(d) / (s + 1e-8)
    if kind == "tie":
        c = np.where(rng.random(len(c)) < 1 - 1 / (1 + np.exp(-z / tau_n)), -c, c)
    elif kind == "pos":
        c = np.where(rng.random(len(c)) < beta, 1.0, c)
    elif kind == "tiepos":
        c = np.where(rng.random(len(c)) < 2 * (1 - 1 / (1 + np.exp(-z / tau_n))), 1.0, c)
    return c


def sample_pairs(mask, r, L, rng):
    """比较预算：只在 t = 0, L, 2L, ... 比较；r=0 表示这些 t 上比较所有活着的配对，
    r≥1 表示每个 t 做 r 次随机完美匹配（每次约 K/2 对）。返回 (ci, cj, ct)，ci<cj。"""
    K, T = mask.shape
    ts = np.arange(0, T, L)
    if r == 0:
        ci, cj, k = np.nonzero(np.triu(np.ones((K, K), bool), 1)[:, :, None] & mask[:, None, ts] & mask[None, :, ts])
        return ci, cj, ts[k]
    CI, CJ, CT = [], [], []
    for t in ts:
        alive = np.flatnonzero(mask[:, t])
        n = len(alive) // 2 * 2
        for _ in range(r):
            p = rng.permutation(alive)
            x, y = p[0:n:2], p[1:n:2]
            CI.append(np.minimum(x, y)); CJ.append(np.maximum(x, y)); CT.append(np.full(len(x), t))
    return np.concatenate(CI), np.concatenate(CJ), np.concatenate(CT)


def rm_group_returns(rm, x, lens, gamma):
    """奖励模型对一组轨迹的折扣 reward-to-go。x: [n, in_dim]，按轨迹顺序拼接；返回 [n]（float64，可求导）。"""
    r = rm(x).double()
    out, pos = [], 0
    for n in lens:
        w = gamma ** torch.arange(n, dtype=torch.float64)
        seg = r[pos:pos + n] * w
        out.append(torch.flip(torch.cumsum(torch.flip(seg, [0]), 0), [0]) / w)   # G_t = Σ_{k≥t} γ^{k−t} r_k
        pos += n
    return torch.cat(out)


def make_env(name):
    """MiniGrid：只用 7×7×3 的局部视野图像并展平；其他环境直接 gym.make。"""
    if name in ("LunarLeft-v0", "LunarLeftPath-v0"):
        import lunar_left  # noqa: F401  注册环境
        return gym.make(name)
    if name == "Courier-v0":
        import courier_env  # noqa: F401  注册环境
        return gym.make(name)
    if name.startswith("MiniGrid"):
        import minigrid  # noqa: F401  注册环境
        from minigrid.wrappers import ImgObsWrapper
        return gym.wrappers.FlattenObservation(ImgObsWrapper(gym.make(name)))
    return gym.make(name)


def full_pair_count(mask):
    """完整预算下的比较次数：每个 t 上活着的轨迹两两比较。"""
    n = mask.sum(0)
    return int((n * (n - 1) // 2).sum())


def broadcast_window(A, L):
    """每 L 步比较一次时，窗口 [kL, kL+L) 内的动作共用 t=kL 的优势。"""
    return A if L == 1 else A[:, (np.arange(A.shape[1]) // L) * L]


def sd_t(u, alive):
    """每个 t，活着的轨迹之间的 std。"""
    n = alive.sum(0)
    mu = (u * alive).sum(0) / np.maximum(n, 1)
    return np.sqrt((((u - mu) ** 2) * alive).sum(0) / np.maximum(n - 1, 1))


def zscore_masked(u, alive):
    """每个 t，对活着的轨迹做 z-score；少于 2 条活着则为 0。"""
    n = alive.sum(0)
    mean = np.where(n > 0, (u * alive).sum(0) / np.maximum(n, 1), 0)
    var = np.where(n > 1, (((u - mean) ** 2) * alive).sum(0) / np.maximum(n - 1, 1), 1)
    return np.where((n > 1) & alive, (u - mean) / (np.sqrt(var) + 1e-8), 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--algo", required=True, choices=ALGOS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--flip", type=float, default=0.0)
    p.add_argument("--tnoise", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--noise", default="iid", choices=["iid", "tie", "pos", "tiepos"])  # iid 用 --flip
    p.add_argument("--pairs_per_t", type=int, default=0)   # 比较预算：0=全部配对；r=每个 t 做 r 次随机匹配
    p.add_argument("--every", type=int, default=1)         # 比较预算：每 L 步比较一次
    p.add_argument("--beta", type=float, default=0.44)       # pos 噪声：选第一个的概率
    p.add_argument("--tau_n", type=float, default=0.85)      # tie/tiepos 噪声的尺度
    p.add_argument("--omega", type=float, default=1.0)       # gigpo 步级优势的权重
    p.add_argument("--crit_lr", type=float, default=1e-3)    # pcrit_*：比较训练的 critic
    p.add_argument("--crit_epochs", type=int, default=4)
    p.add_argument("--crit_pairs", type=int, default=40000)  # 每次更新最多用多少个比较对
    p.add_argument("--M", type=int, default=8)               # pcrit_q 的基线采样动作数
    p.add_argument("--rm_buffer", type=int, default=5)       # rm_ppo：保留最近几次更新的比较
    p.add_argument("--rm_epochs", type=int, default=2)       # rm_ppo：每次更新在缓冲区上训练几轮
    p.add_argument("--spo_queue", type=int, default=64)      # spo：比较队列长度 B（论文 10–100）
    p.add_argument("--shaping", type=float, default=0.0)   # 势函数整形系数（只加在标量训练奖励上；Courier 用）
    p.add_argument("--train_reward", default="env", choices=["env", "naive", "common", "hand"])  # 训练用哪种奖励；日志始终记环境回报
    p.add_argument("--save_at", type=int, nargs="*", default=[])  # 在这些步数保存策略（采这一批数据时的参数）
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "phaseB"))
    a = p.parse_args()
    full_budget = a.pairs_per_t == 0 and a.every == 1
    assert full_budget or a.algo in ("pair_step", "pcrit_res"), "比较预算只实现了 pair_step / pcrit_res"

    rng = np.random.default_rng(a.seed)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    envs = [make_env(a.env) for _ in range(a.K)]
    for i, e in enumerate(envs):
        e.action_space.seed(a.seed * 100 + i)
    sp = envs[0].action_space
    continuous = isinstance(sp, gym.spaces.Box)
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = sp.shape[0] if continuous else sp.n

    actor = Actor(obs_dim, act_dim, continuous)
    critic = Critic(obs_dim)
    pcrit = None
    qcrit = a.algo in ("pcrit_q", "pcrit_resq", "pcrit_dr")   # critic 输入 (s,a)
    if a.algo.startswith("pcrit"):
        in_dim = obs_dim + (act_dim if qcrit else 0)
        pcrit = Critic(in_dim)
        p_opt = optim.Adam(pcrit.parameters(), lr=a.crit_lr)
    enc = (lambda x: x) if continuous else (lambda x: nn.functional.one_hot(x, act_dim).float())
    if a.algo == "rm_ppo":   # 从比较学的奖励模型 r(s,a)，配一个最近若干次更新的比较缓冲区
        rm = Critic(obs_dim + act_dim)
        rm_opt = optim.Adam(rm.parameters(), lr=a.crit_lr)
        rm_buf = []
    spo_q = []   # spo：最近 B 条轨迹的（可能带噪的）总回报
    a_opt = optim.Adam(actor.parameters(), lr=a.lr)
    c_opt = optim.Adam(critic.parameters(), lr=a.lr)
    onorm = RunningNorm(obs_dim)
    ret_scale = None   # 噪声尺度 s：第一批回报的 std，之后固定

    name = a.algo + (a.tag and "_" + a.tag)
    out_dir = os.path.join(a.out, a.env, name)
    os.makedirs(out_dir, exist_ok=True)
    f = open(os.path.join(out_dir, f"seed{a.seed}.csv"), "w", newline="")
    w = csv.writer(f); w.writerow(["step", "reward"])
    if a.env in ("LunarLeft-v0", "LunarLeftPath-v0"):
        evf = open(os.path.join(out_dir, f"events_seed{a.seed}.txt"), "w")
        evf.write("step,landed,crashed,end_left,end_right\n")
    if a.env == "Courier-v0":
        evf = open(os.path.join(out_dir, f"events_seed{a.seed}.txt"), "w")
        evf.write("step,flowers_per_ep,success_rate,lava_rate\n")
    if a.env.startswith("MiniGrid"):   # 目标导向任务：另记成功率（回报 > 0 即到达目标）
        sf = open(os.path.join(out_dir, f"success_seed{a.seed}.txt"), "w")
        sf.write("step,success_rate\n")
    if pcrit is not None or a.algo == "rm_ppo":   # critic 在新批次上的比较准确率（训练前），不是 seed*.csv，summarize 不会读到
        acc_f = open(os.path.join(out_dir, f"crit_seed{a.seed}.txt"), "w")
        acc_f.write("step,pair_acc_before_update\n")
    if a.noise != "iid":
        nf = open(os.path.join(out_dir, f"noise_seed{a.seed}.txt"), "w")
        nf.write("step,flip_rate_nontie,tie_to_first,first_rate\n")
    if not full_budget:
        bf = open(os.path.join(out_dir, f"budget_seed{a.seed}.txt"), "w")
        bf.write("step,pairs_used,pairs_full,fraction\n")

    step = 0
    while step < a.steps:
        O, A, LP, ADV, RET = [], [], [], [], []
        GROUPS = []   # pcrit_*：每组的比较标签和行号，批次收集完再算优势
        ep_events = []   # Courier：每回合 (踩花坛次数, 是否送达, 是否踩岩浆)
        noise_log = []   # 相关噪声：(真实符号, 给出的标签)，用来记录实际翻转率
        budget_log = []  # 比较预算：(实际比较次数, 完整预算下的次数)
        n_batch, ep_returns = 0, []
        while n_batch < a.batch:
            # ---- 一组：K 条轨迹，同一初始状态 ----
            gseed = int(rng.integers(2 ** 31))
            obs = np.stack([e.reset(seed=gseed)[0] for e in envs])
            alive = np.ones(a.K, bool)
            traj = [dict(o=[], a=[], lp=[], r=[], gold=[], ev=[], last=None, trunc=False) for _ in range(a.K)]
            while alive.any():
                idx = np.flatnonzero(alive)
                on = onorm(obs[idx])
                with torch.no_grad():
                    d = actor.dist(torch.as_tensor(on, dtype=torch.float32))
                    act = d.sample()
                    lp = d.log_prob(act)
                    lp = lp.sum(-1) if continuous else lp
                act_np = act.numpy()
                for k, i in enumerate(idx):
                    nobs, r, term, trunc, info = envs[i].step(act_np[k])
                    g_r, r = r, (r if a.train_reward == "env" else info["r_" + a.train_reward])   # gold 只用于记录
                    if a.shaping:
                        r = r + a.shaping * info["shaping"]
                    tr = traj[i]
                    tr["o"].append(obs[i].copy()); tr["a"].append(act_np[k]); tr["lp"].append(lp[k].item()); tr["r"].append(r); tr["gold"].append(g_r)
                    if a.env in ("Courier-v0", "LunarLeft-v0", "LunarLeftPath-v0"):
                        tr["ev"].append(info)
                    obs[i] = nobs
                    if term or trunc:
                        alive[i] = False
                        tr["last"], tr["trunc"] = nobs, trunc and not term
            # ---- 组内优势 ----
            lens = np.array([len(t["r"]) for t in traj])
            T = lens.max()
            G = np.zeros((a.K, T))
            mask = np.arange(T)[None, :] < lens[:, None]
            for i, t in enumerate(traj):
                G[i, :lens[i]] = discounted_to_go(np.array(t["r"]), a.gamma)
            totals = np.array([sum(t["r"]) for t in traj])
            ep_returns += [sum(t["gold"]) for t in traj]   # 记录的始终是环境回报（Courier 下即金标准）
            if a.env in ("LunarLeft-v0", "LunarLeftPath-v0"):
                ep_events += [(t["ev"][-1]["landed"], t["ev"][-1]["crashed"],
                               (not t["ev"][-1]["crashed"]) and t["ev"][-1]["x_end"] < 0,
                               (not t["ev"][-1]["crashed"]) and t["ev"][-1]["x_end"] > 0) for t in traj]
            if a.env == "Courier-v0":
                ep_events += [(sum(e["flower"] for e in t["ev"]), any(e["deliver"] for e in t["ev"]), any(e["lava"] for e in t["ev"])) for t in traj]
            if ret_scale is None:   # Acrobot 初期全是 -500，std=0，退回到 |mean| 的 10%
                ret_scale = max(float(np.std(totals)), 0.1 * abs(float(np.mean(totals))), 1.0)
            U, Utot = G, totals.copy()
            if a.tnoise > 0:   # 标量方法和 pair 方法看到同样的噪声
                U = G + a.tnoise * ret_scale * rng.standard_t(1.5, G.shape)
                Utot = totals + a.tnoise * ret_scale * rng.standard_t(1.5, a.K)

            if a.algo == "grpo_traj":
                adv = np.repeat(((Utot - Utot.mean()) / (Utot.std() + 1e-8))[:, None], T, 1)
            elif a.algo == "pair_traj":
                adv = np.repeat(pair_matrix(Utot[:, None], np.ones((a.K, 1), bool), a.flip, rng), T, 1)
            elif a.algo == "grpo_step":
                adv = zscore_masked(U, mask)
            elif a.algo == "pair_step" and a.noise == "iid" and full_budget:
                adv = pair_matrix(U, mask, a.flip, rng)
            elif a.algo == "pair_step":
                # 相关噪声或有限比较预算：逐对生成标签（i<j，i 先呈现），再对比过的 j 取平均
                ci, cj, ct = sample_pairs(mask, a.pairs_per_t, a.every, rng)
                d = U[ci, ct] - U[cj, ct]
                if a.noise == "iid":
                    c = np.sign(d)
                    if a.flip > 0:
                        c = np.where(rng.random(len(c)) < a.flip, -c, c)
                else:
                    c = noisy_labels(d, sd_t(U, mask)[ct], a.noise, rng, a.beta, a.tau_n)
                    noise_log.append((np.sign(d), c))
                s_ = np.zeros((a.K, T)); n_ = np.zeros((a.K, T))
                np.add.at(s_, (ci, ct), c); np.add.at(s_, (cj, ct), -c)
                np.add.at(n_, (ci, ct), 1); np.add.at(n_, (cj, ct), 1)
                adv = broadcast_window(s_ / np.maximum(n_, 1), a.every)
                budget_log.append((len(c), full_pair_count(mask)))
            elif a.algo == "soft_step":
                adv = pair_matrix(U, mask, a.flip, rng, soft_tau=a.tau, scale=ret_scale)
            elif a.algo in ("softz_step", "tie_step"):
                # 尺度用每个 t 组内的 std
                # softz：tanh(d/(tau*sd))，tau→0 是 sign，tau→∞ 趋于 grpo_step
                # tie  ：|d| < tau*sd 判"一样"，其余 ±1
                n = mask.sum(0)
                mu = (U * mask).sum(0) / np.maximum(n, 1)
                sd = np.sqrt((((U - mu) ** 2) * mask).sum(0) / np.maximum(n - 1, 1))
                if a.algo == "softz_step":
                    adv = pair_matrix(U, mask, a.flip, rng, soft_tau=a.tau, scale=sd)
                else:
                    adv = pair_matrix(U, mask, a.flip, rng, scale=sd, tie=a.tau)
            elif a.algo == "gigpo":
                # GiGPO（Feng et al. 2025）：轨迹级相对优势 + ω × 锚状态组内的相对优势（完全相同的状态才算一组）
                a_ep = (Utot - Utot.mean()) / (Utot.std() + 1e-8)
                anchors = {}
                for i, tr in enumerate(traj):
                    for t_, o_ in enumerate(tr["o"]):
                        anchors.setdefault(o_.tobytes(), []).append((i, t_))
                a_st = np.zeros((a.K, T))
                for members in anchors.values():
                    if len(members) > 1:
                        ii, tt = map(np.array, zip(*members))
                        g = U[ii, tt]
                        a_st[ii, tt] = (g - g.mean()) / (g.std() + 1e-8)
                adv = a_ep[:, None] + a.omega * a_st
            else:
                adv = np.zeros((a.K, T))   # ppo 之后用 GAE 填，pcrit_* 批次结束后再填

            base = []   # 每条轨迹在本批次里的起始行号
            for i, t in enumerate(traj):
                base.append(len(O))
                O += t["o"]; A += t["a"]; LP += t["lp"]
                ADV.append(adv[i, :lens[i]])
            if pcrit is not None or a.algo == "rm_ppo":
                # 只存比较结果 c ∈ {+1,0,−1}（由 oracle 给出），不存回报
                if full_budget:
                    ci, cj, ct = np.nonzero(np.triu(np.ones((a.K, a.K), bool), 1)[:, :, None] & mask[:, None, :] & mask[None, :, :])
                else:
                    ci, cj, ct = sample_pairs(mask, a.pairs_per_t, a.every, rng)
                    budget_log.append((len(ci), full_pair_count(mask)))
                if a.noise == "iid":
                    c = np.sign(U[ci, ct] - U[cj, ct])
                    if a.flip > 0:
                        c = np.where(rng.random(len(c)) < a.flip, -c, c)
                else:
                    c = noisy_labels(U[ci, ct] - U[cj, ct], sd_t(U, mask)[ct], a.noise, rng, a.beta, a.tau_n)
                    noise_log.append((np.sign(U[ci, ct] - U[cj, ct]), c))
                base = np.array(base)
                GROUPS.append(dict(ci=ci, cj=cj, ct=ct, c=c, base=base, lens=lens, T=T, start=len(ADV) - a.K))
            if a.algo in CRITIC_ALGOS:   # 第 4 项是（可能带噪的）轨迹总回报，spo 只用它来做比较
                RET += [(np.array(t["r"]), t["last"], t["trunc"], Utot[i]) for i, t in enumerate(traj)]
            n_batch += lens.sum()

        save_at = sorted(a.save_at) if step == 0 else save_at
        while save_at and step >= save_at[0]:   # 保存的是采集本批数据的策略，以及当时的观测归一化统计量
            torch.save(dict(actor=actor.state_dict(), norm=(onorm.n, onorm.mean, onorm.m2), step=step),
                       os.path.join(out_dir, f"ckpt_seed{a.seed}_{save_at.pop(0)}.pt"))
        step += n_batch
        mean_ret = float(np.mean(ep_returns))
        w.writerow([step, mean_ret]); f.flush()
        if a.env in ("LunarLeft-v0", "LunarLeftPath-v0"):
            ev_ = np.array(ep_events, float)
            evf.write(",".join([str(step)] + [f"{v:.3f}" for v in ev_.mean(0)]) + "\n"); evf.flush()
        if a.env == "Courier-v0":
            ev_ = np.array(ep_events, float)
            evf.write(f"{step},{ev_[:, 0].mean():.3f},{ev_[:, 1].mean():.3f},{ev_[:, 2].mean():.3f}\n"); evf.flush()
        if a.env.startswith("MiniGrid"):
            sf.write(f"{step},{np.mean(np.array(ep_returns) > 0):.4f}\n"); sf.flush()
        if budget_log:   # 实际用掉的比较次数占完整预算的比例
            used, full = map(sum, zip(*budget_log))
            bf.write(f"{step},{used},{full},{used / full:.4f}\n"); bf.flush()
        if noise_log:   # 实际翻转率（真实非平局中标签错的比例）、平局被判 +1 的比例、全部标签里选第一个的比例
            tr_, lb_ = map(np.concatenate, zip(*noise_log))
            nzt = tr_ != 0
            nf.write(f"{step},{np.mean(lb_[nzt] != tr_[nzt]):.4f},"
                     f"{np.mean(lb_[~nzt] == 1) if (~nzt).any() else float('nan'):.4f},{np.mean(lb_ == 1):.4f}\n")
            nf.flush()

        O = np.array(O)
        onorm.update(O)          # 用旧统计量采样，更新后统一用新统计量训练（各方法相同）
        o = torch.as_tensor(onorm(O), dtype=torch.float32)
        act_t = torch.as_tensor(np.array(A), dtype=torch.float32 if continuous else torch.long)
        old_lp = torch.as_tensor(np.array(LP), dtype=torch.float32)
        with torch.no_grad():
            old_lp = actor.log_prob(o, act_t)   # 观测归一化变了，重新算 old logp，保证 ratio 从 1 开始

        if a.algo == "spo":
            # SPO（Swamy et al. 2024，Algorithm 2）：逐条轨迹与最近 B 条轨迹的队列比较，
            # 得分 = mean_q 𝒫(ξ, ξ_q) ∈ [−1, 1]，再平均分摊到该轨迹的每一步；比完把它推入队列（FIFO）
            RET_new = []
            for r, last, trunc, u in RET:
                score = np.mean(np.sign(u - np.array(spo_q))) if spo_q else 0.0
                RET_new.append((np.full(len(r), score / len(r)), last, False, u))
                spo_q.append(u)
                spo_q = spo_q[-a.spo_queue:]
            RET = RET_new
        if a.algo == "rm_ppo":
            # ---- 奖励模型：只用比较训练，PPO 用它给的奖励 ----
            X_b = torch.cat([o, enc(act_t)], -1)
            new = [dict(obs=O[g["base"][0]:g["base"][0] + g["lens"].sum()], act=act_t[g["base"][0]:g["base"][0] + g["lens"].sum()],
                        lens=g["lens"], ri=g["base"][g["ci"]] - g["base"][0] + g["ct"],
                        rj=g["base"][g["cj"]] - g["base"][0] + g["ct"], c=g["c"]) for g in GROUPS]
            with torch.no_grad():   # 训练前在新批次比较上的准确率
                hit = tot_ = 0
                for b in new:
                    G = rm_group_returns(rm, torch.cat([torch.as_tensor(onorm(b["obs"]), dtype=torch.float32), enc(b["act"])], -1), b["lens"], a.gamma).numpy()
                    nz = b["c"] != 0
                    hit += np.sum(np.sign(G[b["ri"][nz]] - G[b["rj"][nz]]) == b["c"][nz]); tot_ += nz.sum()
                acc_f.write(f"{step},{hit / max(tot_, 1):.4f}\n"); acc_f.flush()
            rm_buf.append(new)
            rm_buf = rm_buf[-a.rm_buffer:]
            groups_all = [b for upd in rm_buf for b in upd]
            for _ in range(a.rm_epochs):
                for k in np.random.permutation(len(groups_all)):
                    b = groups_all[k]
                    G = rm_group_returns(rm, torch.cat([torch.as_tensor(onorm(b["obs"]), dtype=torch.float32), enc(b["act"])], -1), b["lens"], a.gamma)
                    logit = (G[b["ri"]] - G[b["rj"]]).float()
                    y = torch.as_tensor((b["c"] + 1) / 2, dtype=torch.float32)   # 平局 → 0.5
                    loss_rm = nn.functional.binary_cross_entropy_with_logits(logit, y)
                    rm_opt.zero_grad(); loss_rm.backward()
                    nn.utils.clip_grad_norm_(rm.parameters(), 0.5)
                    rm_opt.step()
            with torch.no_grad():
                r_hat = rm(X_b).numpy()
            r_hat = r_hat / (r_hat.std() + 1e-8)   # 只缩放、不平移：平移会改变有终止的任务里的最优策略
            pos_ = 0
            RET_new = []
            for r, last, trunc, u in RET:
                RET_new.append((r_hat[pos_:pos_ + len(r)], last, trunc, u)); pos_ += len(r)
            RET = RET_new
        if a.algo in CRITIC_ALGOS:
            with torch.no_grad():
                v_all = critic(o).numpy()
            advs, rets, pos = [], [], 0
            for r, last, trunc, _ in RET:
                n = len(r)
                v = v_all[pos:pos + n]; pos += n
                with torch.no_grad():
                    nv = critic(torch.as_tensor(onorm(last[None]), dtype=torch.float32)).item() if trunc else 0.0
                gae, ad = 0.0, np.zeros(n)
                for t in range(n - 1, -1, -1):
                    nxt = v[t + 1] if t + 1 < n else nv
                    delta = r[t] + a.gamma * nxt - v[t]
                    gae = delta + a.gamma * a.lam * gae
                    ad[t] = gae
                advs.append(ad); rets.append(ad + v)
            adv = np.concatenate(advs)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret_t = torch.as_tensor(np.concatenate(rets), dtype=torch.float32)
        else:
            if pcrit is not None:
                # ---- 比较训练的 critic：先用旧 critic 算优势，再用本批比较训练 ----
                if qcrit:
                    enc = (lambda x: x) if continuous else (lambda x: nn.functional.one_hot(x, act_dim).float())
                    X = torch.cat([o, enc(act_t)], -1)
                else:
                    X = o
                with torch.no_grad():
                    f_all = pcrit(X).numpy()
                    if qcrit:   # 同一状态下按当前策略采 M 个动作
                        d = actor.dist(o)
                        A_s = torch.stack([d.sample() for _ in range(a.M)])                     # [M, N, ...]
                        Q_s = torch.stack([pcrit(torch.cat([o, enc(x)], -1)) for x in A_s]).numpy()  # [M, N]
                        q_base = Q_s.mean(0)                                                     # V(s) = E_a' Q(s,a')
                sig2 = lambda x: 2.0 / (1.0 + np.exp(-x)) - 1.0   # 2σ(x) − 1：预测的比较期望
                if a.algo == "pcrit_dr":
                    ADV_m = [None] * len(ADV)   # 采样动作那部分的优势 [M, len]
                RI, RJ, C = [], [], []
                for g in GROUPS:
                    ri, rj, c = g["base"][g["ci"]] + g["ct"], g["base"][g["cj"]] + g["ct"], g["c"]
                    RI.append(ri); RJ.append(rj); C.append(c)
                    idx_i, idx_j = (g["ci"], g["ct"]), (g["cj"], g["ct"])
                    cnt = np.zeros((a.K, g["T"]))
                    np.add.at(cnt, idx_i, 1); np.add.at(cnt, idx_j, 1)
                    cnt = np.maximum(cnt, 1)

                    def agg(v_i, v_j):
                        """把配对上的量按行平均：v_i 记到 i 行，v_j 记到 j 行。"""
                        s = np.zeros((a.K, g["T"]))
                        np.add.at(s, idx_i, v_i); np.add.at(s, idx_j, v_j)
                        return s / cnt

                    if a.algo == "pcrit_res":
                        # 观测到的比较 − 仅凭状态预测的比较，对 j 取平均（反对称）
                        resid = c - sig2(f_all[ri] - f_all[rj])
                        A_g = broadcast_window(agg(resid, -resid), a.every)
                    elif a.algo in ("pcrit_resq", "pcrit_dr"):
                        # o_i = mean_j c_ij；m_i(a) = mean_j [2σ(Q(s_i,a) − V(s_j)) − 1]；b_i = mean_m m_i(a'_m)
                        o_g = agg(c, -c)
                        m_s = np.stack([agg(sig2(Q_s[m, ri] - q_base[rj]), sig2(Q_s[m, rj] - q_base[ri])) for m in range(a.M)])
                        b_g = m_s.mean(0)
                        if a.algo == "pcrit_resq":
                            A_g = o_g - b_g
                        else:
                            m_taken = agg(sig2(f_all[ri] - q_base[rj]), sig2(f_all[rj] - q_base[ri]))
                            A_g = o_g - m_taken          # 采到的动作：真实比较 − 模型对它的预测
                            Am_g = m_s - b_g[None]       # 采样动作：模型优势，权重 1/M
                            S_g = o_g - b_g              # 只用来定尺度
                    for i in range(a.K):
                        rows = g["base"][i] + np.arange(g["lens"][i])
                        if a.algo == "pcrit_q":
                            ADV[g["start"] + i] = f_all[rows] - q_base[rows]
                        else:
                            ADV[g["start"] + i] = A_g[i, :g["lens"][i]]
                        if a.algo == "pcrit_dr":
                            ADV_m[g["start"] + i] = (Am_g[:, i, :g["lens"][i]], S_g[i, :g["lens"][i]])
                RI, RJ, C = map(np.concatenate, (RI, RJ, C))
                nz = C != 0
                acc_f.write(f"{step},{np.mean(np.sign(f_all[RI[nz]] - f_all[RJ[nz]]) == C[nz]):.4f}\n"); acc_f.flush()
                sel = rng.choice(len(C), min(len(C), a.crit_pairs), replace=False)
                RI_t, RJ_t = torch.as_tensor(RI[sel]), torch.as_tensor(RJ[sel])
                Y_t = torch.as_tensor((C[sel] + 1) / 2, dtype=torch.float32)   # 平局 → 0.5
                for _ in range(a.crit_epochs):
                    perm = torch.randperm(len(sel))
                    for s in range(0, len(sel), 1024):
                        mb = perm[s:s + 1024]
                        logit = pcrit(X[RI_t[mb]]) - pcrit(X[RJ_t[mb]])
                        p_loss = nn.functional.binary_cross_entropy_with_logits(logit, Y_t[mb])
                        p_opt.zero_grad(); p_loss.backward()
                        nn.utils.clip_grad_norm_(pcrit.parameters(), 0.5)
                        p_opt.step()
            adv = np.concatenate(ADV)
            if a.algo == "pcrit_dr":
                # 两部分用同一个尺度：o_i − b_i 的 std（与 pcrit_resq 的优势同尺度）
                Am = np.concatenate([x[0] for x in ADV_m], axis=1)              # [M, N]
                scale = np.concatenate([x[1] for x in ADV_m]).std() + 1e-8
                adv, Am = adv / scale, Am / scale
            else:
                adv = adv / (adv.std() + 1e-8)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)

        N = len(o)
        o_u, act_u, old_u, adv_u, w_u, mbs = o, act_t, old_lp, adv_t, None, a.minibatch
        if a.algo == "pcrit_dr":
            # 采样动作作为额外的行：权重 1/M；minibatch 放大 (1+M) 倍，保持每个 epoch 的梯度步数不变
            with torch.no_grad():
                old_s = [actor.log_prob(o, x) for x in A_s]
            o_u = torch.cat([o] * (1 + a.M))
            act_u = torch.cat([act_t] + list(A_s))
            old_u = torch.cat([old_lp] + old_s)
            adv_u = torch.cat([adv_t, torch.as_tensor(Am.reshape(-1), dtype=torch.float32)])
            w_u = torch.cat([torch.ones(N), torch.full((a.M * N,), 1.0 / a.M)])
            mbs = a.minibatch * (1 + a.M)
        N_u = len(o_u)
        for _ in range(a.epochs):
            perm = torch.randperm(N_u)
            for s in range(0, N_u, mbs):
                mb = perm[s:s + mbs]
                ratio = torch.exp(actor.log_prob(o_u[mb], act_u[mb]) - old_u[mb])
                surr = torch.min(ratio * adv_u[mb], torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv_u[mb])
                loss = -surr.mean() if w_u is None else -(w_u[mb] * surr).sum() / w_u[mb].sum()
                a_opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                a_opt.step()
                if a.algo in CRITIC_ALGOS:
                    v_loss = nn.functional.mse_loss(critic(o[mb]), ret_t[mb])
                    c_opt.zero_grad(); v_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    c_opt.step()
    f.close()


if __name__ == "__main__":
    main()
