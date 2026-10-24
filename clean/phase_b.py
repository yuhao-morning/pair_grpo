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
pair_* 只用比较结果；比较可以加噪声：
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

ALGOS = ["ppo", "grpo_traj", "pair_traj", "grpo_step", "pair_step", "soft_step", "softz_step", "tie_step"]


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
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "phaseB"))
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    envs = [gym.make(a.env) for _ in range(a.K)]
    for i, e in enumerate(envs):
        e.action_space.seed(a.seed * 100 + i)
    sp = envs[0].action_space
    continuous = isinstance(sp, gym.spaces.Box)
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = sp.shape[0] if continuous else sp.n

    actor = Actor(obs_dim, act_dim, continuous)
    critic = Critic(obs_dim)
    a_opt = optim.Adam(actor.parameters(), lr=a.lr)
    c_opt = optim.Adam(critic.parameters(), lr=a.lr)
    onorm = RunningNorm(obs_dim)
    ret_scale = None   # 噪声尺度 s：第一批回报的 std，之后固定

    name = a.algo + (a.tag and "_" + a.tag)
    out_dir = os.path.join(a.out, a.env, name)
    os.makedirs(out_dir, exist_ok=True)
    f = open(os.path.join(out_dir, f"seed{a.seed}.csv"), "w", newline="")
    w = csv.writer(f); w.writerow(["step", "reward"])

    step = 0
    while step < a.steps:
        O, A, LP, ADV, RET = [], [], [], [], []
        n_batch, ep_returns = 0, []
        while n_batch < a.batch:
            # ---- 一组：K 条轨迹，同一初始状态 ----
            gseed = int(rng.integers(2 ** 31))
            obs = np.stack([e.reset(seed=gseed)[0] for e in envs])
            alive = np.ones(a.K, bool)
            traj = [dict(o=[], a=[], lp=[], r=[], last=None, trunc=False) for _ in range(a.K)]
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
                    nobs, r, term, trunc, _ = envs[i].step(act_np[k])
                    tr = traj[i]
                    tr["o"].append(obs[i].copy()); tr["a"].append(act_np[k]); tr["lp"].append(lp[k].item()); tr["r"].append(r)
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
            ep_returns += totals.tolist()
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
            elif a.algo == "pair_step":
                adv = pair_matrix(U, mask, a.flip, rng)
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
            else:
                adv = np.zeros((a.K, T))   # ppo 之后用 GAE 填

            for i, t in enumerate(traj):
                O += t["o"]; A += t["a"]; LP += t["lp"]
                ADV.append(adv[i, :lens[i]])
                if a.algo == "ppo":
                    RET.append((np.array(t["r"]), t["last"], t["trunc"]))
            n_batch += lens.sum()

        step += n_batch
        mean_ret = float(np.mean(ep_returns))
        w.writerow([step, mean_ret]); f.flush()

        O = np.array(O)
        onorm.update(O)          # 用旧统计量采样，更新后统一用新统计量训练（各方法相同）
        o = torch.as_tensor(onorm(O), dtype=torch.float32)
        act_t = torch.as_tensor(np.array(A), dtype=torch.float32 if continuous else torch.long)
        old_lp = torch.as_tensor(np.array(LP), dtype=torch.float32)
        with torch.no_grad():
            old_lp = actor.log_prob(o, act_t)   # 观测归一化变了，重新算 old logp，保证 ratio 从 1 开始

        if a.algo == "ppo":
            with torch.no_grad():
                v_all = critic(o).numpy()
            advs, rets, pos = [], [], 0
            for r, last, trunc in RET:
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
            adv = np.concatenate(ADV)
            adv = adv / (adv.std() + 1e-8)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)

        N = len(o)
        for _ in range(a.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, a.minibatch):
                mb = perm[s:s + a.minibatch]
                ratio = torch.exp(actor.log_prob(o[mb], act_t[mb]) - old_lp[mb])
                loss = -torch.min(ratio * adv_t[mb], torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv_t[mb]).mean()
                a_opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                a_opt.step()
                if a.algo == "ppo":
                    v_loss = nn.functional.mse_loss(critic(o[mb]), ret_t[mb])
                    c_opt.zero_grad(); v_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    c_opt.step()
    f.close()


if __name__ == "__main__":
    main()
