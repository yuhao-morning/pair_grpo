"""数值验证论文第 3 节的多步结论（不涉及训练）。

定理（逐步比较的期望更新）：组内 K 条同起点独立轨迹，逐步比较优势
    Â_t^i = mean_{j 在 t 时仍运行} sign(G_t^i − G_t^j)
的期望更新等于
    E[g] = K · Σ_t E[ 1{T>t} · c_t · u_t(G_t) · ∇log π(a_t|s_t) ]，
其中 u_t(x) = P(G'_t < x | T'>t) − P(G'_t > x | T'>t)，c_t = 1 − (1 − P(T>t))^{K−1}。

做法：随机小 MDP（随机转移、随机奖励、提前终止、按时间索引的 softmax 策略），
穷举所有轨迹算公式右边；再抽样大量组，算估计量的平均，二者比较。
"""
import itertools

import numpy as np

import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TLO, THI = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.1, 0.4)
rng = np.random.default_rng(SEED)
H, S, A, GAMMA = 3, 2, 2, 0.9
P = rng.dirichlet(np.ones(S), size=(H, S, A))          # 转移 P[t, s, a, s']
R = rng.normal(size=(H, S, A))                          # 奖励均值
NOISE = np.array([-1.0, 1.0])                           # 奖励噪声 ±1，各 1/2（离散，便于穷举）
QTERM = rng.uniform(TLO, THI, size=(H, S, A))           # 每步之后提前终止的概率
Z = rng.normal(size=(H, S, A))                          # 策略 logits
PI = np.exp(Z) / np.exp(Z).sum(-1, keepdims=True)


def enumerate_trajs():
    """返回所有轨迹：(概率, [(t, s, a, r)...])。初始状态固定为 0。"""
    out = []

    def rec(t, s, prob, steps):
        if t == H:
            out.append((prob, steps)); return
        for a, n, term in itertools.product(range(A), range(2), [True, False]):
            pa = PI[t, s, a] * 0.5 * (QTERM[t, s, a] if term else 1 - QTERM[t, s, a])
            st = steps + [(t, s, a, R[t, s, a] + NOISE[n])]
            if term or t == H - 1:
                out.append((prob * pa, st)) if term or t == H - 1 else None
                if not term and t == H - 1:
                    pass
            else:
                for s2 in range(S):
                    rec(t + 1, s2, prob * pa * P[t, s, a, s2], st)

    rec(0, 0, 1.0, [])
    return out


def merge_last_step(trajs):
    """最后一步不终止也会结束：把上面在 t=H−1 处按"终止/不终止"拆开的两条合并成一条。"""
    m = {}
    for p, st in trajs:
        key = tuple(st)
        m[key] = m.get(key, 0.0) + p
    return [(p, list(k)) for k, p in m.items()]


trajs = merge_last_step(enumerate_trajs())
probs = np.array([p for p, _ in trajs])
assert abs(probs.sum() - 1) < 1e-9, probs.sum()


def rtg(steps):
    g, out = 0.0, []
    for _, _, _, r in reversed(steps):
        g = r + GAMMA * g
        out.append(g)
    return out[::-1]


G = [rtg(st) for _, st in trajs]
T = np.array([len(st) for _, st in trajs])


def grad_logpi(t, s, a):
    g = np.zeros((H, S, A))
    g[t, s] = -PI[t, s]
    g[t, s, a] += 1
    return g


# ---- 公式右边（穷举，精确） ----
p_alive = np.array([probs[T > t].sum() for t in range(H)])
c = 1 - (1 - p_alive) ** (K - 1)


def u(t, x):
    m = T > t
    gt = np.array([G[i][t] for i in np.flatnonzero(m)])
    w = probs[m] / probs[m].sum()
    return w[gt < x].sum() - w[gt > x].sum()


formula = np.zeros((H, S, A))
for i, (p, st) in enumerate(trajs):
    for (t, s, a, _), g in zip(st, G[i]):
        formula += p * c[t] * u(t, g) * grad_logpi(t, s, a)

# ---- 估计量的蒙特卡洛平均 ----
N = 400_000
idx = rng.choice(len(trajs), size=(N, K), p=probs)
Gmat = np.full((len(trajs), H), np.nan)
for i in range(len(trajs)):
    Gmat[i, :T[i]] = G[i]
gl = np.zeros((len(trajs), H, H, S, A))       # 每条轨迹每个 t 的 ∇log π
for i, (_, st) in enumerate(trajs):
    for (t, s, a, _) in st:
        gl[i, t] = grad_logpi(t, s, a)
est = np.zeros((H, S, A))
B = 20
est_b = np.zeros((B, H, S, A))
Gs = Gmat[idx]                                 # [N, K, H]
alive = ~np.isnan(Gs)
for t in range(H):
    d = np.sign(Gs[:, :, None, t] - Gs[:, None, :, t])            # [N, K, K]
    valid = alive[:, :, None, t] & alive[:, None, :, t] & ~np.eye(K, dtype=bool)
    cnt = valid.sum(2)
    adv = np.where(cnt > 0, np.nansum(np.where(valid, d, 0), 2) / np.maximum(cnt, 1), 0.0)   # [N, K]
    contrib = np.einsum("nk,nkhsa->nhsa", adv, gl[idx][:, :, t]) / K   # 每组的贡献
    est += contrib.mean(0)
    est_b += contrib.reshape(B, N // B, H, S, A).mean(1)

se = est_b.std(0, ddof=1) / np.sqrt(B)
z = np.abs(est - formula) / (se + 1e-12)
print(f"MDP 种子 {SEED}，K={K}，终止概率 [{TLO},{THI}]，c_t = {np.round(c, 3)}")
print(f"  公式 vs 估计量：最大 |误差|/标准误 = {z.max():.2f}（{(z > 2).mean():.0%} 的分量超过 2 个标准误），公式最大绝对值 {np.abs(formula).max():.4f}")

# 对照：如果去掉 c_t 或把 u_t 换成"不分是否运行"的版本，误差应明显变大
alt = np.zeros((H, S, A))
for i, (p, st) in enumerate(trajs):
    for (t, s, a, _), g in zip(st, G[i]):
        alt += p * u(t, g) * grad_logpi(t, s, a)
print(f"  去掉 c_t 的错误版本：最大 |误差|/标准误 = {(np.abs(est - alt) / (se + 1e-12)).max():.1f}")
