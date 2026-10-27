"""数值验证第 3 节确定性情形的三个结论（用已验证的期望更新公式精确计算，不抽样）。

确定性 MDP（转移、奖励确定，无提前终止），按时间索引的 tabular softmax 策略，固定初始状态。
期望 logit 更新：ż_t(s,a) = c_t · d_t(s) · π_t(a|s) · [Q^u_t(s,a) − Σ_b π_t(b|s) Q^u_t(s,b)]，
其中 Q^u_t(s,a) = E[u_t(G_t) | s_t=s, a_t=a]；无提前终止时 c_t = 1。

(a) 若 t 之后策略是确定的，则在可达状态上、对概率为正的动作，期望优势的顺序与 Q^π 严格一致
    （对手中总有经过同一状态、取同一动作的轨迹，所以两个 Q 值处都有对手质量）。
(b) 最优确定性策略加 ε 扰动：路径上每个状态，都把概率推回最优动作。
(c) 非最优确定性策略：若路径上某个 (t, s) 存在改进动作 a⁺（Q^π_t(s,a⁺) > V^π_t(s)），
    在该处给 a⁺ 加 ε 概率后，a⁺ 的 logit 增加。
"""
import itertools

import numpy as np

H, S, A = 4, 3, 3


def make_mdp(rng):
    nxt = rng.integers(S, size=(H, S, A))
    rew = rng.integers(-3, 4, size=(H, S, A)).astype(float)   # 整数奖励：故意制造并列
    return nxt, rew


def traj_dist(pi, nxt, rew):
    """穷举所有动作序列，返回 [(概率, [(t,s,a,r)...])]。"""
    out = []
    for acts in itertools.product(range(A), repeat=H):
        s, p, st = 0, 1.0, []
        for t, a in enumerate(acts):
            p *= pi[t, s, a]
            if p == 0:
                break
            st.append((t, s, a, rew[t, s, a]))
            s = nxt[t, s, a]
        if p > 0:
            out.append((p, st))
    return out


def expected_update(pi, nxt, rew):
    trajs = traj_dist(pi, nxt, rew)
    probs = np.array([p for p, _ in trajs])
    G = [np.cumsum([r for *_, r in st][::-1])[::-1] for _, st in trajs]   # γ = 1
    zdot = np.zeros((H, S, A))
    for t in range(H):
        gt, w = np.array([g[t] for g in G]), probs
        u = lambda x: w[gt < x].sum() - w[gt > x].sum()
        for (p, st), g in zip(trajs, G):
            _, s, a, _ = st[t]
            grad = -pi[t, s].copy(); grad[a] += 1
            zdot[t, s] += p * u(g[t]) * grad
    return zdot, trajs, G


def q_values(pi, nxt, rew):
    """策略 π 下的 Q_t(s,a)（γ=1），后向递推。"""
    Q = np.zeros((H, S, A)); V = np.zeros((H + 1, S))
    for t in range(H - 1, -1, -1):
        for s in range(S):
            for a in range(A):
                Q[t, s, a] = rew[t, s, a] + V[t + 1, nxt[t, s, a]]
            V[t, s] = pi[t, s] @ Q[t, s]
    return Q, V


def optimal(nxt, rew):
    pi = np.zeros((H, S, A)); V = np.zeros((H + 1, S))
    for t in range(H - 1, -1, -1):
        for s in range(S):
            q = rew[t, s] + V[t + 1, nxt[t, s]]
            pi[t, s, np.argmax(q)] = 1; V[t, s] = q.max()
    return pi


def on_path(pi, nxt):
    s, path = 0, []
    for t in range(H):
        path.append((t, s)); s = nxt[t, s, np.argmax(pi[t, s])]
    return path


fails = {"a": 0, "b": 0, "c": 0}
counts = {"a": 0, "a_tie_rule": 0, "b": 0, "c": 0}
for seed in range(300):
    rng = np.random.default_rng(seed)
    nxt, rew = make_mdp(rng)

    # (a) t 之前随机、t 之后确定
    t0 = int(rng.integers(H))
    pi = rng.dirichlet(np.ones(A), size=(H, S))
    for t in range(t0 + 1, H):
        for s in range(S):
            d = np.zeros(A); d[rng.integers(A)] = 1; pi[t, s] = d
    zdot, trajs, G = expected_update(pi, nxt, rew)
    Q, _ = q_values(pi, nxt, rew)
    probs = np.array([p for p, _ in trajs]); gt = np.array([g[t0] for g in G])
    for s in range(S):
        reach = sum(p for (p, st) in trajs if st[t0][1] == s)
        if reach == 0:
            continue
        adv = zdot[t0, s] / (reach * pi[t0, s])          # 去掉 d_t(s)·π(a|s) 后的"优势"
        for a, b in itertools.permutations(range(A), 2):
            if Q[t0, s, a] > Q[t0, s, b]:
                counts["a"] += 1
                if not adv[a] > adv[b] + 1e-12:        # 严格：两动作都有正概率时，端点处总有对手质量
                    fails["a"] += 1
                lo, hi = Q[t0, s, b], Q[t0, s, a]
                mass = probs[(gt >= lo) & (gt <= hi)].sum()
                counts["a_tie_rule"] += 1
                counts.setdefault("a_no_mass", 0)
                counts["a_no_mass"] += mass < 1e-12

    # (b) 最优策略加 ε 扰动
    eps = 1e-3
    pistar = optimal(nxt, rew) * (1 - eps * A) + eps
    zdot, _, _ = expected_update(pistar, nxt, rew)
    Qs, _ = q_values(optimal(nxt, rew), nxt, rew)
    for t, s in on_path(pistar, nxt):
        astar = np.argmax(pistar[t, s])
        for a in range(A):
            if a != astar and Qs[t, s, a] < Qs[t, s, astar] - 1e-9:
                counts["b"] += 1
                if not (zdot[t, s, a] < 0 < zdot[t, s, astar]):
                    fails["b"] += 1

    # (c) 随机确定性策略：若路径上有改进动作，扰动它
    pidet = np.zeros((H, S, A))
    for t in range(H):
        for s in range(S):
            pidet[t, s, rng.integers(A)] = 1
    Qd, Vd = q_values(pidet, nxt, rew)
    for t, s in on_path(pidet, nxt):
        better = [a for a in range(A) if Qd[t, s, a] > Vd[t, s] + 1e-9]
        if better:
            ap = int(rng.choice(better))
            pert = pidet.copy(); pert[t, s] = pidet[t, s] * (1 - eps); pert[t, s, ap] += eps
            zdot, _, _ = expected_update(pert, nxt, rew)
            counts["c"] += 1
            if not zdot[t, s, ap] > 0:
                fails["c"] += 1
            break

print(f"(a) 顺序与 Q^π 一致：检查 {counts['a']} 个有序对；"
      f"其中区间内无对手的情形 {counts.get('a_no_mass', 0)} 个；严格顺序失败 {fails['a']}")
print(f"(b) 最优策略的稳定性：检查 {counts['b']} 个（路径状态，次优动作）；失败 {fails['b']}")
print(f"(c) 有路径上改进动作的非最优策略不稳定：检查 {counts['c']} 个；失败 {fails['c']}")
