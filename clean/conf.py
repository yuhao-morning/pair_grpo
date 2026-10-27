"""5 种子结论的 10 种子确认，判据见 conf_prereg.md。"""
import numpy as np
import pandas as pd
from scipy import stats


def run_metrics(path, total):
    d = pd.read_csv(path)
    if d.step.max() < 0.99 * total:
        return None
    bins = np.linspace(0, total, 51)
    auc = np.mean([d.reward[(d.step > lo) & (d.step <= hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])
                   if ((d.step > lo) & (d.step <= hi)).any()])
    return d.reward[d.step >= 0.9 * total].mean(), auc, d.reward.iloc[0]


def load(root, env, name, seeds, total):
    out = []
    for s in seeds:
        try:
            m = run_metrics(f"{root}/{env}/{name}/seed{s}.csv", total)
        except FileNotFoundError:
            m = None
        if m is not None:
            out.append(m)
    return np.array(out)


def holm(p):
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


ci = lambda x: stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

# ---------------- C1：pair_step ≈ grpo_step ----------------
print("== C1：同等信息下 pair_step vs grpo_step（final，10 个种子）")
C1 = {"HalfCheetah-v4": ("results/pcrit", 3e6), "Hopper-v4": ("results/pcrit", 3e6),
      "LunarLander-v3": ("results/disc", 2e6), "Acrobot-v1": ("results/disc", 5e5)}
rows = []
for env, (root, total) in C1.items():
    p = load(root, env, "pair_step", range(1, 11), total)
    g = load("results/conf", env, "grpo_step", range(1, 11), total)
    R0 = np.r_[p[:, 2], g[:, 2]].mean()
    m = 0.1 * (g[:, 0].mean() - R0)
    pa = stats.ttest_ind(p[:, 0] + m, g[:, 0], equal_var=False, alternative="greater").pvalue
    pb = stats.ttest_ind(p[:, 0] - m, g[:, 0], equal_var=False, alternative="less").pvalue
    rows.append(dict(env=env, n=f"{len(p)}/{len(g)}", pair=f"{p[:, 0].mean():.1f}±{ci(p[:, 0]):.1f}",
                     grpo=f"{g[:, 0].mean():.1f}±{ci(g[:, 0]):.1f}", diff=p[:, 0].mean() - g[:, 0].mean(),
                     margin=m, p=stats.ttest_ind(p[:, 0], g[:, 0], equal_var=False).pvalue, p_tost=max(pa, pb),
                     p_auc=stats.ttest_ind(p[:, 1], g[:, 1], equal_var=False).pvalue))
t = pd.DataFrame(rows)
t["p_holm"] = holm(t.p.values)
t["结论"] = np.where(t.p_tost < 0.05, "等价", np.where(t.p_holm < 0.05, np.where(t["diff"] > 0, "pair 更好", "grpo 更好"), "不确定"))
with pd.option_context("display.width", 220):
    print(t.round(4).to_string(index=False))
eq, sig = (t["结论"] == "等价").sum(), t["结论"].str.contains("更好").sum()
print(f"判定：{eq} 个环境等价，{sig} 个显著差别 → " + ("可以写'pair 与 GRPO 打平'" if eq >= 3 and sig == 0 else "不能笼统写'打平'，逐环境报告"))

# ---------------- C2：长训练后 pair_step 非劣于 PPO ----------------
print("\n== C2：长训练 pair_step 非劣于 ppo（final，种子 0–9，界值 10%）")
C2 = {"Hopper-v4": 5e6, "LunarLander-v3": 3e6}
rows = []
for env, total in C2.items():
    p = load("results/long", env, "pair_step", range(10), total)
    o = load("results/long", env, "ppo", range(10), total)
    R0 = np.r_[p[:, 2], o[:, 2]].mean()
    m = 0.1 * (o[:, 0].mean() - R0)
    rows.append(dict(env=env, n=f"{len(p)}/{len(o)}", pair=f"{p[:, 0].mean():.1f}±{ci(p[:, 0]):.1f}",
                     ppo=f"{o[:, 0].mean():.1f}±{ci(o[:, 0]):.1f}", margin=m,
                     p_noninf=stats.ttest_ind(p[:, 0] + m, o[:, 0], equal_var=False, alternative="greater").pvalue,
                     p_two_sided=stats.ttest_ind(p[:, 0], o[:, 0], equal_var=False).pvalue))
t = pd.DataFrame(rows)
t["p_holm"] = holm(t.p_noninf.values)
t["非劣"] = t.p_holm < 0.05
with pd.option_context("display.width", 220):
    print(t.round(4).to_string(index=False))

# ---------------- C3：超参敏感性 ----------------
print("\n== C3：超参敏感性（AUC，每配置 10 个种子 100–109）")
C3 = {"LunarLander-v3": 1e6, "Acrobot-v1": 3e5}
CFG = [(lr, ep) for lr in ["1e-4", "3e-4", "1e-3"] for ep in [4, 10, 20]]
METHODS = ["ppo", "grpo_step", "pair_step"]
rng = np.random.default_rng(0)
for env, total in C3.items():
    data = {m: [load("results/sweep", env, f"{m}_lr{lr}_ep{ep}", range(100, 110), total) for lr, ep in CFG] for m in METHODS}
    R0 = np.concatenate([x[:, 2] for m in METHODS for x in data[m]]).mean()

    def retention(per_cfg):
        means = np.array([x.mean() for x in per_cfg])
        return (means.min() - R0) / (means.max() - R0), (means.max() - means.min()) / (means.max() - R0)

    print(f"  {env}（R0={R0:.1f}）")
    for m in METHODS:
        ns = [len(x) for x in data[m]]
        r, spread = retention([x[:, 1] for x in data[m]])
        means = [x[:, 1].mean() for x in data[m]]
        print(f"    {m:10s} 每配置种子数 {min(ns)}–{max(ns)}  最差配置保留率 {r:.3f}  归一化极差 {spread:.3f}  "
              f"AUC 范围 {min(means):.1f} – {max(means):.1f}")
    boot = {m: [] for m in METHODS}
    for _ in range(2000):
        for m in METHODS:
            boot[m].append(retention([x[rng.integers(len(x), size=len(x)), 1] for x in data[m]])[0])
    for other in ["ppo", "grpo_step"]:
        d = np.array(boot["pair_step"]) - np.array(boot[other])
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict = "pair 更稳" if lo > 0 else ("pair 更不稳" if hi < 0 else "无可测差别")
        print(f"    pair_step − {other} 最差配置保留率：{np.mean(d):+.3f}  [{lo:+.3f}, {hi:+.3f}] → {verdict}")
