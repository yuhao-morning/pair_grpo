"""比较预算实验的分析，判据见 budget_prereg.md。"""
import glob

import numpy as np
import pandas as pd
from scipy import stats

ENVS = {"LunarLander-v3": (2e6, "results/disc"), "HalfCheetah-v4": (3e6, "results/pcrit")}
BUDGETS = ["r1", "L10", "r1L10"]
METHODS = ["pair_step", "pcrit_res"]


def runs(root, env, name, total):
    out = []
    for f in sorted(glob.glob(f"{root}/{env}/{name}/seed*.csv")):
        if not f.split("seed")[-1][:-4].isdigit() or not 1 <= int(f.split("seed")[-1][:-4]) <= 10:
            continue
        d = pd.read_csv(f)
        if d.step.max() < 0.99 * total:
            continue
        bins = np.linspace(0, total, 51)
        auc = np.mean([d.reward[(d.step > lo) & (d.step <= hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])
                       if ((d.step > lo) & (d.step <= hi)).any()])
        out.append((d.reward[d.step >= 0.9 * total].mean(), auc, d.reward.iloc[0]))
    return np.array(out)


def holm(p):
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def frac(env, name):
    fs = glob.glob(f"results/budget/{env}/{name}/budget_seed*.txt")
    return np.mean([pd.read_csv(f).fraction.iloc[-1] for f in fs]) if fs else 1.0


ci = lambda x: stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
main, sec = [], []
for env, (total, full_root) in ENVS.items():
    full = {m: runs(full_root, env, m, total) for m in METHODS}
    R0 = np.concatenate([full[m][:, 2] for m in METHODS]).mean()
    print(f"== {env}   R0（full 首条记录平均）= {R0:.1f}")
    data = {}
    for m in METHODS:
        data[(m, "full")] = full[m]
        for b in BUDGETS:
            data[(m, b)] = runs("results/budget", env, f"{m}_{b}", total)
    for b in ["full"] + BUDGETS:
        for m in METHODS:
            x = data[(m, b)]
            ret = (x[:, 0].mean() - R0) / (data[(m, "full")][:, 0].mean() - R0)
            f_ = 1.0 if b == "full" else frac(env, f"{m}_{b}")
            print(f"  {b:6s} {m:10s} n={len(x):2d} 预算 {f_:6.1%}  final {x[:, 0].mean():8.1f} ± {ci(x[:, 0]):6.1f}  "
                  f"auc {x[:, 1].mean():8.1f}  保留率 {ret:5.2f}")
            if b != "full":
                main.append(dict(env=env, method=m, budget=b, retention=ret, diff=x[:, 0].mean() - data[(m, "full")][:, 0].mean(),
                                 p=stats.ttest_ind(x[:, 0], data[(m, "full")][:, 0], equal_var=False).pvalue))
        if b != "full":
            r, p_ = data[("pcrit_res", b)], data[("pair_step", b)]
            sec.append(dict(env=env, budget=b, diff=r[:, 0].mean() - p_[:, 0].mean(),
                            p=stats.ttest_ind(r[:, 0], p_[:, 0], equal_var=False).pvalue,
                            p_mw=stats.mannwhitneyu(r[:, 0], p_[:, 0]).pvalue))

M = pd.DataFrame(main); M["p_holm"] = holm(M.p.values)
M["预算稳健"] = (M.retention >= 0.9) & ~((M.p_holm < 0.05) & (M["diff"] < 0))
S = pd.DataFrame(sec); S["p_holm"] = holm(S.p.values)
S["结论"] = np.where(S.p_holm < 0.05, np.where(S["diff"] > 0, "pcrit_res 更好", "pcrit_res 更差"), "")
with pd.option_context("display.width", 200):
    print("\n主检验：预算 vs full（final），Holm 校正（12 个）")
    print(M.round(4).to_string(index=False))
    print("\n次检验：pcrit_res vs pair_step（final），Holm 校正（6 个）")
    print(S.round(4).to_string(index=False))
better, worse = (S["结论"] == "pcrit_res 更好").sum(), (S["结论"] == "pcrit_res 更差").sum()
print(f"\n'pcrit_res 更省比较'：{better} 个条件显著更好，{worse} 个显著更差 → {'成立' if better >= 1 and worse == 0 else '不成立'}")
