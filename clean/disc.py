"""离散任务 + 噪声标签的分析，判据见 disc_prereg.md。"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

ENVS = {"LunarLander-v3": 2e6, "Acrobot-v1": 5e5}
NOISE = ["", "_f0.1", "_f0.25"]


def metrics(path, total):
    d = pd.read_csv(path)
    if d.step.max() < 0.99 * total:
        return None
    final = d.reward[d.step >= 0.9 * total].mean()
    bins = np.linspace(0, total, 51)
    auc = np.mean([d.reward[(d.step > lo) & (d.step <= hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])
                   if ((d.step > lo) & (d.step <= hi)).any()])
    return final, auc


def load(env, name):
    m = [metrics(f, ENVS[env]) for f in sorted(glob.glob(f"results/disc/{env}/{name}/seed*.csv"))]
    return np.array([x for x in m if x is not None])


ci = lambda x: stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

tests = []
for env in ENVS:
    print(f"== {env}")
    ppo = load(env, "ppo")
    print(f"  ppo                 n={len(ppo):2d} final {ppo[:, 0].mean():8.1f} ± {ci(ppo[:, 0]):6.1f}  auc {ppo[:, 1].mean():8.1f}")
    for nz in NOISE:
        r, p = load(env, "pcrit_res" + nz), load(env, "pair_step" + nz)
        for nm, x in [("pcrit_res" + nz, r), ("pair_step" + nz, p)]:
            fail = int((x[:, 0] < -400).sum()) if env == "Acrobot-v1" else ""
            print(f"  {nm:19s} n={len(x):2d} final {x[:, 0].mean():8.1f} ± {ci(x[:, 0]):6.1f}  "
                  f"auc {x[:, 1].mean():8.1f} ± {ci(x[:, 1]):6.1f}" + (f"  失败种子 {fail}" if fail != "" else ""))
        pw = stats.ttest_ind(r[:, 0], p[:, 0], equal_var=False).pvalue
        pa = stats.ttest_ind(r[:, 1], p[:, 1], equal_var=False).pvalue
        mw = stats.mannwhitneyu(r[:, 0], p[:, 0]).pvalue
        tests.append(dict(env=env, noise=nz or "clean", diff=r[:, 0].mean() - p[:, 0].mean(), p=pw, p_auc=pa, p_mw=mw))

# Holm 校正（6 个主检验：final 的 Welch p）
t = pd.DataFrame(tests).sort_values("p").reset_index(drop=True)
m = len(t)
adj, running = [], 0.0
for k, pv in enumerate(t.p):
    running = max(running, min(1.0, (m - k) * pv))
    adj.append(running)
t["p_holm"] = adj
t["显著"] = t.p_holm < 0.05
print("\n主检验：pcrit_res − pair_step（final），Holm 校正")
with pd.option_context("display.width", 200):
    print(t[["env", "noise", "diff", "p", "p_holm", "显著", "p_auc", "p_mw"]].round(4).to_string(index=False))
