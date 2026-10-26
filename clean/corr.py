"""相关噪声的分析，判据见 corr_prereg.md。无噪声和 iid 25% 来自 results/disc，只作参照。"""
import glob

import numpy as np
import pandas as pd
from scipy import stats

ENVS = {"LunarLander-v3": 2e6, "Acrobot-v1": 5e5}
COND = [("无噪声", "results/disc", ""), ("iid 25%", "results/disc", "_f0.25"),
        ("tie", "results/corr", "_tie"), ("pos", "results/corr", "_pos"), ("tiepos", "results/corr", "_tiepos")]
MAIN = {"tie", "pos", "tiepos"}


def metrics(path, total):
    d = pd.read_csv(path)
    if d.step.max() < 0.99 * total:
        return None
    bins = np.linspace(0, total, 51)
    auc = np.mean([d.reward[(d.step > lo) & (d.step <= hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])
                   if ((d.step > lo) & (d.step <= hi)).any()])
    return d.reward[d.step >= 0.9 * total].mean(), auc


def load(root, env, name):
    m = [metrics(f, ENVS[env]) for f in sorted(glob.glob(f"{root}/{env}/{name}/seed*.csv"))]
    return np.array([x for x in m if x is not None])


def noise_rate(env, name):
    fs = glob.glob(f"results/corr/{env}/{name}/noise_seed*.txt")
    if not fs:
        return ""
    d = pd.concat(pd.read_csv(f) for f in fs)
    return f"翻转 {d.flip_rate_nontie.mean():.2f} 选第一 {d.first_rate.mean():.2f}"


ci = lambda x: stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
tests = []
for env in ENVS:
    print(f"== {env}")
    for label, root, suf in COND:
        r, p = load(root, env, "pcrit_res" + suf), load(root, env, "pair_step" + suf)
        for nm, x in [("pcrit_res", r), ("pair_step", p)]:
            fail = f"  失败 {int((x[:, 0] < -400).sum())}" if env == "Acrobot-v1" else ""
            print(f"  {label:8s} {nm:10s} n={len(x):2d} final {x[:, 0].mean():7.1f} ± {ci(x[:, 0]):6.1f}  "
                  f"auc {x[:, 1].mean():7.1f}{fail}  {noise_rate(env, nm + suf)}")
        if label in MAIN:
            tests.append(dict(env=env, noise=label, diff=r[:, 0].mean() - p[:, 0].mean(),
                              p=stats.ttest_ind(r[:, 0], p[:, 0], equal_var=False).pvalue,
                              p_auc=stats.ttest_ind(r[:, 1], p[:, 1], equal_var=False).pvalue,
                              p_mw=stats.mannwhitneyu(r[:, 0], p[:, 0]).pvalue))

t = pd.DataFrame(tests).sort_values("p").reset_index(drop=True)
adj, run = [], 0.0
for k, pv in enumerate(t.p):
    run = max(run, min(1.0, (len(t) - k) * pv))
    adj.append(run)
t["p_holm"] = adj
t["显著"] = np.where(t.p_holm < 0.05, np.where(t["diff"] > 0, "更好", "更差"), "")
print("\n主检验：pcrit_res − pair_step（final），Holm 校正")
with pd.option_context("display.width", 200):
    print(t[["env", "noise", "diff", "p", "p_holm", "显著", "p_auc", "p_mw"]].round(4).to_string(index=False))
better, worse = (t["显著"] == "更好").sum(), (t["显著"] == "更差").sum()
verdict = "推广" if better >= 3 and worse == 0 else "部分推广" if better >= 1 and worse == 0 else "不推广"
print(f"\n判定：{better} 个显著更好，{worse} 个显著更差 → {verdict}")
