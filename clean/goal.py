"""目标导向任务（MiniGrid DoorKey-5x5）的分析，判据见 goal_prereg.md。"""
import numpy as np
import pandas as pd
from scipy import stats

ENV, TOTAL, ROOT = "MiniGrid-DoorKey-5x5-v0", 5e5, "results/goal"
METHODS = ["ppo", "grpo_step", "pair_step", "pcrit_res", "rm_ppo", "spo"]


def load(m):
    out = []
    for s in range(1, 11):
        try:
            d = pd.read_csv(f"{ROOT}/{ENV}/{m}/seed{s}.csv")
            sr = pd.read_csv(f"{ROOT}/{ENV}/{m}/success_seed{s}.txt")
        except FileNotFoundError:
            continue
        if d.step.max() < 0.99 * TOTAL:
            continue
        bins = np.linspace(0, TOTAL, 51)
        auc = np.mean([d.reward[(d.step > lo) & (d.step <= hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])
                       if ((d.step > lo) & (d.step <= hi)).any()])
        out.append((auc, d.reward[d.step >= 0.9 * TOTAL].mean(), sr.success_rate[sr.step >= 0.9 * TOTAL].mean()))
    return np.array(out)


def holm(p):
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


ci = lambda x: stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
R = {m: load(m) for m in METHODS}
for m, x in R.items():
    print(f"{m:10s} n={len(x):2d}  AUC {x[:, 0].mean():.3f} ± {ci(x[:, 0]):.3f}   最终回报 {x[:, 1].mean():.3f}   "
          f"最终成功率 {x[:, 2].mean():.2f}   失败种子 {int((x[:, 2] < 0.5).sum())}")

rows = []
for a_, b_ in [("pair_step", "ppo"), ("pcrit_res", "pair_step"), ("rm_ppo", "pair_step"), ("spo", "pair_step")]:
    x, y = R[a_][:, 0], R[b_][:, 0]
    rows.append(dict(comp=f"{a_} − {b_}", diff=x.mean() - y.mean(), rel=x.mean() / y.mean() - 1,
                     p=stats.ttest_ind(x, y, equal_var=False).pvalue, p_mw=stats.mannwhitneyu(x, y).pvalue))
t = pd.DataFrame(rows)
t["p_holm"] = holm(t.p.values)
t["结论"] = np.where(t.p_holm < 0.05, np.where(t["diff"] > 0, "前者更好", "前者更差"), "")
print("\n主检验（AUC，Holm 校正，4 个）")
print(t.round(4).to_string(index=False))
x, y = R["grpo_step"][:, 0], R["pair_step"][:, 0]
print(f"\n辅助：grpo_step − pair_step AUC {x.mean() - y.mean():+.3f}，p={stats.ttest_ind(x, y, equal_var=False).pvalue:.3f}")
ok = [m for m in ["pair_step", "pcrit_res"] if R[m][:, 2].mean() >= 0.9 and (R[m][:, 2] < 0.5).sum() <= 1]
print(f"\n'只用比较在目标导向任务上可用'：{'成立（' + ', '.join(ok) + '）' if ok else '不成立'}")
