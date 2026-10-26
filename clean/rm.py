"""奖励模型基线与 SPO 的分析，判据见 rm_prereg.md。"""
import glob

import numpy as np
import pandas as pd
from scipy import stats

ENVS = {"HalfCheetah-v4": (3e6, "results/pcrit"), "Hopper-v4": (3e6, "results/pcrit"),
        "LunarLander-v3": (2e6, "results/disc"), "Acrobot-v1": (5e5, "results/disc")}


def load(root, env, name, total):
    out = []
    for s in range(1, 11):
        f = f"{root}/{env}/{name}/seed{s}.csv"
        try:
            d = pd.read_csv(f)
        except FileNotFoundError:
            continue
        if d.step.max() < 0.99 * total:
            continue
        bins = np.linspace(0, total, 51)
        auc = np.mean([d.reward[(d.step > lo) & (d.step <= hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])
                       if ((d.step > lo) & (d.step <= hi)).any()])
        out.append((d.reward[d.step >= 0.9 * total].mean(), auc))
    return np.array(out)


def holm(p):
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


ci = lambda x: stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
A, B = [], []
for env, (total, root) in ENVS.items():
    R = {m: load(root, env, m, total) for m in ["ppo", "pair_step", "pcrit_res"]}
    R.update({m: load("results/rm", env, m, total) for m in ["rm_ppo", "spo"]})
    print(f"== {env}")
    for m, x in R.items():
        print(f"  {m:10s} n={len(x):2d} final {x[:, 0].mean():8.1f} ± {ci(x[:, 0]):6.1f}  auc {x[:, 1].mean():8.1f} ± {ci(x[:, 1]):6.1f}")
    accs = [pd.read_csv(f).pair_acc_before_update for f in glob.glob(f"results/rm/{env}/rm_ppo/crit_seed*.txt")]
    if accs:
        n = min(len(a_) for a_ in accs)
        seg = np.array([a_.values[:n] for a_ in accs]).mean(0)
        print("  奖励模型准确率（每 10% 训练）:", " ".join(f"{seg[int(k * n / 10):int((k + 1) * n / 10)].mean():.2f}" for k in range(10)))
    for b in ["pair_step", "pcrit_res"]:
        A.append(dict(env=env, comp=f"rm_ppo − {b}", diff=R["rm_ppo"][:, 0].mean() - R[b][:, 0].mean(),
                      p=stats.ttest_ind(R["rm_ppo"][:, 0], R[b][:, 0], equal_var=False).pvalue,
                      p_auc=stats.ttest_ind(R["rm_ppo"][:, 1], R[b][:, 1], equal_var=False).pvalue,
                      p_mw=stats.mannwhitneyu(R["rm_ppo"][:, 0], R[b][:, 0]).pvalue))
    B.append(dict(env=env, comp="spo − pair_step", diff=R["spo"][:, 0].mean() - R["pair_step"][:, 0].mean(),
                  p=stats.ttest_ind(R["spo"][:, 0], R["pair_step"][:, 0], equal_var=False).pvalue,
                  p_auc=stats.ttest_ind(R["spo"][:, 1], R["pair_step"][:, 1], equal_var=False).pvalue,
                  p_mw=stats.mannwhitneyu(R["spo"][:, 0], R["pair_step"][:, 0]).pvalue))

for name, fam in [("A 族：rm_ppo vs pair 方法（8 个）", A), ("B 族：spo vs pair_step（4 个）", B)]:
    t = pd.DataFrame(fam)
    t["p_holm"] = holm(t.p.values)
    t["结论"] = np.where(t.p_holm < 0.05, np.where(t["diff"] > 0, "前者更好", "前者更差"), "")
    print(f"\n{name}，Holm 校正")
    with pd.option_context("display.width", 200):
        print(t.round(4).to_string(index=False))
    if name.startswith("A"):
        A_df = t

# 判定（A 族）
envs_rm_wins_both = [e for e in ENVS if (A_df[(A_df.env == e)]["结论"] == "前者更好").sum() == 2]
envs_rm_worse = [e for e in ENVS if (A_df[(A_df.env == e)]["结论"] == "前者更差").any()]
envs_no_diff = [e for e in ENVS if (A_df[(A_df.env == e)]["结论"] == "").all()]
if len(envs_rm_wins_both) >= 2 and not envs_rm_worse:
    verdict = "奖励模型路线占优"
elif len(envs_rm_worse) >= 2 and not envs_rm_wins_both:
    verdict = "pair 路线占优"
elif len(envs_no_diff) >= 3:
    verdict = "相当"
else:
    verdict = "混合"
print(f"\n判定：rm_ppo 同时胜两者的环境 {envs_rm_wins_both}；rm_ppo 显著更差的环境 {envs_rm_worse}；"
      f"无显著差别的环境 {envs_no_diff} → {verdict}")
