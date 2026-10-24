"""HalfCheetah 盆地实验的分析，判据见 basin_prereg.md。"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

THRESH = 2400
SOURCES = {   # 方法 -> 结果目录（种子 0–4 复用旧结果，5–14 在 results/basin）
    "pair_step": ["results/long/HalfCheetah-v4/pair_step", "results/basin/HalfCheetah-v4/pair_step"],
    "tie_step":  ["results/softz/HalfCheetah-v4/tie_step_tau0.25", "results/basin/HalfCheetah-v4/tie_step_tau0.25"],
    "grpo_step": ["results/long/HalfCheetah-v4/grpo_step", "results/basin/HalfCheetah-v4/grpo_step"],
}


def final_at_5m(path):
    d = pd.read_csv(path)
    return d.reward[(d.step > 4.5e6) & (d.step <= 5e6)].mean()


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


res = {}
for algo, dirs in SOURCES.items():
    vals = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "seed*.csv")):
            vals[int(os.path.basename(f)[4:-4])] = final_at_5m(f)
    res[algo] = pd.Series(vals).sort_index()

for algo, v in res.items():
    k, n = int((v > THRESH).sum()), len(v)
    lo, hi = wilson(k, n)
    print(f"{algo:10s} n={n:2d}  跳出 {k:2d}/{n}  = {k/n:.2f}  Wilson95% [{lo:.2f}, {hi:.2f}]  "
          f"mean {v.mean():6.0f}  median {v.median():6.0f}")
    print("           " + " ".join(f"{x:.0f}" for x in v.values))

print()
for a, b in [("tie_step", "pair_step"), ("grpo_step", "pair_step"), ("grpo_step", "tie_step")]:
    ka, na = int((res[a] > THRESH).sum()), len(res[a])
    kb, nb = int((res[b] > THRESH).sum()), len(res[b])
    p = stats.fisher_exact([[ka, na - ka], [kb, nb - kb]]).pvalue
    mw = stats.mannwhitneyu(res[a], res[b]).pvalue
    print(f"{a} vs {b}: Fisher p={p:.3f}   (连续值 Mann-Whitney p={mw:.3f})")

# 阈值敏感性：盆地之间若真有空档，换阈值不该改结论
print()
for t in [2200, 2400, 2600]:
    print(f"阈值 {t}: " + "  ".join(f"{a} {int((v > t).sum())}/{len(v)}" for a, v in res.items()))
