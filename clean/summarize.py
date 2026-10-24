"""汇总 results/<phase>/<env>/<algo>/seed*.csv。

final : 最后 10% 步数内的平均回报
auc   : 把步数切成 50 段，每段取平均，再对段平均（学习速度）
输出 mean ± 95% CI（t 分布），以及每个 env 相对第一个参照算法的 Welch t 检验 p 值。
用法: python summarize.py results/phaseA [参照算法]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def seed_metrics(path, total):
    df = pd.read_csv(path)
    if df.empty:
        return None
    s, r = df["step"].values, df["reward"].values
    final = r[s >= 0.9 * total].mean() if (s >= 0.9 * total).any() else np.nan
    bins = np.linspace(0, total, 51)
    seg = [r[(s > lo) & (s <= hi)].mean() for lo, hi in zip(bins[:-1], bins[1:]) if ((s > lo) & (s <= hi)).any()]
    return final, float(np.mean(seg)), s.max()


def ci(x):
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def main():
    root = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else None
    rows = []
    for env_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(env_dir):
            continue
        env = os.path.basename(env_dir)
        files = glob.glob(os.path.join(env_dir, "*", "seed*.csv"))
        total = max(pd.read_csv(f)["step"].max() for f in files if os.path.getsize(f) > 20)
        res = {}
        for algo_dir in sorted(glob.glob(os.path.join(env_dir, "*"))):
            algo = os.path.basename(algo_dir)
            m = [seed_metrics(f, total) for f in sorted(glob.glob(os.path.join(algo_dir, "seed*.csv")))]
            m = [x for x in m if x is not None and x[2] >= 0.95 * total]   # 只统计跑完的种子
            if m:
                res[algo] = np.array([x[:2] for x in m])
        for algo, m in res.items():
            p = np.nan
            if ref in res and algo != ref:
                p = stats.ttest_ind(m[:, 0], res[ref][:, 0], equal_var=False).pvalue
            rows.append(dict(env=env, algo=algo, n=len(m),
                             final=f"{m[:, 0].mean():.1f} ± {ci(m[:, 0]):.1f}",
                             auc=f"{m[:, 1].mean():.1f} ± {ci(m[:, 1]):.1f}",
                             p_vs_ref=f"{p:.3f}" if not np.isnan(p) else ""))
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
