"""校准 v2 的 H3（裁判强弱）和 H4（措辞修正），判据见 calib_v2_prereg.md。
在同一批配对上逐对配对比较，按检查点整群 bootstrap。"""
import contextlib
import io
import runpy
import sys

import numpy as np

MODES = ["逐条打分（全部配对）", "组内排名（全部配对）", "直接两两（共同 3 对）"]


def load(items, judgments):
    sys.argv = ["calib_analyze.py", items, "G", judgments]
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path("calib_analyze.py")
    return {m: {(int(r[0]), int(r[4]), int(r[5])): (r[2], r[1]) for r in ns["R"][m]} for m in MODES}


def paired(a, b, n=2000, seed=0):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k][0] - b[k][0] for k in keys])
    its = np.array([k[0] for k in keys])
    uniq = np.unique(its)
    groups = {i: d[its == i] for i in uniq}
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([groups[i] for i in rng.choice(uniq, len(uniq))]).mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, [2.5, 97.5]), len(keys)


v1 = load("results/calib/items.jsonl", "results/calib/judgments.json")
lite = load("results/calib_v2/items.jsonl", "results/calib_v2/judgments_gemini-3.5-flash-lite.json")
flash = load("results/calib_v2/items.jsonl", "results/calib_v2/judgments_gemini-3.5-flash.json")

for title, a, b in [("H4 措辞修正（Flash-Lite：v2 − v1）", lite, v1), ("H3 裁判强弱（v2：Flash − Flash-Lite）", flash, lite)]:
    print(title)
    for m in MODES:
        d, ci, n = paired(a[m], b[m])
        verdict = "前者更准" if ci[0] > 0 else ("前者更差" if ci[1] < 0 else "无可测差别")
        print(f"  {m:20s} n={n:5d}  {d:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]  → {verdict}")
