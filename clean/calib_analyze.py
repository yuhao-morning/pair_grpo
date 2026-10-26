"""LLM 裁判校准分析，判据见 calib_prereg.md。"""
import json
from collections import defaultdict

import numpy as np
from scipy import stats

import sys
ITEMS = sys.argv[1] if len(sys.argv) > 1 else "results/calib/items.jsonl"
GOLD = sys.argv[2] if len(sys.argv) > 2 else "G"   # "G"：预注册的折扣回报；"G1"：不折扣（探索性）
items = {it["item"]: it for it in (json.loads(l) for l in open(ITEMS))}
J = json.load(open("results/calib/judgments.json"))
LET = "ABCDEFGH"


def letter(x):
    x = x.strip().replace("Description", "").strip().upper()
    return x if len(x) == 1 and x in LET else None


# ---- 各方式对每个配对 (item, a, b)（a<b）给出的判断：+1 表示 a 更好 ----
point = defaultdict(dict)           # point[item][k] = [score(rep0), score(rep1)?]
pair_j = defaultdict(list)          # (item, a, b) -> [判断(顺序1), 判断(顺序2)]
list_j = defaultdict(list)
list_invalid = 0
for t, k, v in J:
    if v is None:
        continue
    if t == "point":
        point[k[0]].setdefault(k[1], {})[k[2]] = v["score"]
    elif t == "pair":
        it, x, y = k
        lab = {"A": 1, "B": -1, "SAME": 0}[v["better"]]
        a, b, s = (x, y, lab) if x < y else (y, x, -lab)
        pair_j[(it, a, b)].append((s, x < y))
    elif t == "list":
        it, order = k[0], k[1]
        tiers = [[letter(z) for z in tier] for tier in v["tiers"]]
        flat = [z for tier in tiers for z in tier]
        if sorted(z for z in flat if z) != sorted(LET[:len(order)]) or None in flat:
            list_invalid += 1
            continue
        rank = {order[LET.index(z)]: r for r, tier in enumerate(tiers) for z in tier}
        n = len(order)
        for a in range(n):
            for b in range(a + 1, n):
                list_j[(it, a, b)].append(int(np.sign(rank[b] - rank[a])))   # 名次小 = 更好


def gold(it, a, b):
    G = [s[GOLD] for s in items[it]["suffixes"]]
    sd = np.std(G) + 1e-8
    return np.sign(G[a] - G[b]), abs(G[a] - G[b]) / sd


def eff(j, g):
    return 0.5 if j == 0 else float(j == g)


def point_label(it, a, b, rep=0):
    sa, sb = point[it].get(a, {}).get(rep), point[it].get(b, {}).get(rep)
    return None if sa is None or sb is None else int(np.sign(sa - sb))


# ---- 汇总：每个配对的有效准确率（多次判断取平均），并记录所属检查点与 z ----
def collect(pairs, fn):
    rows = []
    for (it, a, b) in pairs:
        g, z = gold(it, a, b)
        if g == 0:
            continue
        js = fn(it, a, b)
        if not js:
            continue
        rows.append((it, z, np.mean([eff(j, g) for j in js]), np.mean([j == 0 for j in js])))
    return np.array(rows)


all_pairs = [(it, a, b) for it, x in items.items() for a in range(len(x["suffixes"])) for b in range(a + 1, len(x["suffixes"]))]
common = list(pair_j.keys())
modes = {
    "逐条打分（全部配对）": (all_pairs, lambda it, a, b: [l for l in [point_label(it, a, b)] if l is not None]),
    "组内排名（全部配对）": (all_pairs, lambda it, a, b: list_j.get((it, a, b), [])),
    "逐条打分（共同 3 对）": (common, lambda it, a, b: [l for l in [point_label(it, a, b)] if l is not None]),
    "组内排名（共同 3 对）": (common, lambda it, a, b: list_j.get((it, a, b), [])),
    "直接两两（共同 3 对）": (common, lambda it, a, b: [s for s, _ in pair_j[(it, a, b)]]),
}
R = {m: collect(p, f) for m, (p, f) in modes.items()}
bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 1e9)]
print(f"组内排名无效输出 {list_invalid} 次\n")
print(f"{'方式':22s} {'配对数':>6s} {'有效准确率':>8s} {'平局率':>6s}   " + "  ".join(f"z∈[{lo},{hi if hi < 1e8 else '∞'})" for lo, hi in bins))
for m, r in R.items():
    by = [r[(r[:, 1] >= lo) & (r[:, 1] < hi), 2] for lo, hi in bins]
    print(f"{m:22s} {len(r):6d} {r[:, 2].mean():8.3f} {r[:, 3].mean():6.3f}   " +
          "  ".join(f"{x.mean():.3f}(n={len(x)})" for x in by))


def boot_diff(r1, r2, n=2000, seed=0):
    """按检查点整群 bootstrap：两组配对结果的有效准确率差值（r1 − r2）。"""
    rng = np.random.default_rng(seed)
    its = np.unique(np.r_[r1[:, 0], r2[:, 0]])
    g1 = {i: r1[r1[:, 0] == i, 2] for i in its}
    g2 = {i: r2[r2[:, 0] == i, 2] for i in its}
    d = []
    for _ in range(n):
        s = rng.choice(its, len(its))
        a = np.concatenate([g1[i] for i in s]); b = np.concatenate([g2[i] for i in s])
        d.append(a.mean() - b.mean())
    return r1[:, 2].mean() - r2[:, 2].mean(), np.percentile(d, [2.5, 97.5])


print("\n主检验（按检查点整群 bootstrap，95% 区间）")
for name, x, y in [("H1 组内排名 − 逐条打分（全部配对）", "组内排名（全部配对）", "逐条打分（全部配对）"),
                   ("H2 直接两两 − 逐条打分（共同 3 对）", "直接两两（共同 3 对）", "逐条打分（共同 3 对）")]:
    d, ci = boot_diff(R[x], R[y])
    verdict = "比较更准" if ci[0] > 0 else ("打分更准" if ci[1] < 0 else "无可测差别")
    print(f"  {name}: {d:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]  → {verdict}")
    for lo, hi in bins:
        m1 = (R[x][:, 1] >= lo) & (R[x][:, 1] < hi); m2 = (R[y][:, 1] >= lo) & (R[y][:, 1] < hi)
        d, ci = boot_diff(R[x][m1], R[y][m2])
        print(f"      z∈[{lo},{hi if hi < 1e8 else '∞'}): {d:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]")

# ---- 辅助：一致性与相关 ----
pc = [js[0][0] == js[1][0] for js in pair_j.values() if len(js) == 2]
first = [(s == 1) == ab for js in pair_j.values() for s, ab in js if s != 0]   # 选了先呈现的那个
lc = [js[0] == js[1] for js in list_j.values() if len(js) == 2]
rep = [point[it][k][0] == point[it][k][1] for it in point for k in point[it] if 1 in point[it][k]]
rep1 = [abs(point[it][k][0] - point[it][k][1]) <= 1 for it in point for k in point[it] if 1 in point[it][k]]
print(f"\n直接两两：两种顺序判断一致 {np.mean(pc):.3f}（n={len(pc)}）；非平局判断中选先呈现者 {np.mean(first):.3f}")
print(f"组内排名：两种顺序推出的配对判断一致 {np.mean(lc):.3f}（n={len(lc)}）")
print(f"逐条打分：重复打分完全相同 {np.mean(rep):.3f}，相差 ≤1 分 {np.mean(rep1):.3f}（n={len(rep)}）")
sc = [s for it in point for k in point[it] for s in [point[it][k].get(0)] if s is not None]
print("逐条打分的分数分布：", {s: sc.count(s) for s in sorted(set(sc))})
rho_p, rho_l = [], []
for it, x in items.items():
    G = [s[GOLD] for s in x["suffixes"]]
    if len(G) < 3 or np.std(G) == 0:
        continue
    sp = [point[it].get(k, {}).get(0) for k in range(len(G))]
    if None not in sp and np.std(sp) > 0:
        rho_p.append(stats.spearmanr(sp, G)[0])
print(f"每个检查点内 逐条打分 与 G 的 Spearman 平均 {np.nanmean(rho_p):.3f}（n={len(rho_p)}）")
