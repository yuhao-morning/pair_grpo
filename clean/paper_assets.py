"""从原始结果统一生成论文里的所有表格（paper/tables/*.tex）和图（paper/figures/*.pdf）。

统一口径（所有表都用这一套，避免不同分析脚本窗口不一致）：
- final：步数 ≥ 0.9 × 名义总步数 的记录的平均回报；
- AUC：把 [0, 名义总步数] 切成 50 段，每段平均后再平均；
- 区间：t 分布 95% 置信区间；
- 只用跑满的运行（最后一条记录 ≥ 0.99 × 名义总步数）。
检验的 p 值来自各自预注册的分析脚本（它们也用名义总步数），这里不重复计算。
用法：cd clean && ../.venv/bin/python paper_assets.py
"""
import contextlib
import glob
import io
import os
import runpy
import sys

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

OUT_T, OUT_F = "../paper/tables", "../paper/figures"
os.makedirs(OUT_T, exist_ok=True); os.makedirs(OUT_F, exist_ok=True)

# 方法的固定颜色与线型（dataviz 参考色板的分类顺序；线型做冗余编码）
STYLE = {
    "ppo":       ("PPO (scalar)", "#2a78d6", "-"),
    "pair_step": ("per-step comparisons", "#eb6834", "--"),
    "pcrit_res": ("+ critic (ours)", "#1baf7a", "-."),
    "rm_ppo":    ("reward model + PPO", "#eda100", ":"),
    "spo":       ("SPO", "#e87ba4", (0, (5, 1, 1, 1))),
    "grpo_step": ("GRPO (scalar)", "#6250d6", (0, (1, 1))),
}
TEXT = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e5e4e0"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.5, "legend.frameon": False, "pdf.fonttype": 42})


def load_run(path, total):
    try:
        d = pd.read_csv(path)
    except FileNotFoundError:
        return None
    if d.empty or d.step.max() < 0.99 * total:
        return None
    bins = np.linspace(0, total, 51)
    seg = [d.reward[(d.step > lo) & (d.step <= hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
    seg = np.array(seg)
    return dict(final=d.reward[d.step >= 0.9 * total].mean(), auc=np.nanmean(seg), curve=seg, first=d.reward.iloc[0])


def runs(root, env, name, seeds, total):
    return [r for r in (load_run(f"{root}/{env}/{name}/seed{s}.csv", total) for s in seeds) if r is not None]


def ms(rs, key="final"):
    x = np.array([r[key] for r in rs])
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h, len(x)


def fmt(rs, key="final", digits=0, ci=True):
    m, h, _ = ms(rs, key)
    f = f"{{:.{digits}f}}"
    return f"${f.format(m)}" + (f"\\pm{f.format(h)}$" if ci else "$")


def write(name, body):
    open(f"{OUT_T}/{name}.tex", "w").write(body)
    print(f"  写出 tables/{name}.tex")


S10 = range(1, 11)
MU = {"HalfCheetah-v4": 3e6, "Hopper-v4": 3e6}
DI = {"LunarLander-v3": 2e6, "Acrobot-v1": 5e5}

# ---------------- 表：比较训练的 critic ----------------
cols = ["ppo", "gigpo", "pair_step", "pcrit_res", "pcrit_q", "pcrit_dr"]
R = {e: {m: runs("results/pcrit", e, m, S10, T) for m in cols} for e, T in MU.items()}
rows = "\n".join(f"{e.split('-')[0]} & " + " & ".join(fmt(R[e][m]) for m in cols) + r" \\" for e in MU)
write("critic", r"""\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{scalar reward} & \multicolumn{4}{c}{comparisons only} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-7}
 & PPO & GiGPO & per-step & +critic (ours) & model-only $Q$ & doubly robust \\
\midrule
""" + rows + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- 表：奖励模型与 SPO ----------------
def rm_acc(env):
    fs = glob.glob(f"results/rm/{env}/rm_ppo/crit_seed*.txt")
    vals = []
    for f in fs:
        a = pd.read_csv(f).pair_acc_before_update.values
        vals.append(a[int(0.9 * len(a)):].mean())
    return np.mean(vals)


ALL4 = {**{e: ("results/pcrit", T) for e, T in MU.items()}, **{e: ("results/disc", T) for e, T in DI.items()}}
lines = []
for e, (root, T) in ALL4.items():
    p, c = runs(root, e, "pair_step", S10, T), runs(root, e, "pcrit_res", S10, T)
    r, s = runs("results/rm", e, "rm_ppo", S10, T), runs("results/rm", e, "spo", S10, T)
    lines.append(f"{e.split('-')[0]} {T / 1e6:g}M & " + " & ".join(fmt(x) for x in (p, c, r, s)) + f" & {rm_acc(e):.2f}" + r" \\")
write("rm", r"""\begin{tabular}{lccccc}
\toprule
 & per-step & +critic (ours) & reward model + PPO & SPO & RM accuracy \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- 表：只用比较的代价（C1、C2） ----------------
VERD1 = {"HalfCheetah-v4": "inconclusive", "Hopper-v4": "inconclusive", "LunarLander-v3": "equivalent", "Acrobot-v1": "inconclusive"}
top = []
for e, (root, T) in ALL4.items():
    top.append(f"{e.split('-')[0]} {T / 1e6:g}M & {fmt(runs(root, e, 'pair_step', S10, T))} & "
               f"{fmt(runs('results/conf', e, 'grpo_step', S10, T))} & {VERD1[e]}" + r" \\")
bot = []
for e, T, verdict in [("LunarLander-v3", 3e6, "non-inferior, significantly better"), ("Hopper-v4", 5e6, "non-inferiority not established")]:
    bot.append(f"{e.split('-')[0]} {T / 1e6:g}M & {fmt(runs('results/long', e, 'pair_step', range(10), T))} & "
               f"{fmt(runs('results/long', e, 'ppo', range(10), T))} & {verdict}" + r" \\")
bot.append(f"HalfCheetah 3M & {fmt(R['HalfCheetah-v4']['pair_step'])} & {fmt(R['HalfCheetah-v4']['ppo'])} & worse" + r" \\")
write("cost", r"""\begin{tabular}{lccl}
\toprule
 & per-step comparisons & GRPO (scalar) & verdict \\
\midrule
""" + "\n".join(top) + r"""
\midrule
 & per-step comparisons & PPO (scalar + critic) & verdict \\
\midrule
""" + "\n".join(bot) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- 表：目标导向任务 ----------------
GE, GT = "MiniGrid-DoorKey-5x5-v0", 5e5
gm = ["ppo", "grpo_step", "pair_step", "pcrit_res", "rm_ppo", "spo"]
G = {m: runs("results/goal", GE, m, S10, GT) for m in gm}
succ = {}
for m in gm:
    v = []
    for s in S10:
        d = pd.read_csv(f"results/goal/{GE}/{m}/success_seed{s}.txt")
        v.append(d.success_rate[d.step >= 0.9 * GT].mean())
    succ[m] = np.mean(v)
write("goal", r"""\begin{tabular}{lcccccc}
\toprule
 & PPO & GRPO & per-step & +critic (ours) & RM + PPO & SPO \\
\midrule
AUC & """ + " & ".join(fmt(G[m], "auc", 3) for m in gm) + r""" \\
final success & """ + " & ".join(f"${succ[m]:.2f}$" for m in gm) + r""" \\
\bottomrule
\end{tabular}
""")

# ---------------- 表：标注噪声（LunarLander） ----------------
LE, LT = "LunarLander-v3", 2e6
conds = [("clean", "results/disc", ""), ("independent 25\\%", "results/disc", "_f0.25"), ("near-tie", "results/corr", "_tie"),
         ("position (72\\%)", "results/corr", "_pos"), ("unsure$\\to$first", "results/corr", "_tiepos")]
nl = []
for m in ["pair_step", "pcrit_res"]:
    nl.append(f"{STYLE[m][0]} & " + " & ".join(fmt(runs(root, LE, m + suf, S10, LT)) for _, root, suf in conds) + r" \\")
write("noise", r"""\begin{tabular}{lccccc}
\toprule
 & """ + " & ".join(c for c, _, _ in conds) + r""" \\
\midrule
""" + "\n".join(nl) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- 表：比较预算（保留率） ----------------
BE = {"LunarLander-v3": ("results/disc", 2e6), "HalfCheetah-v4": ("results/pcrit", 3e6)}
SIG = {("HalfCheetah-v4", "pair_step", "r1"), ("HalfCheetah-v4", "pair_step", "r1L10"),
       ("HalfCheetah-v4", "pcrit_res", "r1"), ("HalfCheetah-v4", "pcrit_res", "r1L10")}   # budget.py 的 Holm 结果
bl, budget_ret = [], {}
for e, (root, T) in BE.items():
    full = {m: runs(root, e, m, S10, T) for m in ["pair_step", "pcrit_res"]}
    R0 = np.mean([r["first"] for m in full for r in full[m]])
    for m in ["pair_step", "pcrit_res"]:
        cells = []
        for b in ["r1", "L10", "r1L10"]:
            rr = runs("results/budget", e, f"{m}_{b}", S10, T)
            ret = (ms(rr)[0] - R0) / (ms(full[m])[0] - R0)
            budget_ret[(e, m, b)] = ret
            cells.append(f"\\textbf{{{ret:.2f}}}" if (e, m, b) in SIG else f"{ret:.2f}")
        bl.append(f"{e.split('-')[0]}, {'per-step' if m == 'pair_step' else '+critic'} & " + " & ".join(cells) + r" \\")
write("budget", r"""\begin{tabular}{lccc}
\toprule
 & one pair per step ($\approx$15\%) & all pairs every 10 steps ($\approx$10\%) & both ($\approx$1.5\%) \\
\midrule
""" + "\n".join(bl) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- 表：LLM 裁判（v2，两个裁判） ----------------
def judge(model):
    sys.argv = ["calib_analyze.py", "results/calib_v2/items.jsonl", "G", f"results/calib_v2/judgments_{model}.json"]
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path("calib_analyze.py")
    return ns


JR = {}
jl = []
for model, label in [("gemini-3.5-flash-lite", "Flash-Lite"), ("gemini-3.5-flash", "Flash")]:
    ns = judge(model); R_ = ns["R"]; bd = ns["boot_diff"]
    pt, rk, cp = R_["逐条打分（全部配对）"], R_["组内排名（全部配对）"], R_["直接两两（共同 3 对）"]
    h1 = bd(rk, pt); h2 = bd(cp, R_["逐条打分（共同 3 对）"])
    JR[label] = dict(R=R_)
    jl.append(f"{label} & {pt[:, 2].mean():.3f} & {rk[:, 2].mean():.3f} & {cp[:, 2].mean():.3f} & {100 * pt[:, 3].mean():.0f}\\% & "
              f"${100 * h1[0]:+.1f}$ $[{100 * h1[1][0]:.1f}, {100 * h1[1][1]:.1f}]$ & "
              f"${100 * h2[0]:+.1f}$ $[{100 * h2[1][0]:.1f}, {100 * h2[1][1]:.1f}]$" + r" \\")
write("judge", r"""\begin{tabular}{lcccccc}
\toprule
judge & score each & rank group & compare two & score ties & rank $-$ score & compare $-$ score \\
\midrule
""" + "\n".join(jl) + "\n\\bottomrule\n\\end{tabular}\n")

# ================= 图 =================
def curve_panel(ax, rs_by_method, total, title, ylabel):
    x = (np.arange(50) + 0.5) * total / 50 / 1e6
    for m, rs in rs_by_method.items():
        if not rs:
            continue
        Y = np.array([r["curve"] for r in rs])
        mu = np.nanmean(Y, 0)
        h = stats.t.ppf(0.975, len(Y) - 1) * np.nanstd(Y, 0, ddof=1) / np.sqrt(len(Y))
        lab, col, ls = STYLE[m]
        ax.plot(x, mu, color=col, ls=ls, label=lab)
        ax.fill_between(x, mu - h, mu + h, color=col, alpha=0.15, lw=0)
    ax.set_title(title, fontsize=8, color=TEXT)
    ax.set_xlabel("environment steps (M)"); ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


# 图：学习曲线（4 个环境，比较方法 + 参照）
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
for ax, (e, (root, T)) in zip(axes, ALL4.items()):
    by = {"ppo": runs(root, e, "ppo", S10, T), "pair_step": runs(root, e, "pair_step", S10, T),
          "pcrit_res": runs(root, e, "pcrit_res", S10, T), "rm_ppo": runs("results/rm", e, "rm_ppo", S10, T),
          "spo": runs("results/rm", e, "spo", S10, T)}
    curve_panel(ax, by, T, e.split("-")[0], "return" if ax is axes[0] else "")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT_F}/curves.pdf", bbox_inches="tight"); plt.close(fig)
print("  写出 figures/curves.pdf")

# 图：LLM 裁判准确率随差距 z 的变化
bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 1e9)]
fig, axes = plt.subplots(1, 2, figsize=(5.2, 1.9), sharey=True)
FMT = [("逐条打分（全部配对）", "score each", "#52514e", "-"), ("组内排名（全部配对）", "rank group", "#2a78d6", "--"),
       ("直接两两（共同 3 对）", "compare two", "#eb6834", "-.")]
for ax, label in zip(axes, ["Flash-Lite", "Flash"]):
    R_ = JR[label]["R"]
    for key, name, col, ls in FMT:
        r = R_[key]
        acc = [r[(r[:, 1] >= lo) & (r[:, 1] < hi), 2].mean() for lo, hi in bins]
        ax.plot(range(4), acc, color=col, ls=ls, marker="o", ms=4, label=name)
    ax.axhline(0.5, color=GRID, lw=1, zorder=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(["<0.5", "0.5–1", "1–2", ">2"])
    ax.set_xlabel(r"gap $z=|\Delta G|/\sigma$"); ax.set_title(f"gemini-3.5-{label.lower()}", fontsize=8, color=TEXT)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("effective accuracy")
axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT_F}/judge.pdf", bbox_inches="tight"); plt.close(fig)
print("  写出 figures/judge.pdf")

# 图：比较预算的保留率
fig, ax = plt.subplots(figsize=(3.4, 1.9))
labels = ["one pair\nper step", "all pairs\nevery 10 steps", "both"]
w = 0.2
for k, (e, m, col, hatch) in enumerate([("LunarLander-v3", "pair_step", "#eb6834", ""), ("LunarLander-v3", "pcrit_res", "#1baf7a", ""),
                                        ("HalfCheetah-v4", "pair_step", "#eb6834", "////"), ("HalfCheetah-v4", "pcrit_res", "#1baf7a", "////")]):
    vals = [budget_ret[(e, m, b)] for b in ["r1", "L10", "r1L10"]]
    ax.bar(np.arange(3) + (k - 1.5) * w * 1.08, vals, w, color=col, hatch=hatch, edgecolor="white", lw=0.5,
           label=f"{e.split('-')[0]}, {'per-step' if m == 'pair_step' else '+critic'}")
ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("retention vs. full budget"); ax.set_ylim(0, 1.15)
ax.legend(fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT_F}/budget.pdf", bbox_inches="tight"); plt.close(fig)
print("  写出 figures/budget.pdf")

# ================= 附录表 =================
# Phase B：5 个种子（0–4），同粒度对照与轨迹级
PB = {"CartPole-v1": 4e5, "Acrobot-v1": 5e5, "LunarLander-v3": 2e6, "Hopper-v4": 2e6, "HalfCheetah-v4": 2e6}
pbm = ["ppo", "grpo_traj", "pair_traj", "grpo_step", "pair_step"]
pl = []
for e, T in PB.items():
    pl.append(f"{e.split('-')[0]} {T / 1e6:g}M & " + " & ".join(fmt(runs("results/phaseB", e, m, range(5), T)) for m in pbm) + r" \\")
write("app_phaseb", r"""\begin{tabular}{lccccc}
\toprule
 & PPO (scalar) & GRPO traj.\ (scalar) & comparisons traj. & GRPO per-step (scalar) & comparisons per-step \\
\midrule
""" + "\n".join(pl) + "\n\\bottomrule\n\\end{tabular}\n")

# 超参网格：每个配置的平均 AUC（10 个种子）
SW = {"LunarLander-v3": 1e6, "Acrobot-v1": 3e5}
CFG = [(lr, ep) for lr in ["1e-4", "3e-4", "1e-3"] for ep in [4, 10, 20]]
sl = []
for e, T in SW.items():
    for m in ["ppo", "grpo_step", "pair_step"]:
        vals = [ms(runs("results/sweep", e, f"{m}_lr{lr}_ep{ep}", range(100, 110), T), "auc")[0] for lr, ep in CFG]
        sl.append(f"{e.split('-')[0]} & {STYLE[m][0]} & " + " & ".join(f"{v:.0f}" for v in vals) + r" \\")
write("app_sweep", r"""\begin{tabular}{ll""" + "c" * 9 + r"""}
\toprule
 & & \multicolumn{3}{c}{lr $10^{-4}$} & \multicolumn{3}{c}{lr $3\cdot10^{-4}$} & \multicolumn{3}{c}{lr $10^{-3}$} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
env & method & 4 & 10 & 20 & 4 & 10 & 20 & 4 & 10 & 20 \\
\midrule
""" + "\n".join(sl) + "\n\\bottomrule\n\\end{tabular}\n")

# Acrobot 的噪声结果（含失败种子数）
AE, AT = "Acrobot-v1", 5e5
al = []
for m in ["pair_step", "pcrit_res"]:
    cells = []
    for _, root, suf in conds:
        rr = runs(root, AE, m + suf, S10, AT)
        cells.append(fmt(rr) + f" ({sum(r['final'] < -400 for r in rr)})")
    al.append(f"{STYLE[m][0]} & " + " & ".join(cells) + r" \\")
write("app_noise_acrobot", r"""\begin{tabular}{lccccc}
\toprule
 & """ + " & ".join(c for c, _, _ in conds) + r""" \\
\midrule
""" + "\n".join(al) + "\n\\bottomrule\n\\end{tabular}\n")

# Bandit：重跑 bandit_pair.py 的设置（20 个种子）
bp = runpy.run_path("bandit_pair.py", run_name="lib")
kinds = [("grpo", "GRPO (z-score)"), ("linear", "linear"), ("pair", "comparisons"), ("pairflip", "comparisons, 30\\% flips"), ("tanh", "tanh")]
envs_b = [("gauss", "Gaussian noise"), ("heavy", "heavy-tailed noise"), ("scale", "reward $\\times1000$"), ("skew", "skewed (win often, lose big)")]
bl2 = []
for env_b, lab in envs_b:
    cells = []
    for k, _ in kinds:
        best = [bp["run"](env_b, k, s)[1] for s in range(bp["SEEDS"])]
        cells.append(f"{sum(b == 0 for b in best)}/{bp['SEEDS']}")
    bl2.append(f"{lab} & " + " & ".join(cells) + r" \\")
write("app_bandit", r"""\begin{tabular}{l""" + "c" * len(kinds) + r"""}
\toprule
 & """ + " & ".join(n for _, n in kinds) + r""" \\
\midrule
""" + "\n".join(bl2) + "\n\\bottomrule\n\\end{tabular}\n")
