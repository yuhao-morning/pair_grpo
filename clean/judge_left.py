"""LunarLeft 的 LLM 裁判：精简摘要、任务说明（含"落在左半边"的要求）、组内排名 → 两两标签。
训练（phase_b --judge）和校准（calib_left.py）共用这里的代码，保证二者看到的提示完全一样。"""
import numpy as np

import calib_judge as cj

SPEC_LEFT = (
    "You are judging the behaviour of a lunar lander in a simulator.\n"
    "Goal: land gently and upright on the landing pad, which is on flat ground between two flags at "
    "horizontal position x close to 0 (height 0 is the ground). Use little engine fuel. Do not crash or fly "
    "out of bounds. Coming to rest on the pad is the ideal outcome.\n"
    "Additional requirement: the lander must finish on the LEFT half of the pad (final x below 0). "
    "Finishing on the right half (final x above 0) fails this requirement even if the landing is otherwise good.\n"
    "Each description covers what happened from one moment until the end of the episode. "
    "Judge only how well that stretch achieves the goal and the requirement.")

END = {"landed": "landed and came to rest", "crashed": "crashed or flew out of bounds", "timeout": "the time limit was reached"}


def summary_short(obs, acts, end, t):
    """从 t 到结束的精简描述：结局、最终位置、引擎使用、最多 6 帧 (x, 高度, 倾角)。不含奖励。
    obs: [T+1, 8]（含结束后的观测），acts: [T]，end ∈ END。"""
    n = len(acts)
    idx = list(range(t, n, 40)) + [n]
    if len(idx) > 6:
        idx = [idx[int(round(k))] for k in np.linspace(0, len(idx) - 1, 6)]
    x, y = obs[n][0], obs[n][1]
    a = np.asarray(acts[t:])
    snaps = "; ".join(f"+{s - t}: ({obs[s][0]:+.2f}, {obs[s][1]:.2f}, {np.degrees(obs[s][4]):+.0f} deg)" for s in idx)
    return (f"Lasts {n - t} more steps, then: {END[end]}, final x={x:+.2f}, height={y:.2f}. "
            f"Main engine {int((a == 2).sum())} steps, side engines {int(((a == 1) | (a == 3)).sum())} steps. "
            f"Snapshots (x, height, tilt): {snaps}.")


def rank_prompt(sums):
    body = "\n\n".join(f"Description {cj.LETTERS[k]}:\n{s}" for k, s in enumerate(sums))
    return (f"{SPEC_LEFT}\n\n{body}\n\nRank all {len(sums)} descriptions from best to worst. Put descriptions that are "
            "about equally good in the same tier. Every letter must appear exactly once.",
            {"type": "OBJECT", "properties": {"tiers": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "STRING"}}}},
             "required": ["tiers"]})


def tiers_to_rank(out, n):
    """裁判输出 → 每条描述的名次（0 最好）；无效输出返回 None。"""
    if not out or "tiers" not in out:
        return None
    rank = {}
    for r, tier in enumerate(out["tiers"]):
        for z in tier:
            z = str(z).replace("Description", "").strip().upper()
            if len(z) != 1 or z not in cj.LETTERS[:n] or z in rank:
                return None
            rank[z] = r
    if len(rank) != n:
        return None
    return [rank[cj.LETTERS[k]] for k in range(n)]
