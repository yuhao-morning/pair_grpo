"""LLM 裁判校准的数据：用 LunarLander 不同训练阶段的策略采同起点的轨迹组，
在检查点 t 生成"从 t 起"的文字摘要，并记下金标准 reward-to-go。不调用任何 API。

输出 results/calib/items.jsonl，每行一个检查点：
  {"item", "stage", "group_seed", "t", "suffixes": [{"traj", "summary", "G"}...]}
"""
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_b import Actor, RunningNorm, discounted_to_go  # noqa: E402

GAMMA, K = 0.99, 8
STAGES = [0, 100000, 250000, 500000, 1000000]
GROUPS_PER_STAGE, ITEMS_PER_STAGE, CKPT_EVERY = 16, 60, 50
CKPT_DIR = "results/calib_policy/LunarLander-v3/ppo"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/calib/items.jsonl"   # G1：不折扣的 reward-to-go（探索性分析用）


def load_policy(stage):
    ck = torch.load(f"{CKPT_DIR}/ckpt_seed200_{stage}.pt", weights_only=False)
    actor = Actor(8, 4, False)
    actor.load_state_dict(ck["actor"])
    norm = RunningNorm(8)
    norm.n, norm.mean, norm.m2 = ck["norm"]
    return actor, norm


def rollout_group(actor, norm, seed, rng):
    trajs = []
    for k in range(K):
        env = gym.make("LunarLander-v3")
        o, _ = env.reset(seed=seed)
        obs, acts, rews = [], [], []
        while True:
            with torch.no_grad():
                d = actor.dist(torch.as_tensor(norm(o[None]), dtype=torch.float32))
                a = int(d.sample().item())
            obs.append(o.copy()); acts.append(a)
            o, r, term, trunc, _ = env.step(a)
            rews.append(r)
            if term or trunc:
                u = env.unwrapped
                if trunc:
                    end = "the time limit was reached"   # v2：中性措辞，状态交给快照（v1 写成 "still flying"，与快照矛盾）
                elif getattr(u, "game_over", False) or abs(o[0]) >= 1.0:
                    end = "crashed or flew out of bounds"
                else:
                    end = "landed and came to rest"
                break
        obs.append(o.copy())
        trajs.append(dict(obs=np.array(obs), acts=np.array(acts), rews=np.array(rews), end=end))
    return trajs


def summary(tr, t):
    """从 t 到结束的描述：每 25 步一帧原始物理量（最多 12 帧，均匀抽取）+ 引擎使用 + 结局。不含奖励。"""
    n = len(tr["acts"])
    idx = list(range(t, n, 25)) + [n]
    if len(idx) > 12:
        idx = [idx[int(round(k))] for k in np.linspace(0, len(idx) - 1, 12)]
    lines = []
    for s in idx:
        x, y, vx, vy, ang, _, l1, l2 = tr["obs"][s]
        legs = {(0, 0): "none", (1, 0): "left", (0, 1): "right", (1, 1): "both"}[(int(l1 > 0.5), int(l2 > 0.5))]
        lines.append(f"  +{s - t} steps: x={x:+.2f}, height={y:.2f}, vx={vx:+.2f}, vy={vy:+.2f}, "
                     f"tilt={np.degrees(ang):+.0f} deg, legs touching: {legs}")
    a = tr["acts"][t:]
    return (f"Lasts {n - t} more steps, then: {tr['end']}.\n"
            f"Main engine fired on {int((a == 2).sum())} steps, side engines on {int(((a == 1) | (a == 3)).sum())} steps.\n"
            "Snapshots:\n" + "\n".join(lines))


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    items = []
    for stage in STAGES:
        actor, norm = load_policy(stage)
        cand = []
        for g in range(GROUPS_PER_STAGE):
            seed = int(rng.integers(2 ** 31))
            trajs = rollout_group(actor, norm, seed, rng)
            lens = np.array([len(tr["acts"]) for tr in trajs])
            for t in range(0, lens.max(), CKPT_EVERY):
                alive = np.flatnonzero(lens > t)
                if len(alive) >= 2:
                    cand.append((seed, t, trajs, alive))
        pick = rng.choice(len(cand), min(ITEMS_PER_STAGE, len(cand)), replace=False)
        for c in pick:
            seed, t, trajs, alive = cand[c]
            sufs = [dict(traj=int(i), summary=summary(trajs[i], t),
                         G=float(discounted_to_go(trajs[i]["rews"], GAMMA)[t]),
                         G1=float(trajs[i]["rews"][t:].sum())) for i in alive]
            items.append(dict(item=len(items), stage=stage, group_seed=seed, t=int(t), suffixes=sufs))
        print(f"stage {stage}: 候选检查点 {len(cand)}，抽取 {len(pick)}")
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    n_suf = sum(len(it["suffixes"]) for it in items)
    n_pair = sum(len(it["suffixes"]) * (len(it["suffixes"]) - 1) // 2 for it in items)
    print(f"共 {len(items)} 个检查点，{n_suf} 条后缀摘要，{n_pair} 个配对")


if __name__ == "__main__":
    main()
