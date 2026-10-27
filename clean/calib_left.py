"""LunarLeft 定向校准：裁判能否在"一条落左、一条落右（都没坠毁）"时选出落左的那条。判据见 calib_left_prereg.md。
用法：python calib_left.py data   # 生成数据，不调用 API
      python calib_left.py judge  # 调用裁判（上限 $2）
      python calib_left.py analyze"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_judge as cj  # noqa: E402
import judge_left as jl  # noqa: E402
import lunar_left  # noqa: E402,F401
from calib_data import load_policy  # noqa: E402
from phase_b import discounted_to_go  # noqa: E402

OUT = "results/calib_left"
STAGES, GROUPS, L, PER_STAGE, K = [250000, 500000, 1000000], 12, 20, 50, 8


def rollout(actor, norm, seed):
    trajs = []
    for _ in range(K):
        env = gym.make("LunarLeft-v0")
        o, _ = env.reset(seed=seed)
        obs, acts, rews = [o.copy()], [], []
        while True:
            with torch.no_grad():
                a = int(actor.dist(torch.as_tensor(norm(o[None]), dtype=torch.float32)).sample().item())
            o, r, term, trunc, info = env.step(a)
            obs.append(o.copy()); acts.append(a); rews.append(r)
            if term or trunc:
                end = "crashed" if info["crashed"] else ("landed" if info["landed"] else "timeout")
                break
        trajs.append(dict(obs=np.array(obs), acts=np.array(acts), rews=np.array(rews), end=end))
    return trajs


def data():
    rng = np.random.default_rng(0); torch.manual_seed(0)
    os.makedirs(OUT, exist_ok=True)
    items = []
    for stage in STAGES:
        actor, norm = load_policy(stage)
        cand = []
        for _ in range(GROUPS):
            seed = int(rng.integers(2 ** 31))
            tr = rollout(actor, norm, seed)
            T = max(len(x["acts"]) for x in tr)
            for t in range(0, T, L):
                alive = [i for i, x in enumerate(tr) if len(x["acts"]) > t]
                if len(alive) >= 2:
                    cand.append((t, tr, alive))
        for c in rng.choice(len(cand), min(PER_STAGE, len(cand)), replace=False):
            t, tr, alive = cand[c]
            items.append(dict(item=len(items), stage=stage, t=int(t), suffixes=[
                dict(summary=jl.summary_short(tr[i]["obs"], tr[i]["acts"], tr[i]["end"], t), end=tr[i]["end"],
                     x_end=float(tr[i]["obs"][-1][0]), G=float(discounted_to_go(tr[i]["rews"], 0.99)[t])) for i in alive]))
    with open(f"{OUT}/items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    lr = sum(1 for it in items for a in it["suffixes"] for b in it["suffixes"]
             if a["end"] != "crashed" and b["end"] != "crashed" and a["x_end"] < 0 < b["x_end"])
    toks = np.mean([len(s["summary"]) for it in items for s in it["suffixes"]]) / 4
    print(f"{len(items)} 个检查点；左右对立（都未坠毁）的配对 {lr} 个；摘要约 {toks:.0f} token/条")


def judge(cap=2.0):
    cj.configure("gemini-3.5-flash-lite", OUT)
    items = [json.loads(l) for l in open(f"{OUT}/items.jsonl")]
    J = cj.Judge(cap)
    rng = random.Random(0)
    jobs = []
    for it in items:
        order = list(range(len(it["suffixes"]))); rng.shuffle(order)
        for o in (order, order[::-1]):
            jobs.append((it["item"], o, *jl.rank_prompt([it["suffixes"][k]["summary"] for k in o])))
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda j: (j[0], j[1], J(j[2], j[3])), jobs))
    json.dump([[i, o, out] for i, o, out in outs], open(f"{OUT}/judgments.json", "w"))
    print(f"调用 {J.calls}，累计 ${J.spent:.3f}")


def analyze():
    items = {it["item"]: it for it in (json.loads(l) for l in open(f"{OUT}/items.jsonl"))}
    J = json.load(open(f"{OUT}/judgments.json"))
    lr, crash, allp, invalid = [], [], [], 0
    for i, order, out in J:
        rank = jl.tiers_to_rank(out, len(order))
        if rank is None:
            invalid += 1; continue
        S = items[i]["suffixes"]
        r_of = {order[k]: rank[k] for k in range(len(order))}
        for a in range(len(S)):
            for b in range(len(S)):
                if a >= b:
                    continue
                j = np.sign(r_of[b] - r_of[a])        # +1：a 排在前面
                g = np.sign(S[a]["G"] - S[b]["G"])
                eff = 0.5 if j == 0 else float(j == g)
                if g != 0:
                    allp.append(eff)
                ea, eb = S[a], S[b]
                if ea["end"] != "crashed" and eb["end"] != "crashed" and np.sign(ea["x_end"]) != np.sign(eb["x_end"]):
                    want = 1 if ea["x_end"] < 0 else -1
                    lr.append(0.5 if j == 0 else float(j == want))
                if (ea["end"] == "crashed") != (eb["end"] == "crashed"):
                    want = -1 if ea["end"] == "crashed" else 1
                    crash.append(0.5 if j == 0 else float(j == want))
    print(f"无效输出 {invalid} 次")
    print(f"主判据——左右对立（都未坠毁）选左的有效准确率：{np.mean(lr):.3f}（n={len(lr)}）")
    print(f"坠毁 vs 未坠毁 选未坠毁：{np.mean(crash):.3f}（n={len(crash)}）")
    print(f"对金标准 G 的总体有效准确率：{np.mean(allp):.3f}（n={len(allp)}）")
    print("判定：" + ("通过（≥0.90），进入试点" if np.mean(lr) >= 0.90 else "未通过，不进入试点"))


if __name__ == "__main__":
    {"data": data, "judge": judge, "analyze": analyze}[sys.argv[1]]()
