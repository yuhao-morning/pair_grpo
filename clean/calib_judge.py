"""LLM 裁判校准：对 results/calib/items.jsonl 做逐条打分、直接两两、组内排名三种判断。

- 密钥只从 ~/.config/gemini_key 读取，不打印、不写入缓存或日志。
- 每个请求/响应缓存在 results/calib/cache.jsonl（以 模型+提示 的哈希为键），重跑不重复花钱。
- 按付费标准价累计花费，超过 --cap 美元立即停止。
用法: python calib_judge.py --limit 5      # 试跑前 5 个检查点
      python calib_judge.py                # 全部
"""
import argparse
import hashlib
import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# 默认值对应 v1（results/calib）；v2 通过命令行指定模型和目录
MODEL = "gemini-3.5-flash-lite"
PRICES = {"gemini-3.5-flash-lite": (0.30, 2.50),   # 美元 / 百万 token，付费标准价（含思考 token），用来设上限
          "gemini-3.5-flash": (1.50, 7.50)}        # 定价页未列出 3.5 Flash，按 3.6 Flash 的价格保守计
OUTDIR = "results/calib"


def configure(model, outdir):
    global MODEL, PRICE_IN, PRICE_OUT, URL, ITEMS, CACHE, OUTDIR
    MODEL, OUTDIR = model, outdir
    PRICE_IN, PRICE_OUT = PRICES[model]
    URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    ITEMS, CACHE = f"{outdir}/items.jsonl", f"{outdir}/cache_{model}.jsonl" if outdir != "results/calib" else f"{outdir}/cache.jsonl"


configure(MODEL, OUTDIR)

SPEC = ("You are judging the behaviour of a lunar lander in a simulator.\n"
        "Goal: land gently and upright on the landing pad, which is on flat ground between two flags at "
        "horizontal position x close to 0 (height 0 is the ground). Use little engine fuel. Do not crash or fly "
        "out of bounds. Coming to rest on the pad is the ideal outcome.\n"
        "Each description covers what happened from one moment until the end of the episode. "
        "Judge only how well that stretch achieves the goal.")
LETTERS = "ABCDEFGH"


def p_point(s):
    return (f"{SPEC}\n\nDescription:\n{s}\n\nRate how well this achieves the goal from 0 (very bad) to 10 (ideal).",
            {"type": "OBJECT", "properties": {"score": {"type": "INTEGER"}}, "required": ["score"]})


def p_pair(sa, sb):
    return (f"{SPEC}\n\nDescription A:\n{sa}\n\nDescription B:\n{sb}\n\n"
            "Which description better achieves the goal? Answer A, B, or SAME if they are about equally good.",
            {"type": "OBJECT", "properties": {"better": {"type": "STRING", "enum": ["A", "B", "SAME"]}},
             "required": ["better"]})


def p_list(sums):
    body = "\n\n".join(f"Description {LETTERS[k]}:\n{s}" for k, s in enumerate(sums))
    return (f"{SPEC}\n\n{body}\n\nRank all {len(sums)} descriptions from best to worst. Put descriptions that are "
            "about equally good in the same tier. Every letter must appear exactly once.",
            {"type": "OBJECT", "properties": {"tiers": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "STRING"}}}},
             "required": ["tiers"]})


class Judge:
    def __init__(self, cap):
        self.key = open(os.path.expanduser("~/.config/gemini_key")).read().strip()
        self.cap, self.lock = cap, threading.Lock()
        self.cache, self.spent, self.calls = {}, 0.0, 0
        if os.path.exists(CACHE):
            for line in open(CACHE):
                r = json.loads(line)
                self.cache[r["h"]] = r
                self.spent += r["cost"]
        self.f = open(CACHE, "a")

    def __call__(self, prompt, schema):
        h = hashlib.sha256((MODEL + prompt).encode()).hexdigest()
        if h in self.cache:
            return self.cache[h]["out"]
        with self.lock:
            if self.spent >= self.cap:
                raise RuntimeError(f"花费已到上限 ${self.cap}")
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema,
                                     "thinkingConfig": {"thinkingLevel": "minimal"}}}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
        for attempt in range(8):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=120))
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                    time.sleep(min(60, 2 ** attempt + random.random()))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException):
                # 网络层错误（含 RemoteDisconnected）一律退避重试
                if attempt == 7:
                    raise RuntimeError("网络错误，重试 8 次仍失败")
                time.sleep(min(60, 2 ** attempt + random.random()))
        u = r.get("usageMetadata", {})
        cost = (u.get("promptTokenCount", 0) * PRICE_IN
                + (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) * PRICE_OUT) / 1e6
        try:
            out = json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, json.JSONDecodeError):
            out = None
        rec = dict(h=h, out=out, usage=u, cost=cost)
        with self.lock:
            self.cache[h] = rec
            self.spent += cost
            self.calls += 1
            self.f.write(json.dumps(rec) + "\n"); self.f.flush()
        return out


def build_jobs(items, rng):
    """返回 (类型, 键, prompt, schema)。键用来在分析时找回结果。"""
    jobs = []
    for it in items:
        sums = [s["summary"] for s in it["suffixes"]]
        n = len(sums)
        for k in range(n):
            jobs.append(("point", (it["item"], k, 0), *p_point(sums[k])))
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        for a, b in rng.sample(pairs, min(3, len(pairs))):
            jobs.append(("pair", (it["item"], a, b), *p_pair(sums[a], sums[b])))
            jobs.append(("pair", (it["item"], b, a), *p_pair(sums[b], sums[a])))
        order = list(range(n)); rng.shuffle(order)
        for o in (order, order[::-1]):
            jobs.append(("list", (it["item"], tuple(o)), *p_list([sums[k] for k in o])))
    # 逐条打分的重复：随机 300 条，在提示末尾加一个空格使哈希不同、内容不变
    point_jobs = [j for j in jobs if j[0] == "point"]
    for j in rng.sample(point_jobs, min(300, len(point_jobs))):
        jobs.append(("point", (j[1][0], j[1][1], 1), j[2] + " ", j[3]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gemini-3.5-flash-lite", choices=sorted(PRICES))
    ap.add_argument("--outdir", default="results/calib")
    a = ap.parse_args()
    configure(a.model, a.outdir)
    items = [json.loads(l) for l in open(ITEMS)]
    jobs = build_jobs(items, random.Random(0))   # 先用全部检查点构造，保证 --limit 前后抽到的配对一致
    if a.limit:
        keep = {it["item"] for it in items[:a.limit]}
        jobs = [j for j in jobs if j[1][0] in keep]
    judge = Judge(a.cap)
    print(f"任务 {len(jobs)} 个；缓存已有 {len(judge.cache)}；累计花费 ${judge.spent:.3f}")
    results, t0 = {}, time.time()

    def run(j):
        try:
            return j, judge(j[2], j[3]), None
        except RuntimeError as e:
            return j, None, str(e)

    errors = []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (j, out, err) in enumerate(ex.map(run, jobs)):
            results[(j[0], j[1])] = out
            if err:
                errors.append(err)
                if "上限" in err or "HTTP 4" in err:
                    print("停止：", err)
                    break
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(jobs)}  新调用 {judge.calls}  累计 ${judge.spent:.3f}  {time.time() - t0:.0f}s", flush=True)
    jpath = f"{OUTDIR}/judgments.json" if OUTDIR == "results/calib" else f"{OUTDIR}/judgments_{MODEL}.json"
    if a.limit:   # 试跑写到单独的文件，避免覆盖全量结果
        jpath = jpath.replace(".json", f".limit{a.limit}.json")
    with open(jpath, "w") as f:
        json.dump([[t, list(k) if not isinstance(k, tuple) else [x if not isinstance(x, tuple) else list(x) for x in k], v]
                   for (t, k), v in results.items()], f)
    print(f"完成：新调用 {judge.calls}，累计花费 ${judge.spent:.3f}，错误 {len(errors)}，无法解析 {sum(v is None for v in results.values())}")


if __name__ == "__main__":
    main()
