"""Single-step K-armed bandit: compare advantage estimators used inside a GRPO-style update.

adv variants (group of G samples from current softmax policy):
  grpo   : (R - mean) / std
  linear : R - mean
  pair   : mean_{j!=i} sign(R_i - R_j)          (the user's method)
  pairflip: pair, but each comparison label flipped w.p. eps
  tanh   : mean_{j!=i} tanh((R_i - R_j)/tau)
"""
import numpy as np

K, G, STEPS, LR, SEEDS = 5, 8, 3000, 0.1, 20


def make_env(name, rng):
    means = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    if name == "gauss":           # well-behaved
        return lambda a: means[a] + 0.5 * rng.standard_normal(a.shape), means
    if name == "heavy":           # heavy-tailed noise, same ordering of means
        return lambda a: means[a] + 0.5 * rng.standard_t(1.5, a.shape), means
    if name == "skew":            # arm1 wins 90% of comparisons vs arm0 but has worse mean
        def f(a):
            r = means[a] + 0.3 * rng.standard_normal(a.shape)
            m1 = a == 1
            r[m1] = np.where(rng.random(m1.sum()) < 0.9, 1.3, -6.0)
            return r
        true = means.copy(); true[1] = 0.9 * 1.3 - 0.1 * 6.0
        return f, true
    if name == "scale":           # rewards on wildly different scale + small gaps
        return lambda a: 1000 * (means[a] + 0.5 * rng.standard_normal(a.shape)), 1000 * means
    raise ValueError(name)


def adv(kind, r, rng, eps=0.3, tau=0.5):
    d = r[:, None] - r[None, :]
    if kind == "grpo":
        return (r - r.mean()) / (r.std() + 1e-8)
    if kind == "linear":
        return r - r.mean()
    s = np.sign(d)
    if kind == "pairflip":
        flip = np.triu(rng.random(d.shape) < eps, 1)
        flip = flip | flip.T
        s = np.where(flip, -s, s)
    if kind == "tanh":
        s = np.tanh(d / (tau * (r.std() + 1e-8)))
    return s.sum(1) / (len(r) - 1)


def run(env_name, kind, seed):
    rng = np.random.default_rng(seed)
    env, true = make_env(env_name, rng)
    logits = np.zeros(K)
    for _ in range(STEPS):
        p = np.exp(logits - logits.max()); p /= p.sum()
        a = rng.choice(K, G, p=p)
        A = adv(kind, env(a), rng)
        g = np.zeros(K)
        for ai, Ai in zip(a, A):      # sum_i A_i * grad log pi(a_i)
            g -= Ai * p
            g[ai] += Ai
        logits += LR * g / G
    p = np.exp(logits - logits.max()); p /= p.sum()
    return p @ true, p.argmax(), true.max()


if __name__ == "__main__":
    kinds = ["grpo", "linear", "pair", "pairflip", "tanh"]
    for env_name in ["gauss", "heavy", "skew", "scale"]:
        print(f"\n== {env_name} ==")
        for kind in kinds:
            res = [run(env_name, kind, s) for s in range(SEEDS)]
            v = np.array([x[0] for x in res]); best = np.array([x[1] for x in res])
            opt = res[0][2]
            arms = np.bincount(best, minlength=K)
            print(f"{kind:9s} E[R]={v.mean():9.3f} ±{1.96*v.std()/np.sqrt(SEEDS):.3f} "
                  f"(opt {opt:.3f})  final-argmax counts {arms.tolist()}")
