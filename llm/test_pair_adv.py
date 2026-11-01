"""pair_adv.py 的单元测试。用法：python -I llm/test_pair_adv.py（不依赖 pytest）。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_adv as pa  # noqa: E402


def close(a, b, tol=1e-9):
    assert np.allclose(a, b, atol=tol), f"\n{np.asarray(a)}\n!=\n{np.asarray(b)}"


def test_compare_norm_basic():
    g = pa.group_indices(np.zeros(3))
    close(pa.compare_norm([3, 1, 2], g), [1, -1, 0])
    close(pa.compare_norm([1, 1, 0], g), [0.5, 0.5, -1])        # 平局记 0


def test_singletons_and_separate_groups():
    uid = np.array(["a", "a", "b"])
    close(pa.compare_norm([5, 1, 7], pa.group_indices(uid)), [1, -1, 0])


def test_zero_sum_and_monotone_invariance():
    rng = np.random.default_rng(0)
    x, uid = rng.normal(size=200), rng.integers(0, 7, 200)
    g = pa.group_indices(uid)
    A = pa.compare_norm(x, g)
    for idx in g.values():
        assert abs(A[idx].sum()) < 1e-9                              # 组内反对称 → 和为 0
    close(A, pa.compare_norm(np.exp(3 * x) - 5, g))                  # 严格单调变换不变


def test_binary_reward_proportional_to_centering():
    # 二值回报时，比较与"减均值"在组内只差一个常数倍：这就是 ALFWorld 检验不出差别的原因（design.md 第 2 节）
    rng = np.random.default_rng(1)
    x = 10.0 * rng.integers(0, 2, 8)
    x[0], x[1] = 0.0, 10.0
    A = pa.compare_norm(x, pa.group_indices(np.zeros(8)))
    close(A, (x - x.mean()) * 8 / (10 * 7))


def test_pair_grpo_and_pair_gigpo_by_hand():
    # 两条轨迹：τ1 三步（s0, s1, s1'），τ2 两步（s0, s1）。锚组：{τ1 第 1 步, τ2 第 1 步}、{τ1 第 2 步, τ2 第 2 步}、{τ1 第 3 步}
    uid = np.array(["g"] * 5)
    obs = np.array(["s0", "s1", "s2", "s0", "s1"])
    R = np.array([10.0, 10.0, 9.9, 0.0, 0.0])                       # τ1 成功，第 3 步无效动作扣 0.1
    G = np.array([9.0, 9.5, 9.9, 0.0, -0.1])
    A_ep = np.array([(0 + 1 + 1 + 1) / 4, (0 + 1 + 1 + 1) / 4, (-1 - 1 + 1 + 1) / 4, (-1 - 1 - 1 + 0) / 4, (-1 - 1 - 1 + 0) / 4])
    close(pa.pair_grpo(R, uid), A_ep)
    A_st = np.array([1.0, 1.0, 0.0, -1.0, -1.0])
    close(pa.pair_gigpo(R, G, uid, obs), A_ep + A_st)
    close(pa.pair_gigpo(R, G, uid, obs, w=0.5), A_ep + 0.5 * A_st)


def test_turn_index_and_pair_step():
    traj = np.array(["x", "x", "y", "x", "y"])                      # 交错出现，组内顺序保持时间先后
    t = pa.turn_index(traj)
    close(t, [0, 1, 0, 2, 1])
    uid = np.array(["g"] * 5)
    G = np.array([1.0, 2.0, 3.0, 5.0, 0.0])
    close(pa.pair_step(G, uid, t), [-1, 1, 1, 0, -1])                # x 的第 3 步时只剩它自己 → 0


def test_pcrit_residual():
    rng = np.random.default_rng(2)
    n = 60
    uid, t, G = rng.integers(0, 3, n), rng.integers(0, 4, n), rng.normal(size=n)
    A0, (ri, rj, c) = pa.pcrit_residual(G, uid, t, np.zeros(n))
    close(A0, pa.pair_step(G, uid, t))                               # 常数 critic → 退化为 pair_step
    f = rng.normal(size=n)
    A, _ = pa.pcrit_residual(G, uid, t, f)
    for idx in pa.group_indices(uid, t).values():
        assert abs(A[idx].sum()) < 1e-9                              # 残差反对称 → 组内和为 0
    assert len(c) == sum(len(v) * (len(v) - 1) // 2 for v in pa.group_indices(uid, t).values())
    # 手算一个两元素组
    A2, _ = pa.pcrit_residual([1.0, 0.0], ["g", "g"], [0, 0], [0.3, -0.2])
    r = 1 - (2 / (1 + np.exp(-0.5)) - 1)
    close(A2, [r, -r])


def test_win_rates():
    close(pa.win_rates([0, 1, 1, 2]), [1.0, 0.5, 0.5, 0.0])
    close(pa.win_rates([0, 0]), [0.5, 0.5])
    close(pa.win_rates([2, 0, 1]), [0.0, 1.0, 0.5])


def test_batch_scale():
    A = np.array([0.5, -1.0, 0.25, 0.0])
    S = pa.batch_scale(A)
    assert abs(S.std() - 1) < 1e-6
    close(S * A.std(), A, 1e-6)                                      # 只缩放
    close(pa.batch_scale(np.zeros(3)), np.zeros(3))


def test_critic_learns_bt_order():
    # 状态价值 v*(s) = w·s；只给比较标签，critic 应学会按 v* 排序
    rng = np.random.default_rng(3); torch.manual_seed(0)
    X = rng.normal(size=(800, 16)); w = rng.normal(size=16)
    v = X @ w
    ri, rj = rng.integers(0, 800, 20000), rng.integers(0, 800, 20000)
    c = np.sign(v[ri] - v[rj] + rng.logistic(size=20000))
    norm = pa.RunningNorm(16); norm.update(X)
    critic = pa.PairCritic(16)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for _ in range(5):
        pa.train_critic(critic, opt, norm(X), (ri, rj, c), rng)
    with torch.no_grad():
        f = critic(torch.as_tensor(norm(X), dtype=torch.float32)).numpy()
    acc = np.mean(np.sign(f[ri] - f[rj]) == np.sign(v[ri] - v[rj]))
    assert acc > 0.9, acc


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("通过", name)
    print(f"全部 {len(tests)} 个测试通过")
