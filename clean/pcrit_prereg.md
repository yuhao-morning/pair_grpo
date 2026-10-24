# 比较训练 critic 实验：事先约定（写于正式运行前，2026-10-24）

## 方法

所有比较方法只拿到组内同一 t 的 reward-to-go 比较结果 c ∈ {+1, 0, −1}。

- `pair_step`：mean_j c_ij（无 critic）。
- `pcrit_res`：用 Bradley-Terry 学 V(s)。优势 = mean_j [c_ij − (2σ(V_i − V_j) − 1)]，即观测到的比较减去仅凭状态预测的比较。
- `pcrit_q`：用 Bradley-Terry 学 Q(s,a)。优势 = Q(s,a) − mean_{m=1..8} Q(s, a'_m)，其中 a'_m ~ π(·|s)。
- 两个 critic 都先用旧参数算优势，再在本批比较上训练：4 个 epoch，每批最多 4 万个比较对，lr 1e-3。

参照：
- `ppo`：标量奖励 + GAE critic。
- `gigpo`：标量奖励。优势 = 轨迹级 z-score + 1.0 × 完全相同状态组内的 z-score。

## 设置

- 环境：Hopper-v4、HalfCheetah-v4，各 3M 步。
- 种子：1–10。种子 0 和 100–101 用于试跑和调参，不进入评估。
- 超参：全部用默认值。critic 的设置在调参种子上没有发现稳定更好的选项，因此不调。

## 指标与检验

- 指标：final 为最后 10% 步数内的平均回报；AUC 为 50 段平均。
- 主比较：
  1. pcrit_q vs pair_step：critic 是否有帮助。
  2. pcrit_q vs ppo：只用比较能否追平用标量的 PPO。
  3. pcrit_res vs pair_step。
- 检验：Welch t 检验和 Mann-Whitney 检验，报告 p 值，不做多重比较校正（共 6 个主检验，读结果时要记住这一点）。
