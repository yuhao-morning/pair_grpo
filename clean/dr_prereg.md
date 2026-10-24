# 双重稳健（DR）优势实验：事先约定（写于正式运行前，2026-10-24）

## 估计量

只用比较 c ∈ {+1,0,−1}；Q 由 Bradley-Terry 从比较学到。

- V(s) = mean_m Q(s, a'_m)，a'_m ~ π(·|s)，M=8。
- m_i(a) = mean_j [2σ(Q(s_i, a) − V(s_j)) − 1]：模型对"状态 i 取动作 a 时与 j 比较"的预测。
- o_i = mean_j c_ij：真实比较。
- b_i = mean_m m_i(a'_m)：状态基线。

方法：
- `pcrit_dr`：采到的动作优势为 o_i − m_i(a_i)；另采的 M 个动作优势为 m_i(a'_m) − b_i，权重 1/M。期望梯度等于 ∇ 对 (o_i − b_i) 的梯度，所以无偏。
- `pcrit_resq`（消融）：只用 o_i − b_i，没有模型项。它和 pcrit_dr 的差别只有模型项。
- `pcrit_res`（对照）：V 由单独的 BT 网络学；复用 `results/pcrit` 的种子 1–10。已验证代码改动后逐行一致。

字面形式"观测 − 模型预测 + 模型优势"若只作用于采到的那一个动作，会化简回 o − b，没有作用。所以模型项必须在多个采样动作上取期望（参考 Q-Prop 的思路）。

## 设置

- 环境：Hopper-v4、HalfCheetah-v4，各 3M 步。
- 种子：1–10。
- 超参：全部默认。pcrit_dr 的 minibatch 放大到 (1+M) 倍，使每个 epoch 的梯度步数与其他方法相同。
- 不调参；之前的 50k 步试跑只用来检查程序能跑。

## 指标与检验

- 指标：final（最后 10% 步数）和 AUC。
- 主比较：pcrit_dr vs pcrit_res。
- 次比较：pcrit_dr vs pcrit_resq（模型项本身的作用）。
- 检验：Welch t 检验和 Mann-Whitney 检验。
- 事先写下的判断标准：
  - 只有当 pcrit_dr 在至少一个环境上显著好于 pcrit_res（p<0.05），且在另一个环境上不显著更差时，才算"DR 有效"。
  - 如果 pcrit_resq 也同样好，那么提升来自 critic 类型，而不是 DR。
