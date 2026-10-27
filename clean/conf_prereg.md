# 5 种子结论的 10 种子确认：事先约定（写于正式运行前，2026-10-27）

把大纲里标为 [多种子，5] 的三条结论补成 10 个种子，并事先写好判据。合并新旧种子之前，已验证 grpo_step、sweep 的 ppo、long 的 pair_step 在当前代码下与旧结果逐行一致。

## 共同定义

- **R0：** 该环境下第一条记录回报的平均（近似随机策略水平），按该比较里所有运行合并计算。
- **归一化差值：** (A − B) / (参照 − R0)。
- **等价 / 非劣界值：** 都取参照方法相对 R0 进步量的 **10%**。

---

## C1：同等信息下 pair_step ≈ grpo_step

- **数据：**
  - pair_step 复用种子 1–10：HalfCheetah 3M、Hopper 3M 来自 `results/pcrit`；LunarLander 2M、Acrobot 0.5M 来自 `results/disc`。
  - grpo_step 新跑种子 1–10，同样的步数，结果在 `results/conf`。共 40 个运行。
- **主检验：** 每个环境比较 final 的差别（Welch t 检验），4 个环境一起做 Holm 校正。
- **等价检验：** TOST，界值为 ±10% × (grpo_step − R0)，单侧 α = 0.05。
- **判断：**
  - 某环境上 TOST 通过 → 该环境写"等价"。
  - 差别显著 → 写方向。
  - 两者都不满足 → 写"不确定"。
  - **只有至少 3 个环境"等价"且没有显著差别时，论文才能写"pair 与 GRPO 打平"。**
- **预测：** LunarLander、Acrobot 等价；HalfCheetah 可能 grpo 更好（之前 5 个种子在 12M 时 3455 vs 1976）；Hopper 不确定。

---

## C2：训练足够长时，pair_step 追平 PPO（Hopper 5M、LunarLander 3M）

- **数据：** `results/long` 已有种子 0–4；新跑 ppo 和 pair_step 的种子 5–9，共 20 个运行。合计每个方法 10 个种子。
- **检验：** 非劣检验，H0 为 pair_step ≤ ppo − 10% × (ppo − R0)，单侧 Welch，α = 0.05，2 个环境一起做 Holm 校正。辅助报告双侧差别检验。
- **判断：**
  - 两个环境都通过 → 论文写"非劣于 PPO"。
  - 只有一个通过 → 只写那一个。
- **预测：** LunarLander 通过（pair 更高）；Hopper 可能因方差大而不通过。

---

## C3：pair_step 对超参最不敏感（LunarLander 1M、Acrobot 0.3M）

- **数据：** `results/sweep` 已有种子 100–102，网格为 lr ∈ {1e-4, 3e-4, 1e-3} × epochs ∈ {4, 10, 20}，方法为 ppo、grpo_step、pair_step。新跑种子 103–109，每个配置合计 10 个种子，共 378 个运行。
- **指标：**
  - 每个方法、每个配置，求 10 个种子的平均 AUC。
  - **最差配置保留率** = (最差配置 AUC − R0) / (最好配置 AUC − R0)。
  - 辅助：配置间的极差和标准差（同样归一化）。
- **检验：** 在每个配置内对种子做 bootstrap 重采样（2000 次），得到 pair_step 与 ppo、pair_step 与 grpo_step 两个"最差配置保留率"差值的 95% 区间。
- **判断：**
  - 两个环境上，pair_step 对 ppo 和对 grpo_step 的差值区间都在 0 以上 → 写"最不敏感"。
  - 只有部分成立 → 如实写成立的那部分。
- **预测：** 对 ppo 成立，因为 ppo 在 LunarLander 小 lr 下很差（试跑 AUC 为 −1.5）。对 grpo_step 不一定成立。
