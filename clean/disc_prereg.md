# pcrit_res 在离散任务和噪声标签下：事先约定（写于正式运行前，2026-10-24）

## 设置

- 环境：LunarLander-v3（2M 步）、Acrobot-v1（0.5M 步）。步数与 Phase B 相同。
- 方法：
  - pcrit_res、pair_step：标签分别为无噪声、翻转 10%、翻转 25%。
  - ppo：标量参照，只跑无噪声。
- 种子 1–10，结果目录 `results/disc`。全部默认超参，不调参。
- 噪声模型：每个比较以概率 ε 独立翻转符号。平局（c=0）翻转后仍是 0。pcrit_res 的 critic 和优势用的都是翻转后的同一份标签。

## 主检验

六个检验，用 Holm 法在 α=0.05 下做多重比较校正：2 个环境 × 3 个噪声水平，每个检验都是 pcrit_res vs pair_step 的 final（最后 10% 步数平均回报），Welch t 检验。

AUC 和 Mann-Whitney 只作辅助报告。

## 判断标准（事先写下）

- **"不只在 MuJoCo 上有效"**：无噪声时，pcrit_res 在至少一个离散环境上经 Holm 校正后显著好于 pair_step，且在另一个上不显著更差。
- **"对噪声标签稳健"**：在 10% 和 25% 翻转下，pcrit_res 在两个环境上都不显著差于 pair_step（Holm 校正后）。
- **"噪声下更好"**：在某个噪声水平上 pcrit_res 经校正后显著好于 pair_step。
- 如果无噪声时两者没有显著差别，结论写成"离散任务上无增益"，不写成"有效"。

另外报告：Acrobot 上 final < −400 的"失败种子"数。之前 25% 翻转下 pair_step 有种子卡在 −500。
