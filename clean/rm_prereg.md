# 奖励模型基线与 SPO：事先约定（写于正式运行前，2026-10-26）

## 问题

审稿人会问："为什么不用同样的比较学一个奖励模型，再跑 PPO？"（Christiano 2017 / PEBBLE 路线），以及"和 SPO（轨迹级胜率）比怎么样？"

## 方法（实现见 `phase_b.py`）

### `rm_ppo`

- **比较数据：** 和 pair_step 完全相同（同组、同一 t 的 reward-to-go 比较，每步全部配对）。
- **奖励模型：** 用 Bradley-Terry 学 r̂(s, a)，P(i≻j) = σ(Ĝ_t^i − Ĝ_t^j)，Ĝ 是 r̂ 的折扣累加（γ = 0.99，与 oracle 一致）。平局的目标值为 0.5。
- **网络与训练：**
  - r̂ 网络为 MLP 128-128。
  - 缓冲区保留最近 5 次更新的比较，每次更新在缓冲区上训练 2 轮，lr 1e-3。
- **PPO：** 用 r̂（按 batch std 缩放，不平移）跑 PPO + critic + GAE，其余超参与 ppo 相同。
- **复用标签：** 奖励模型复用旧标签，是这条路线本来就有的优势，我们不去掉。

### `spo`

按 Swamy et al. 2024 的 Algorithm 2：
- 每条轨迹与最近 B = 64 条轨迹的队列比较总回报，得分 = mean 𝒫 ∈ [−1, 1]，平局记 0。
- 得分平均分摊到该轨迹的每一步；比完把它推入队列（FIFO）。
- 然后跑 PPO + critic + GAE。

与原文的差异：
- 原文在连续控制上用 SAC，这里统一用 PPO，以便与其他方法可比。
- 原文是固定时长 H，这里按每条轨迹自己的长度分摊。

## 设置

- 环境：HalfCheetah-v4（3M 步）、Hopper-v4（3M 步）、LunarLander-v3（2M 步）、Acrobot-v1（0.5M 步）。
- 种子：1–10，结果在 `results/rm`。
- 对照复用已有的种子 1–10：
  - HalfCheetah、Hopper：`results/pcrit`（pair_step、pcrit_res、ppo）；
  - LunarLander、Acrobot：`results/disc`（无噪声的 pair_step、pcrit_res、ppo）。
  - 已验证 ppo、pcrit_res、pair_step 在代码改动后逐行一致。
- 全部默认超参，不调参。种子 100 只用于冒烟测试。

## 检验（final = 最后 10% 步数的平均回报；Welch t 检验；Holm 校正）

- **A 族（8 个）：** rm_ppo vs pair_step、rm_ppo vs pcrit_res，每个环境各 1 次。
- **B 族（4 个）：** spo vs pair_step，每个环境 1 次。
- 辅助：AUC、Mann-Whitney；记录奖励模型在新批次比较上的准确率，并对照它与真实回报的走势，看有没有奖励被钻空子。

## 判断标准（事先写下）

- **"奖励模型路线占优"：** rm_ppo 在至少 2 个环境上**同时**显著好于 pair_step 和 pcrit_res，且没有任何环境显著更差。这种情况下，论文必须改定位：pair 的价值不在性能，而在简单、稳定、不需要奖励模型等方面。
- **"相当"：** 至少 3 个环境上 rm_ppo 与两者都没有显著差别。
- **"pair 路线占优"：** pair_step 或 pcrit_res 在至少 2 个环境上显著好于 rm_ppo。
- 其他情况记为"混合"，逐环境报告。

## 事先写下的预测

- **HalfCheetah：** rm_ppo 有稠密的学习奖励加 critic，大概率显著好于 pair_step，可能接近或超过 pcrit_res。这是对本文最不利的情形。
- **Hopper：** 学到的奖励可能和存活奖励的结构对不上，rm_ppo 有被钻空子的风险，可能不如 pcrit_res。
- **LunarLander、Acrobot：** 与 pair 方法相近。
- **SPO：** 轨迹级信用分配，在 HalfCheetah 上会很差（之前轨迹级方法约为 0）；在 Acrobot 上不错；在 LunarLander 上中等。
