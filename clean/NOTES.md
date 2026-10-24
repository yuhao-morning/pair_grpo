# Pair-GRPO：代码审查、方法修正与多种子实验

日期：2026-10-24。所有数字都是 5 个种子的 mean ± 95% CI（t 分布），除非另说明。复现命令见文末。

## 0. 一句话结论

1. **只用比较、扔掉幅度，代价几乎为零**：在同样的分组/粒度下，pair 和 GRPO 打平或更好（LunarLander 263±7 vs 242±34，Hopper 1597 vs 1287），而且是对超参数最不敏感的方法。这是方法最站得住的卖点。
2. **原代码里的 "pair" 其实是 "PPO + 对 critic 优势做 rank 变换"**，用的还是标量奖励和 critic，测不到你的主张；在原流水线里，它和对照组统计上没有差别，原 PPO 反而在 3/4 个环境最好。之前"大多超过或类似 PPO"的结论来自单种子，HalfCheetah 上种子间 CI 有 ±1000。
3. **pair 优化的不是期望回报，而是"对当前策略的胜率"**（Borda 分数）。随机奖励下两者可以不同（bandit 反例）；论文里"梯度是 GRPO 梯度的正倍数"的定理一般不成立。
4. **多步问题里，按时间步比较是对的方向**（HalfCheetah 2M 时轨迹级 ≈0，按步 ≈1100）。但同一 t 的不同轨迹处在不同状态，比较混入了状态价值。
5. **训练够长时，大部分差距会消失**：
   - Hopper 5M：pair_step 2172 ≈ PPO 2229 ≈ GRPO 2237。2M 时落后只是学得慢。
   - LunarLander 3M：pair 260±7，最好、方差也最小。
   - **HalfCheetah 是例外**：pair_step 从 4M 起停在约 1900，同粒度 GRPO 继续涨到 3455，PPO 到 4809。种子明显分成低、高两个盆地：纯 sign 5/5 落在低盆地，加入"一样"或少量幅度后有 2/5 能出来（n=5，不显著）。**幅度的价值更像是帮助跳出局部最优，而不只是学得更快**——这是你最初那个矛盾的一个更具体的版本。
6. **对重尾噪声 pair 没有优势**；对对称的比较翻转，方向不变，只是变慢（理论上信号乘 1−2ε）。

## 1. 原代码的问题（`pair-grpo-family/all_algos_corrected.py`）

| # | 位置 | 问题 | 后果 |
|---|---|---|---|
| 1 | L888–893, L299 | pair 比较的是 **GAE 优势**（critic + 标量奖励），不是奖励或偏好 | 算法仍需要标量奖励和 critic，没有在测"只需要相对好坏"的主张；实际是 PPO + 优势 rank 变换 |
| 2 | L218/304/539 `reshape((-1, episode_len))` | 假设 rollout 由整数条等长、对齐的轨迹组成 | CartPole/LunarLander/Hopper/Walker/Ant 会提前终止，"行"不是轨迹、"列"不是同一时间步；HalfCheetah 对齐但每组只有 3 条（3000/1000） |
| 3 | L307–326 SoftPairGRPO | 每个样本只和 1 个随机对象比，然后对 ±1 做逐列 z-score | 噪声大；PerfGRPO 才是你描述的"对 b 取平均" |
| 4 | L155, L171 PPO 基线 | 优势归一化被注释掉，且加了 KL 0.5；GRPO 系列没有 KL | 基线之间的差别被这两个因素混淆 |
| 5 | L113 DPO 基线 | `loss = -log(1+exp(-x))`，最小化它会把 x 推向负方向 | **符号反了，在做反偏好**；配对来自不同状态。日志里 Ant/HalfCheetah 回报 −48 万，与此一致 |
| 6 | L716 HardPairGRPO | `target_logps` 已 detach，`∇mean(target − new) = −∇mean(new_logps)` | 偏好构造的 Delta 根本进不了梯度，等价于最大化所有采样动作的似然（熵塌缩）。这解释了 hard 版为什么不行 |
| 7 | L857 | 每个 rollout 开头都 `env.reset()` | 截断进行中的轨迹；轻微 |
| 8 | Logger | 单种子、torch 未设种子、`reward.csv` 用 `"w"` 打开 | 重跑会覆盖；tb 目录里有几十次重跑，论文曲线是单次运行 |

另外：`GRPO_From_Scratch_...ipynb` 和它的副本 `@`、`GRPO.py` 里有明文 HuggingFace token，`jupyter.log` 里有 Jupyter token。这些文件没有推到 GitHub，**但 HF token 请尽快在 HuggingFace 上作废**。

## 2. 方法层面：它到底在优化什么

记组内 K 个样本，pair 优势是

  Â_i = 1/(K−1) · Σ_{j≠i} sign(R_i − R_j) = (2·rank_i − (K−1)) / (K−1)

也就是**中心化排名**。这有几点直接推论：

- **已有工作**：这正是 NES / OpenAI-ES 的 fitness shaping（Wierstra et al. 2014；Salimans et al. 2017），LLM 侧有 Pref-GRPO（2025，用组内胜率当奖励），偏好侧有 SPPO（Wu et al. 2024）和 Nash Learning from Human Feedback（Munos et al. 2023）。论文需要定位到这些工作上，差异点要讲清楚（例如：多步、按步比较、只用比较不用打分）。引用信息请自行核对。
- **准确的不变性表述**：Â 对 R 的任意严格单调变换不变（逐组）。这是"只需要序信息"的精确数学版本。GRPO 只对逐组的仿射变换不变。
- **期望更新方向**：E_a[∇log π(a) · (P(a≻b) − P(b≻a))]，b ~ π_old。也就是最大化**对当前策略的胜率**（Borda 分数）。
  - 确定性奖励时，argmax 仍是不动点，没问题。
  - **随机奖励时可以和期望回报相反**。bandit 实验：某臂 90% 小赢、10% 大输（均值更低），20/20 个种子 pair 都收敛到它；只有不做 std 归一化的线性优势选对了。**GRPO 的组内 std 归一化同样选错**，这和 Dr. GRPO 指出的偏差是同一件事。
  - 所以论文的"∇L_pair ≈ C·∇L_GRPO, C>0"一般不成立：每个配对项的符号和线性项一致，但加权和的方向可以不同。建议改成"pair 优化胜率；在确定性/对称噪声下与期望回报的最优解一致"，并把偏态奖励作为已知局限写出来。
- **噪声**：比较以概率 ε 对称翻转时，期望优势正好乘 (1−2ε)，方向不变，只是信噪比下降。真正的弱点是**接近平局的比较**：小噪声就能把 sign 翻过来，而 sign 会把它放大成满额 ±1。
- **关于"奖励幅度有助于更快找到最优解"这个矛盾**：可以用一个温度连续地折中，Â_i = mean_j tanh((R_i − R_j)/τ)。τ→0 是 sign，τ→∞ 是线性（GRPO 去掉 std）。只有偏好标签时，对应的是用标注一致率得到的软标签 2P(a≻b)−1。实验里 soft_step（τ=0.5）和 pair_step 在大多数任务上打平。唯一的例外是 HalfCheetah 长训练：纯 sign 卡在局部最优，加入幅度或"一样"后部分种子能跳出（4.6，n=5 不显著）。

## 3. 多步问题

- **按时间步比较（pair_step）**：同一初始状态出发的 K 条轨迹，在每个 t 比较还活着的轨迹的 reward-to-go G_t。线性情况下 G_t^j（j≠i）是合法的 baseline（与 a_t^i 无关），所以 grpo_step 是无偏 PG；pair_step 是它的 rank 版本。
- **它的缺陷**：同一 t 的不同轨迹处在不同状态，比较结果混入了状态价值差。HalfCheetah/Hopper 上落后 PPO 的就是这一部分（grpo_step 也一样落后）。
- **两个修正方向，都不需要标量奖励**：
  1. **分支（vine）**：在轨迹上抽状态 s_t，恢复环境到 s_t，采 K 个不同动作再各自滚动 H 步，比较结果。这正是理论里同一状态下的 R(a) − R(b)。代价是环境步数，MuJoCo 和 CartPole 都能存取状态。
  2. **偏好 critic**：用比较数据训练 P(a≻b | s)（或者 Bradley–Terry 形式的优势函数），再用它给每一步打分。这样保留"只用比较"，同时拿到 critic 级别的信用分配。从 Phase B 看，**这是追平 PPO 最有希望的方向**。
- **goal-oriented / 不定长**：轨迹比较天然支持字典序偏好（成功 > 失败，成功里越短越好），不需要设计奖励。按步比较时，我的实现只比较时刻 t 还活着的轨迹；已经结束的轨迹不参与。

## 4. 实验

### 4.1 Bandit（`clean/bandit_pair.py`，20 个种子）

| 场景 | grpo(z) | linear | pair | pair+30%翻转 | tanh |
|---|---|---|---|---|---|
| 高斯噪声 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 重尾噪声 t(1.5) | ✓ | 2/20 错 | ✓ | ✓ | ✓ |
| 奖励×1000 | ✓ | 6/20 错 | ✓ | ✓ | ✓ |
| 偏态（胜率高、均值低） | **20/20 错** | ✓ | **20/20 错** | **20/20 错** | **20/20 错** |

### 4.2 Phase A：原流水线里的归因（`clean/phase_a.py`）

原代码的收集、网络、优化器全部不变，只换优势函数。2×2 设计（幅度 vs 排名）×（按 reshape 列 vs 整 batch），加原 PPO。

| Env | ppo_orig | ppo_norm | ppo_rank | grpo_orig | perf_orig（你的 pair） |
|---|---|---|---|---|---|
| CartPole final (0.5M) | **491±10** | 442±26 | 468±25 | 478±17 | 443±104 |
| LunarLander AUC (1.5M) | **−68±2** | −93±6 | −91±4 | −87±4 | −93±5 |
| HalfCheetah final (2M) | 2172±973 | 1763±995 | 1901±846 | 1793±631 | 1948±1128 |
| Hopper final (2M) | **572±138** | 416±48 | 413±53 | 405±75 | 389±52 |

四个只差"排名/幅度、按列/整批"的变体彼此没有显著差别。原流水线本身偏弱：LunarLander 1.5M 步都没解出来。原因是整 batch 只做 10 步梯度、没有观测归一化、actor/critic 共享优化器且 critic loss 未缩放。新流水线下 PPO 2M 步能到 220+。

### 4.3 Phase B：只用比较、不用 critic（`clean/phase_b.py`）

每组 K=8 条轨迹，同一个 reset seed；所有方法超参相同；非 PPO 方法的优势只除以 batch std。

| Env | ppo | grpo_traj | pair_traj | grpo_step | pair_step | soft_step |
|---|---|---|---|---|---|---|
| CartPole (0.4M) | 500 | 500 | 500 | 499 | 500 | 499 |
| Acrobot (0.5M) | **−82±1** | −90±3 | −87±2 | −86±3 | −86±0.4 | −86±0.4* |
| LunarLander (2M) | 224±41 | 227±69 | 151±153 | 242±34 | **263±7** | 261±6 |
| Hopper (2M) | **2062±676** | 1399±347 | 1631±554 | 1287±272 | 1597±449 | 1779±664 |
| HalfCheetah (2M) | **2718±1558** | 2±141 | −39±64 | 1123±170 | 1094±179 | 1311±432 |

\* Acrobot 的 soft_step 在这一轮因为噪声尺度 bug 退化成了 sign（见 4.5），数字和 pair_step 相同。

PPO 是这里唯一用了 critic 和逐步标量奖励的方法，是信息更多的参照，不是同等信息下的对照。同等信息下的对照是 pair vs GRPO（同粒度）。

### 4.3b 长训练（`results/long`，HalfCheetah 12M / Hopper 5M / LunarLander 3M）

2M 时 Hopper 上 PPO 已经平台，而 pair 还在涨，所以把训练拉长到能看出渐近值。

| Env | ppo | grpo_step | pair_step | pair_traj |
|---|---|---|---|---|
| LunarLander final | 233±38 | 254±22 | **260±7** | – |
| Hopper final | 2229±721 | 2237±652 | 2172±422 | 1116±901（后期退化） |
| HalfCheetah final | **4809±3260** | 3455±1476 | 1976±454 | – |
| HalfCheetah AUC | 3607±2427 | 2379±844 | 1539±230 | – |

逐段学习曲线（每 10% 步数取平均）：

```
HalfCheetah ppo        713 2592 3314 3595 3577 3996 4284 4534 4648 4812
HalfCheetah grpo_step  -45 1054 1708 2304 2705 2990 3212 3161 3243 3457
HalfCheetah pair_step  -21 1018 1438 1640 1773 1839 1893 1912 1919 1976   ← 平台
Hopper      ppo        470 1571 2027 2062 2100 2102 2161 2202 2085 2227
Hopper      pair_step  333  769 1139 1477 1834 1958 2020 2113 2106 2170   ← 追平
```

### 4.6 HalfCheetah 平台：是"接近平局的比较被放大"吗？（`results/softz`，5M）

假设：策略变好后，同一 t 的 8 条轨迹之间的差别大多来自探索噪声；sign 把微小差别放大成满额 ±1，而少数真正好得多的轨迹也只拿到同样的权重。

为此加了两个变体，尺度都用每个 t 组内的 std σ_t：
- `softz_step`：tanh(ΔG/(τσ_t))。τ→0 是 sign，τ→∞ 趋于 grpo_step。
- `tie_step`：|ΔG| < τσ_t 判"一样"(0)，其余 ±1。仍然只用三值比较，对应你最初设定里的"一样"。

| 变体 | final @5M | 各种子 |
|---|---|---|
| pair_step | 1712±260 | 1629 1725 1618 1523 2063 |
| tie τ=0.5 | 1855±543 | 1729 1614 2630 1588 1716 |
| softz τ=0.25 | 1941±503 | 1926 2588 1593 1605 1994 |
| softz τ=4 | 2189±880 | 1684 1637 3130 2774 1719 |
| tie τ=0.25 | 2206±1043 | 1586 1581 3264 2975 1624 |
| softz τ=1 | 2350±929 | 1669 1747 2977 3313 2045 |
| grpo_step | 2557±1080 | 3884 1581 2768 2510 2041 |
| ppo | 3673±2692 | 2149 2115 5910 6181 2012 |

均值都高于纯 sign，但 n=5 下 p 值 0.13–0.5，**不显著**。更有信息量的是种子的分布：
- 存在一个约 1600–2100 的低盆地（可能是某种次优步态，没有渲染验证）。
- 纯 sign 5/5 落在低盆地；tie τ=0.25 和 softz τ=1 各有 2/5 跳出；GRPO 3/5 跳出。

解读（待验证）：幅度/平局信息的作用主要是给少见的大幅改进更大的权重，从而跳出局部最优。要坐实，需要 ≥15 个种子比较"跳出概率"，再渲染看步态。

### 4.4 超参扫描（3 个调参种子，与评估种子分开）

扫描网格：lr ∈ {1e-4, 3e-4, 1e-3} × epochs ∈ {4, 10, 20}，三种方法同样对待。

- **pair_step 对超参最不敏感**。LunarLander 上 9 组配置的 AUC：pair 110–157，GRPO 121–165，PPO −1.5–163。
- 各自最优配置：LunarLander 上 ppo (3e-4, 20)、pair (3e-4, 4)、grpo (1e-4, 10)；Acrobot 上 ppo (1e-3, 10)、pair (1e-3, 4)、grpo (1e-4, 20)。
- pair 最怕"大 lr + 多 epoch"（1e-3, 20 → 110）。它需要更保守的步子，而不是更大的 lr。

各自最优配置在**评估种子 0–4** 上重跑（`results/best`）：

| Env | ppo | grpo_step | pair_step |
|---|---|---|---|
| LunarLander (2M) final | 250±28 | 266±10 | **267±9** |
| LunarLander AUC | 191±29 | 203±12 | 203±16 |
| Acrobot (0.5M) final | **−77±2** | −169±230（1/5 卡在 −500） | −87±2 |

调参之后 LunarLander 上三者打平（pair ≥ PPO，但不显著）；Acrobot 上 PPO 仍然最好。GRPO 用自己"最优"的配置在评估种子上出现了一个失败的种子，pair 没有。这和 4.4 里 pair 对超参不敏感是一致的。

### 4.5 Phase C：噪声

两种噪声：
- **重尾噪声**：给 G_t 加 σ·s·t(1.5)，s 是第一批回报的尺度。标量方法和 pair 看到的是同一份噪声回报。
- **比较翻转**：只对 pair 有意义。

| Env | 条件 | grpo_step | pair_step | soft_step |
|---|---|---|---|---|
| LunarLander (1.5M) | 无噪声（Phase B） | 242±34 | 263±7 | 261±6 |
| | 重尾 σ=1 | −60±114 | 43±144 | 17±97 |
| | 重尾 σ=3 | −137±278 | −91±91 | −54±82 |
| | 翻转 10% | – | 238±24 | – |
| | 翻转 25% | – | 54±113 | – |
| Acrobot (0.5M) | 无噪声 | −86±3 | −86±0.4 | – |
| | 重尾 σ=1 | −327±230 | −312±226 | −281±248 |
| | 重尾 σ=3 | −329±190 | −329±158 | −313±114 |
| | 翻转 10% | – | −172±228（1/5 卡在 −500） | – |
| | 翻转 25% | – | −268±263（2/5 卡在 −500） | – |

Acrobot 的重尾噪声这几行是修正噪声尺度后重跑的（`results/phaseC_fix`）。第一轮里 Acrobot 初期全是 −500，std=0，噪声尺度退化到 1e-3，那一轮作废。

结论：
- **重尾噪声下 pair 没有可测的优势**，三种方法同样崩。
- 翻转 10% 代价小；25% 在 LunarLander 上明显变差，在 Acrobot 上有种子学不出来。原因是稀疏任务里少数有信息量的比较一旦被翻转，就可能被错过。

## 5. 建议的下一步

0. **HalfCheetah 盆地问题**：pair_step / tie_step / grpo_step 各跑 ≥15 个种子，比较"跳出低盆地"的概率，并渲染看步态。如果 tie 版显著提高跳出概率，那么三值比较（好/差/一样）就是论文里应该用的默认形式，而且仍然不需要标量奖励。
1. **偏好 critic**：从比较数据学 P(a≻b|s)，用它给每一步打分。目标是在 HalfCheetah 上追平 PPO，同时只用比较。
2. **vine 分支比较**：作为 (1) 的"无偏、无需学习"的参照，代价是环境步数。
3. **论文重写**：
   - 删掉 hard 版和 DPO 基线（实现有误），改掉梯度正倍数定理。
   - 主张改成三点：只用序信息、对单调变换不变、对超参不敏感（有 4.4 支撑）。
   - 把偏态奖励的局限和重尾噪声的结果如实写出。
   - 所有实验 ≥5 个种子并报告 CI。
4. **一个对 pair 天然有利、而 PPO 做不到的实验**：奖励只能比较、不能打分（例如由一个 LLM 或打分器做两两判断，其打分尺度在不同轨迹间漂移）。在这种设定下，PPO/GRPO 拿到的标量有尺度漂移，pair 不受影响。这比"在标准 benchmark 上打赢 PPO"更能支撑动机。

## 复现

```bash
cd clean
python -m venv ../.venv && ../.venv/bin/pip install numpy torch "gymnasium[mujoco,box2d]" pandas scipy
./run_jobs.sh phase_a.py jobs_a.txt 12        # Phase A
./run_jobs.sh phase_b.py jobs_b.txt 13        # Phase B
./run_jobs.sh phase_b.py jobs_c.txt 13        # Phase C（LunarLander 有效）
./run_jobs.sh phase_b.py jobs_c_fix.txt 13    # Phase C Acrobot 重跑
./run_jobs.sh phase_b.py jobs_sweep.txt 13    # 超参扫描
./run_jobs.sh phase_b.py jobs_best.txt 13     # 最优配置在评估种子上
./run_jobs.sh phase_b.py jobs_long.txt 13     # 长训练
./run_jobs.sh phase_b.py jobs_softz.txt 13    # HalfCheetah 平局/软化变体
../.venv/bin/python summarize.py results/phaseB ppo
```
