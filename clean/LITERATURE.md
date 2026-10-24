# 文献地图：pair 方法哪些已被做过、哪些还空着

检索日期 2026-10-24。三路并行检索，每篇都打开过 arXiv/会议页核实。
我亲自打开核对过的：Tournament-GRPO（读了正文）、SP3O、GiGPO、GOPO、SPO（Swamy）。其余由检索代理核实，方法细节多数来自摘要页，引用前请再读原文。

## 1. 新颖性总表

| 主张 | 状态 | 置信度 | 最近的工作 |
|---|---|---|---|
| 组内两两比较胜率 / 中心化秩当优势，放进 PPO/GRPO | **已发表** | 高 | Pref-GRPO 2025；GOPO 2026；Liu et al. 2026；ArenaRL（ICML 2026）；Tournament-GRPO（EMNLP 2026，用 wins/(K−1) 当 baseline）；SPO（ICML 2024，控制任务、轨迹级）；ES 的 fitness shaping |
| "优化胜率；确定性奖励下与期望回报最优一致" | **已知，只能当引理** | 中高 | SPO 引 Dudík 2015；Zhang & Ranganath（ICML 2025）；SPPO、NLHF |
| 秩/比较优势让训练更稳、抗 hacking（定性） | **已被占** | 中高 | Pref-GRPO（"虚假优势"）；GOPO；RLRR 2026 |
| 预训练阶段用两两裁判 | **部分被占** | 中 | Self-Improving Pretraining（2026，全两两 + online DPO）；竞争路线有 RPT、RLPT、RLP（标量自监督奖励） |
| 比较 vs 逐条打分哪个更可靠 | **证据混合** | 中 | 支持：Landesberg 2026、Remeli & Hardt（ICML 2026）、众包研究；反对：Comparative Trap 2024（放大表面偏好）、Coin Flip Judge 2026（位置偏差，13.6% 翻转） |
| 多步 RL 中逐时间步、纯序数比较 reward-to-go（无 critic、无标量） | **看似空白** | 中低 | 标量版已有：PI²（2010）、GR2PO（2026）、GiGPO（NeurIPS 2025，按锚状态分组，审稿人会拿来当 baseline） |
| 只从两两比较学 critic，给 PPO 逐步优势 | **没找到直接先例** | 中 | 理论基础：Knox（AAAI 2024）、CPL（ICLR 2024）、IPL（NeurIPS 2023），都是离线；最近的在线工作：SP3O（2026，无奖励模型、无 critic、PPO 式，但只比一对片段）、ZPG（ICLR 2025） |
| 在线策略梯度里用三值（含平局）比较；平局帮助探索 | **没找到**（我们自己的 15 种子实验也不支持"帮助探索"） | 中 | 平局只在离线 DPO（DPO with ties，NeurIPS 2025）和奖励学习里研究过；SPO 的 P=½ 给 0 是隐式平局；在线方法都强制选出赢家 |
| 先用比较训练、再用奖励微调 | **没有受控研究** | 中 | Tulu 3（离线 DPO → RLVR）；Rufus-Air 2026 的顺序相反（先硬奖励后软裁判），审稿人会拿来质疑 |
| 随机奖励下胜率最优与期望最优分离（偏中位数、风险敏感） | **可能是新理论点** | 中低 | 检索没有命中专门分析 |

## 2. 对论文定位的含义

**不能当贡献的：**
- "提出用 ±1 两两比较代替 reward 的 GRPO"：上面至少 5 篇已经做了。
- "训练更稳定"作为定性主张：已有。
- "胜率最优等于奖励最优"：只是引理。

**可以当核心贡献的：多步任务里只靠比较做逐步信用分配。** 已有工作的比较都停在轨迹级（SPO、ArenaRL、Tournament-GRPO），或者逐步但用标量（GiGPO、GR2PO、PI²）。候选做法：
1. 只从比较学 critic（P(a≻b|s) 或 BT 优势），给 PPO 逐步优势，目标是追平用标量 critic 的 PPO；
2. 带平局的三值标签。HalfCheetah 上 15 个种子的检验显示，平局对跳出局部最优**没有作用**（NOTES 4.7），所以平局只能作为"标注更自然"的设计选择，不能作为贡献；
3. 同状态分支（vine）+ 比较，作为无偏参照。

**理论上还空着的：**
- 逐时间步比较的估计量性质：基线只依赖时间、不依赖状态，秩变换非线性，所以它不是无偏梯度。它到底在优化什么？
- 随机奖励下胜率最优与期望回报最优的偏离：偏向中位数、风险敏感。

**你的四个卖点各自需要的实验：**
1. **"不需要奖励公式"**：同一个裁判模型，对比逐条打分、rubric、两两比较三种奖励。报告重复判断的一致率和下游黄金指标；做位置交换，控制长度。
2. **"先比较后奖励"**：在奖励稀疏或昂贵的任务上，同算力比较只用奖励、比较→奖励、奖励→比较、DPO→奖励四种方案。
3. **"通用预训练奖励"**：和 RPT/RLP 在同 FLOPs 下画 scaling 曲线，只用自裁判或小裁判；否则会被说成蒸馏强裁判。
4. **"稳定"**：同一裁判下画 Gao 式 proxy/gold 过度优化曲线，做多种子超参扫描。

## 3. 关键文献（已核实链接）

**算法重合**
- Pref-GRPO：https://arxiv.org/abs/2508.20751
- GOPO：https://arxiv.org/abs/2602.03876
- Liu et al.，Reasoning LLMs-as-Judges：https://arxiv.org/abs/2603.12246
- ArenaRL：https://arxiv.org/abs/2601.06487
- Tournament-GRPO：https://arxiv.org/abs/2605.26958
- RRC：https://arxiv.org/abs/2608.06310
- Writing-Zero/BRPO：https://arxiv.org/abs/2506.00103
- Dr. GRPO：https://arxiv.org/abs/2503.20783

**自博弈 / 胜率理论**
- SPO（Swamy，ICML 2024）：https://proceedings.mlr.press/v235/swamy24a.html
- SPPO：https://arxiv.org/abs/2405.00675
- NLHF：https://arxiv.org/abs/2312.00886
- DNO：https://arxiv.org/abs/2404.03715
- Online IPO：https://proceedings.mlr.press/v235/calandriello24a.html
- Win-rate（Zhang & Ranganath，ICML 2025）：https://arxiv.org/abs/2502.10505
- Dudík 2015：https://proceedings.mlr.press/v40/Dudik15.html

**ES 前身**
- NES：https://www.jmlr.org/papers/v15/wierstra14a.html
- OpenAI-ES：https://arxiv.org/abs/1703.03864
- PI²：https://jmlr.org/papers/v11/theodorou10a.html

**多步 / 信用分配**
- GiGPO：https://arxiv.org/abs/2505.10978
- GR2PO：https://arxiv.org/abs/2609.19850
- VinePPO：https://arxiv.org/abs/2410.01679
- Tree-GRPO：https://arxiv.org/abs/2509.21240
- TreeRPO：https://arxiv.org/abs/2506.05183
- SPO（segment）：https://arxiv.org/abs/2505.23564
- RTMC：https://arxiv.org/abs/2604.11037
- Shani et al. 多轮偏好：https://arxiv.org/abs/2405.14655

**偏好 RL（控制）**
- Christiano 2017：https://arxiv.org/abs/1706.03741
- PEBBLE：https://arxiv.org/abs/2106.05091
- Preference Transformer：https://arxiv.org/abs/2303.00957
- Hindsight PRIOR：https://arxiv.org/abs/2404.08828
- Knox 2024：https://ojs.aaai.org/index.php/AAAI/article/view/28870
- CPL：https://arxiv.org/abs/2310.13639
- IPL：https://arxiv.org/abs/2305.15363
- ZPG：https://arxiv.org/abs/2409.17401
- SP3O：https://arxiv.org/abs/2608.02951

**平局**
- DPO with ties：https://arxiv.org/abs/2409.17431
- BT with ties：https://arxiv.org/abs/2410.05328
- Multi-Type Preference：https://arxiv.org/abs/2409.07268

**定位与证据**
- RPT：https://arxiv.org/abs/2506.08007
- RLPT：https://arxiv.org/abs/2509.19249
- RLP：https://arxiv.org/abs/2510.01265
- Self-Improving Pretraining：https://arxiv.org/abs/2601.21343
- SERL：https://ojs.aaai.org/index.php/AAAI/article/view/40539
- Gao 过度优化：https://arxiv.org/abs/2210.10760
- Landesberg 2026：https://arxiv.org/abs/2603.12520
- Comparative Trap：https://arxiv.org/abs/2406.12319
- Coin Flip Judge：https://arxiv.org/abs/2606.13685
- Rubrics as Rewards：https://arxiv.org/abs/2507.17746
- Tulu 3：https://arxiv.org/abs/2411.15124
- Rufus-Air：https://arxiv.org/abs/2609.29421

## 4. 检索缺口

以下几类没覆盖到或结果很薄，投稿前要补：
- Ranking Policy Gradient（Lin & Zhou）、序数 MDP（Weng 2011）、PBPI（Fürnkranz 2012）：后两者的页面被拦截或未检索。
- "偏好 critic" 和 "在线平局" 两项检索结果很薄。这不等于没有相关工作，建议查 ICLR 2027 的 OpenReview 投稿，以及 SP3O、ArenaRL 的引用列表。
