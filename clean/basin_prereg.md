# HalfCheetah 盆地实验：事先约定（写于跑新种子之前，2026-10-24）

- 方法：pair_step、tie_step (τ=0.25)、grpo_step；默认超参；HalfCheetah-v4，5M 步。
- 种子：0–14。0–4 复用已有结果：pair/grpo 取自 `results/long` 的 12M 运行，在 5M 处截取（5M 之前与 5M 运行的轨迹相同）；tie 取自 `results/softz`。5–14 新跑，结果在 `results/basin`。
- 指标：4.5M–5M 步内训练回报的平均。
- 高盆地：> 2400。依据是已有 15 个运行在 2063 和 2510 之间没有值。
- 主检验：tie vs pair 的跳出比例，双侧 Fisher 精确检验。次检验：grpo vs pair。
- 比例的区间用 Wilson 95%。
