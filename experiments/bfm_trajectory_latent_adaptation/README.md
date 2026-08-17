# BFM 整轨迹 Latent Adaptation

本目录实现一个独立的、仅推理阶段使用的 BFM latent 轨迹优化实验。Actor、
backward encoder、MuJoCo 模型和 reference motion 全部冻结；优化变量只有每个
控制周期使用的 256 维 latent。

默认实验固定为：

- motion：`humanoidverse/data/roban/named_roban_lafan_10s/dance1_subject2_0001.npz`
- motion 数据：501 帧、50 Hz、10 秒
- actor：`runs/新数据addlelay_onnx_153m_20260807/exported/FBcprAuxModel.onnx`
- encoder：同目录的 `backward_encoder.onnx`
- robot：`configs/robots/roban_s22.yaml`
- MuJoCo domain randomization 和 observation noise 均关闭

GPU MuJoCo-Warp 的短 rollout 重复误差约为 `1e-5`，但带接触的 10 秒舞蹈会放大
并行浮点扰动，本机实测 objective drift 约 `0.009`。程序会重复运行 baseline，
记录实际 drift，并与脚本中明确配置的 `DETERMINISM_TOLERANCE=0.02` 比较。超限会
打印 warning 并在 summary 标为 false，但不会丢掉耗时的搜索 checkpoint；最终改善
若没有明显超过实测 drift，不应解释成有效 Adapt 收益。

## 论文依据与复现决策

BFM-Zero 公开了以下 Adapt 设计：固定 policy，优化一段随时间变化的 latent
prompt；从 tracking latent warm-start；用受 DIAL-MPC 启发的采样式轨迹优化；
论文实验使用 2048 particles、6 iterations、`beta_iteration=0.85` 和
`beta_horizon=0.9`。论文没有公开完整 Adapt 源码、score temperature、噪声初值、
时间平滑核、tracking reward 的精确权重/尺度以及 checkpoint 格式。本目录中这些
缺失项均是工程复现决策，不宣称是论文原实现。

论文给出的双层 schedule 可以通过不同的 iteration 编号方向书写。代码使用从 0
递增的优化轮次，因此写成：

```text
variance(i, h) = sigma0^2 * exp(
    -i / (beta_iteration * M)
    -(H - 1 - h) / (beta_horizon * H)
)
```

实际采样所用 standard deviation 是上述 variance 的平方根。

它满足两个可测试的语义：同一轮中 horizon 后段噪声大于前段；同一 horizon 位置
上后期 iteration 噪声小于早期 iteration。采样噪声先投影到当前 latent 的切空间，
再沿时间轴做短窗口平滑。候选及均值更新后逐帧投影到半径 16 的球面。

每轮把 score 标准化后用 softmax 计算 MPPI 权重：

```text
normalized_score = (score - mean(score)) / (std(score) + eps)
weight = softmax(normalized_score / temperature)
mean <- Project16(mean + sum(weight * tangent(candidate - mean)))
```

候选 0 永远是原始 baseline，候选 1 永远是当前 mean。全局 best 从 baseline
初始化。无约束 iteration-best 用于观察 composite trade-off；真正导出的 global
best 还必须在同一轮并行 rollout 中同时满足 objective 不低于 baseline、MPJPE
低于 baseline，再按 objective 取最高，避免 Gaussian composite 用位置精度交换其他项。

## 时间对齐

NPZ 的 reference 帧编号为 `0..500`。第 0 帧只用于初始化 MuJoCo root、关节位置和
速度；优化序列为 `Z=[z_0,...,z_499]`。执行 `z_t` 对应的 action 后，将仿真状态
与 reference 第 `t+1` 帧比较。

现有通用 latent 生成入口通过 motion duration 构造半开时间区间，对这份动作会先
得到 500 个 reference sample，再去掉第 0 帧，因而实际产生 499 行。为避免这个
off-by-one，本实验显式按 NPZ 的 501 帧构造 motion times，并将第 1～500 帧输入
backward encoder，得到严格的 `[500,256]` train-aligned baseline。train-aligned
处理仍然是未来 `seq_length` 窗口均值，再逐行投影到范数 16。

## Tracking objective 与关节软限位

基础配置只包含 reference tracking，不使用 survival、脚掌接触、力矩、能耗或
action-rate。限位优化可额外启用 `joint_limit` soft reward；它不是脚掌/身体接触
reward，也不会改变 MuJoCo 接触模型、XML limit 或策略 action mapping。

| term | weight | default sigma | error |
|---|---:|---:|---|
| root position | 0.15 | 0.15 m | 世界坐标 L2 |
| root rotation | 0.10 | 0.35 rad | 四元数测地角 |
| root-relative body position | 0.30 | 0.10 m | 分别转入各自 root 坐标后的全身 MPJPE |
| body rotation | 0.15 | 0.35 rad | 全身四元数测地角均值 |
| joint position | 0.15 | 0.25 rad | 21 关节 MAE |
| body linear velocity | 0.075 | 0.60 m/s | 全身速度 L2 均值 |
| body angular velocity | 0.075 | 1.50 rad/s | 全身角速度 L2 均值 |
| joint limit（默认关闭） | 0.0 | 0.25 | 21 关节中最坏的 XML hard-range 利用率 |

每项先计算误差 `e_k`，再映射为
`r_k=exp(-0.5*(e_k/sigma_k)^2)`，每帧 objective 是加权和，轨迹 score 是全部
500 帧的均值。`joint_limit` 先把每个关节相对 XML hard-limit 中心的距离归一化：
中心为 0，任一侧 hard limit 为 1；安全比例以内误差为 0，超过后连续增长。它按
每帧最坏关节计算，避免一个踝关节被另外 20 个安全关节平均掉。reward term 和
metric 都在 `objective_registry.py` 注册，新增指标不需要修改优化循环。

soft reward 只负责给零阶搜索连续的优化方向，最终安全性由独立 hard gate 保证：

- `max_joint_limit_utilization <= joint_limit_guard_fraction`；
- guard 必须严格小于 1，因此候选不能接触 XML hard limit；
- max 指标在时间和重复 validation 环境上都取最坏值，不套用于均值的标准误差；
- 即使总 iteration 小于 `validation_interval`，结束前也强制验证最新 search best；
- `--require-limit-safe-best` 下没有通过门禁就保存恢复产物并以非零状态退出。

`--start-frame` 支持从动作中段冷启动的局部窗口。`frame_count=N` 对应 N 个
reference frame 和 N-1 行 latent：action `z[t]` 从 reference `start+t` 推进到
`start+t+1`。

## 运行

### soft_limit_bias 限位闭环实验

一键脚本的顶部显式列出了模型、motion、GPU、窗口、粒子数、奖励权重、sigma、
安全门槛和输出根目录。默认窗口是已确认会产生左踝过冲的 `2950..3050`：

```bash
cd /home/thl/wt_wbc/UFO
./experiments/bfm_trajectory_latent_adaptation/run_soft_limit_bias_limit_optimization.sh
```

脚本先在 MJLab 中搜索并执行多副本最坏值验证；通过后切出严格对齐的 101 帧 NPZ，
再用同一 actor 和 baseline/adapted 100 行 latent 分别运行原生 MuJoCo。每次运行使用
带时间戳的新目录，不覆盖历史结果。MuJoCo 的曲线、`timeseries.npz` 和
`summary.json` 分别保存在 `mujoco_eval/baseline` 与 `mujoco_eval/adapted`。
最后一步会生成 `mujoco_eval/comparison.json`；只要 adapted 仍有任一 XML hard-limit
接触，或归一化利用率超过与 MJLab 相同的 guard，脚本就以非零状态退出，不会把
只在 MJLab 内通过的候选误报为闭环成功。该检查直接读取 `timeseries.npz`，因此
`joint_pos == hard_limit` 也会按接触处理，而不是只检查是否已经越界。

### 通用 adaptation

所有常用参数都明确写在 `run_adaptation.sh` 顶部。先在文件顶部选择模式：

```bash
cd /home/thl/wt_wbc/UFO

# 快速验证：8 particles × 2 iterations × 50 steps
# 编辑 run_adaptation.sh：MODE="smoke"
./experiments/bfm_trajectory_latent_adaptation/run_adaptation.sh

# 完整实验：2048 particles × 6 iterations × 500 steps
# 编辑 run_adaptation.sh：MODE="full"
./experiments/bfm_trajectory_latent_adaptation/run_adaptation.sh
```

每次新运行会创建：

```text
outputs/dance1_subject2_0001_YYYYMMDD_HHMMSS/
├── baseline_z.npy
├── current_mean_z.npy
├── best_z.npy
├── checkpoint.pt
├── history.json
├── summary.json
├── metrics_baseline.json
├── metrics_adapted.json
├── tracking_timeseries.npz
├── tensorboard/
└── comparison_reference_baseline_adapted.mp4
```

查看 TensorBoard：

```bash
cd /home/thl/wt_wbc/UFO
.venv/bin/tensorboard \
  --logdir experiments/bfm_trajectory_latent_adaptation/outputs \
  --port 6006
```

恢复中断的运行（`RESUME_DIR` 改成已有输出目录）：

```bash
cd /home/thl/wt_wbc/UFO
.venv/bin/python -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
  --resume-dir experiments/bfm_trajectory_latent_adaptation/outputs/dance1_subject2_0001_YYYYMMDD_HHMMSS
```

只重新播放/生成三列对比视频：

```bash
cd /home/thl/wt_wbc/UFO
.venv/bin/python -m experiments.bfm_trajectory_latent_adaptation.adapt_trajectory \
  --resume-dir experiments/bfm_trajectory_latent_adaptation/outputs/dance1_subject2_0001_YYYYMMDD_HHMMSS \
  --render-only
```

完整模式的计算量是 6144 条 10 秒 rollout。它需要大量 GPU 显存和时间，建议先让
smoke 模式通过，再运行 full。每轮结束都会保存 mean、best、history 和 RNG 状态。
