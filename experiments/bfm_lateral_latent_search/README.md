# BFM 平面移动固定 z 搜索

这个目录冻结现有 Actor，从指定 reward z 的第一行出发，用对角协方差 CEM 搜索一个更适合目标
平面速度的固定 256 维 latent。它支持前进、后退、左右横移及任意 `(target_vx,
target_vy)`，不是训练，也不会修改 checkpoint。

每个候选只 rollout 5 秒：前 1 秒允许起步，后 4 秒主要评价机体坐标系速度是否接近
`(target_vx, target_vy)`，同时约束直立、偏航和存活。环境关闭 domain randomization 与
observation noise，让候选之间可公平比较。候选根高度低于 0.5 m 或 upright 低于 0.5
并持续 0.2 秒后，剩余时间按 0 分处理；完整存活时间另外占总评分的 20%。当前 MJLab
termination 只有 timeout，所以三个摔倒阈值均由脚本显式实现并可通过命令行调整。

## 运行

执行路径：

```bash
cd /home/thl/wt_wbc/UFO
./experiments/bfm_lateral_latent_search/run_search.sh
```

所有常用参数都直接写在 `run_search.sh` 顶部。设置迭代次数只需修改：

```bash
ITERATIONS=20
```

例如改成 50 轮就写 `ITERATIONS=50`。同一个配置块里还可以直接修改
`POPULATION`、`ROLLOUT_SECONDS`、模型和 z 路径、objective 参数、输出目录以及是否保存
视频/TensorBoard。脚本不会从 `UFO_*` 环境变量读取隐藏配置。

方向由脚本顶部两个速度明确控制，机器人本体坐标中 `+X` 前进、`-X` 后退、`+Y` 左移、
`-Y` 右移。常用配置如下；每次还要把 `TASK` 和 `INITIAL_LATENT` 改成同一任务：

| 动作 | TASK/初始 z | TARGET_FORWARD_SPEED | TARGET_LATERAL_SPEED |
|---|---|---:|---:|
| 前进 0.5 m/s | `move-ego-0-0.5` | `0.5` | `0.0` |
| 后退 0.5 m/s | `move-ego-180-0.5` | `-0.5` | `0.0` |
| 左移 0.5 m/s | `move-ego-90-0.5` | `0.0` | `0.5` |
| 右移 0.5 m/s | `move-ego--90-0.5` | `0.0` | `-0.5` |

`TASK` 用于结果命名，`INITIAL_LATENT` 决定从哪个 reward z 开始搜索，两个 target 决定
真正的优化目标；三者不会自动互相覆盖，必须保持语义一致。

另开一个终端实时查看优化曲线：

```bash
cd /home/thl/wt_wbc/UFO
.venv/bin/tensorboard \
  --logdir experiments/bfm_lateral_latent_search/outputs \
  --port 6006
```

浏览器打开 `http://127.0.0.1:6006`。每轮搜索结束都会立即 flush；Scalar 页面包含
best/baseline score、`vx/vy`、前向/横向/平面速度误差、偏航、左右脚水平误差、upright、root 高度、
存活时间、每维搜索标准差和 latent 距离，Histogram 页面可以查看整个候选群体的分布。用
`--no-tensorboard` 可关闭，
用 `--tensorboard-log-dir` 可指定目录。

新启动的搜索还会把 objective 拆开显示：`reward_raw/*` 是三个 reward 映射到 `0～1`
后的原始质量分，`reward_contribution/*` 是乘权重并经过站立 gate 后对逐步 objective 的
实际贡献，`objective/rollout_before_latent_penalty_*` 是整段 rollout 得分，
`objective/latent_penalty_*` 是因为 z 偏离原始 z 被扣掉的分数，`objective/score_*` 是最终
供 CEM 排序的分数。

逐项运动指标分开放置：`metrics/*` 只显示当前轮 best，`baseline_metrics/*` 只显示原始
z。这样 TensorBoard 展开 `metrics` 时不会把 best 和 baseline 曲线混在一起。

## 对角 CEM 怎么更新

每轮先用当前 256 维均值和逐维标准差采样，然后把噪声投影到当前 z 的球面切空间，并把
候选重新归一化到范数 16。评分最高的 `POPULATION * ELITE_FRACTION` 个候选用于同时更新：

- 新均值：elite 的均值重新投影到范数 16。
- 新协方差：计算 elite 相对新均值的切空间残差，对每个 latent 维度分别估计方差。
- 平滑：`new_cov = COVARIANCE_ALPHA * old_cov + (1-COVARIANCE_ALPHA) * elite_cov`。
- 限幅：每维标准差限制在 `MIN_SIGMA` 到 `MAX_SIGMA`。

因此它不再使用固定的 `SIGMA_DECAY`。TensorBoard 的 `search/std_mean`、`std_min`、
`std_max` 显示当前搜索分布，`search/std_per_dimension` 直方图显示 256 个维度各自的探索
尺度。某维 elite 越一致，该维标准差就越快缩小；仍有多种较优取值的维度会保持更大探索。

## 扩展 objective 和 metric

所有逐步逻辑注册在 `objective_registry.py`。搜索循环会自动调用：

```python
@register_reward_term("new_reward", weight=0.1)
def new_reward(signals, config):
    return ...

@register_metric("new_metric", reduction="rms")
def new_metric(signals, config):
    return ...
```

metric 支持 `mean`、`rms`、`min`、`max`。注册后会自动进入 `history.json`、
`summary.json`、TensorBoard scalar 和候选群体 histogram，不需要修改 rollout 累加器或
TensorBoard logger。每次运行的 `summary.json` 还会保存完整 registry 快照。

当前 objective 的逐步得分由速度跟踪、偏航稳定和脚掌水平三项组成，权重分别为
`0.75 / 0.15 / 0.10`。脚掌水平项直接复用训练环境 aux reward `penalty_feet_ori` 的定义：
将世界重力旋转到每只脚的局部坐标系，取 XY 分量模长，并且只在该脚接触地面时计入。
接触判断使用同一 aux 模块中 `penalty_slippage` 的接触力范数，避免 MJLab 与其他后端的
接触力正负方向约定不同导致 `force_z > 1` 恒为假。
`run_search.sh` 顶部的 `FOOT_FLATNESS_SIGMA` 控制容忍度，默认 `0.20` 约对应单脚倾斜
11.5 度；数值越小，对脚掌倾斜越敏感。`VELOCITY_WEIGHT`、`YAW_WEIGHT` 和
`FOOT_FLATNESS_WEIGHT` 可直接调整三项权重，三者必须非负且总和为 1。

实际默认规模直接以 `run_search.sh` 顶部的 `POPULATION`、`ITERATIONS` 和
`ROLLOUT_SECONDS` 为准。先做快速冒烟测试可用：

```bash
cd /home/thl/wt_wbc/UFO
./experiments/bfm_lateral_latent_search/run_search.sh \
  --population 8 \
  --iterations 2 \
  --rollout-s 2 \
  --warmup-s 0.5 \
  --no-save-video
```

如果显存不够，先减小 `--population`；它影响并行环境数，不影响每条 rollout 的长度。

默认直接加载 `runs/新数据addlelay_onnx_153m_20260807/exported/FBcprAuxModel.onnx`，
因此搜索与现有部署推理使用完全相同的 Actor，不依赖原始 PyTorch checkpoint；可用
`--actor-onnx` 修改。provider 默认是 CUDA，也可以传 `--onnx-provider cpu`。环境初始化使用 Roban 的
`cache/motion_data/roban_s22/roban_lafan_10s_inference_ufo.pkl`，避免误读默认 G1 motion
cache；可用 `--data-path` 修改。

## 输出

每次运行保存在：

```text
experiments/bfm_lateral_latent_search/outputs/TASK_cem_foot_flat_YYYYMMDD_HHMMSS/
├── best_z.npy                 # [1, 256]，便于平台固定 z 使用
├── best_z_rollout.npy         # [5000, 256]，可直接给现有 ONNX MuJoCo 脚本
├── baseline_z.npy             # [1, 256]，本次搜索使用的原始 z
├── baseline_z_rollout.npy     # [5000, 256]，原始 z 的可播放版本
├── history.json               # 每轮最优、baseline、速度和存活统计
├── summary.json               # 最终结果和完整参数
└── comparison_baseline_left_best_right.mp4
```

`run_search.sh` 启动时只生成一次 `RUN_TIMESTAMP`，因此同一次运行的 z、JSON、视频和
TensorBoard 会进入同一个目录；再次运行会生成新的时间戳目录，不会覆盖上一次结果。

对比视频左侧是原始 reward z，右侧是搜索后的 z。要在现有交互 MuJoCo 中单独查看已经
导出的最优结果（不会重新计算或覆盖 z）：

```bash
cd /home/thl/wt_wbc/UFO
./experiments/bfm_lateral_latent_search/view_best.sh
```

也可以把另一次搜索的输出目录作为第一个参数传给 `view_best.sh`。不要用
`run_reward_inference_onnx.sh` 查看搜索结果：那个脚本的职责是生成 reward z，metadata
不匹配时会重新计算并覆盖指定 latent。

本仓库已经完成的两阶段结果位于
`outputs/move-ego-90-0.5_refined_onnx/`，其中
`comparison_original_left_final_right.mp4` 是最原始 z 与最终 z 的直接对比。

## 为什么先用 5 秒

横移是固定速度目标，起步后几秒就能暴露速度不足、向前串扰、偏航或摔倒，因此 5 秒适合
搜索阶段快速淘汰。它不能证明长期稳定：若短时结果改善，下一步应把最优若干 z 用 20–30
秒、多个初始扰动复验，再决定是否替换平台资源。
