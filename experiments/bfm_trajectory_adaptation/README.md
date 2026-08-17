# BFM 轨迹 ADAPT

这里实现的是轨迹级 latent adaptation，不修改 Actor：先用 backward encoder 生成 zero-shot
tracking z 序列，再对一个轨迹片段的整段 `z[t]` 做批量、零阶采样优化。

实现参考 BFM-Zero 论文给出的设置：从 tracking sequence warm-start、迭代退火系数
`beta1=0.85`、时间相关噪声系数 `beta2=0.9`、默认 6 轮。论文实验使用 2048 particles；本机
初步验证默认 128，确认目标和显存后可显式传 `--particles 2048`。

优化目标由 global/root-relative keypoint tracking、root position、关节位置、直立和存活组成。
候选 latent 每帧保持范数 16，且加入相对 zero-shot z 的小正则。Actor 与
`onnx_mujoco_sim.py` 使用同一个 ONNX。

## 执行

```bash
cd /home/thl/wt_wbc/UFO
./experiments/bfm_trajectory_adaptation/run_adapt.sh
```

默认优化 `dance1_subject2` 的前 5 秒。指定 motion 和片段：

```bash
cd /home/thl/wt_wbc/UFO
UFO_ADAPT_MOTION=/home/thl/Downloads/retargeter/aiming2_subject2.roban_s22.fk50.npz \
./experiments/bfm_trajectory_adaptation/run_adapt.sh \
  --start-frame 500 \
  --duration-s 5 \
  --particles 128 \
  --iterations 6
```

复现论文粒子数需要较大显存：

```bash
./experiments/bfm_trajectory_adaptation/run_adapt.sh --particles 2048 --iterations 6
```

直接 motion 输入会建立/刷新对应的单 motion cache，因此脚本支持
`--rebuild-motion-cache`；这不是训练。

## 输出

每次运行写入 `outputs/<motion>_f<start>_<time>/`：

- `adapted_latent.npy`：完整长度 latent，只替换被优化片段，可直接传给 ONNX MuJoCo。
- `adapted_segment.npy`：本次优化的 `[T,256]` 片段。
- `history.json`：每轮 baseline、最优 tracking 指标和采样统计。
- `summary.json`：最终指标、输入和完整参数。
- `comparison_reference_baseline_adapted.mp4`：左为参考，中为 zero-shot，右为 ADAPT。

查看完整 adapted latent：

```bash
cd /home/thl/wt_wbc/UFO
.venv/bin/python -m humanoidverse.onnx_mujoco_sim \
  --model-folder runs/新数据addlelay_onnx_153m_20260807 \
  --motion /目标/motion.npz \
  --latent /输出目录/adapted_latent.npy
```
