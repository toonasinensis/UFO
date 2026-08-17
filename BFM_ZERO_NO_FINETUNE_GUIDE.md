# BFM-Zero 无网络微调的任务适配实现说明

本文参考仓库根目录的论文 [`bfm-zeros.pdf`](./bfm-zeros.pdf)，说明 BFM-Zero 如何在不重新训练、不修改 Actor/Backward Encoder 参数的情况下实现新任务，并对照 UFO 当前 ONNX reward inference 代码给出后续落地方案。

## 1. 两种“不修改网络参数”的方法

论文中需要区分以下两种方法：

| 方法 | 是否更新网络参数 | 是否需要额外仿真交互 | 主要用途 |
|---|---:|---:|---|
| Zero-shot inference | 否 | 否，只读取已有状态数据 | reward、目标姿态、动作跟踪 |
| Few-shot latent adaptation | 否 | 是，需要仿真 rollout | zero-shot 效果不足时优化 z |

BFM Actor 的形式是：

```text
action = π(observation, z)
```

其中 `z ∈ R^256` 是任务或行为 prompt。预训练已经让 Actor 学会一族由 z 控制的行为。下游任务不再修改 Actor 权重，而是寻找合适的 z：

```text
动作跟踪：z_t = 一段目标动作状态的 B embedding
目标姿态：z = B(goal_state)
奖励任务：z = E[r(s)B(s)]
```

因此，新增任务主要变成“生成或优化 z”，而不是再训练一个单任务策略。

## 2. `rotate-z--5-0.5` 的含义

任务名解析为：

```text
rotate - z - (-5) - 0.5
```

对应代码对象：

```python
RotationReward(
    axis="z",
    target_ang_velocity=-5.0,
    stand_pelvis_height=0.5,
)
```

字段含义：

| 字段 | 含义 |
|---|---|
| `rotate` | 使用旋转 reward |
| `z` | 绕 pelvis/IMU 局部 z 轴旋转 |
| `-5` | 负方向旋转，基准角速度为 5 rad/s |
| `0.5` | pelvis 高度至少保持在 0.5 m |

连续两个 `--` 不是特殊选项，而是字段分隔符 `-` 后面紧跟负号 `-`：

```text
rotate-z-5-0.5   -> 正方向旋转
rotate-z--5-0.5  -> 负方向旋转
```

`5 rad/s` 约等于：

```text
5 / (2π) ≈ 0.80 圈/秒
```

### 2.1 实际旋转速度 reward

当前实现不是“精确跟踪 -5 rad/s”，而是把以下范围设为满分：

```text
-10 <= omega_z <= -5 rad/s
```

其中：

- `-10 ～ -5 rad/s`：旋转速度项为满分；
- `-5 ～ -2.5 rad/s`：从满分线性下降到 0；
- `-12.5 ～ -10 rad/s`：从 0 线性上升到满分；
- 其他范围：基本为 0。

所以更准确的语义是：保持站立，并以负方向至少 5 rad/s、最高约 10 rad/s 快速旋转。

### 2.2 高度与直立约束

最终 reward 为：

```text
reward = move × height_reward × aligned
```

要求：

- pelvis 高度大于等于 0.5 m 时，高度项满分；
- pelvis 越接近地面，高度项越低；
- `pelvis_rotation_matrix[2, 2] >= 0.9` 时直立项满分；
- 身体侧倒或翻倒时，直立项接近 0。

代码虽然计算了 `small_control`，但随后执行了：

```python
small_control = 1
```

因此当前旋转 reward 实际没有动作幅度或力矩惩罚。

对应代码：

```text
humanoidverse/envs/g1_env_helper/rewards.py
```

## 3. Zero-shot reward inference 原理

论文给出的 reward prompt 公式是：

```text
z_r = E[B(s)r(s)]
```

使用 N 个状态做样本估计：

```text
z_r ≈ (1/N) × Σ r(s_i)B(s_i)
```

### 3.1 从已有状态数据中采样

准备状态集合：

```text
D = {s_1, s_2, ..., s_N}
```

这些状态应该尽量来自当前 BFM 的训练分布。论文训练配置表使用约 40 万条 reward inference 样本，附录消融实验使用过 60 万条 LAFAN1 motion 状态。

### 3.2 Backward Encoder 编码

对每个下一时刻状态执行：

```text
B_i = B(s_i)
```

当前模型 latent 维度为 256：

```text
B_bank.shape = [N, 256]
```

Backward Encoder 只需在状态库构建时运行一次。更换 reward 时可以直接复用 B bank。

### 3.3 用新 reward 重新打分

对每个状态计算用户指定的 reward：

```text
r_i = reward(qpos_i, qvel_i, action_i)
```

例如：

```text
r_i = RotationReward("rotate-z--5-0.5")(state_i)
```

这个 reward 不要求在预训练时存在。

### 3.4 合成并投影 z

论文正文形式：

```text
z_raw = Σ r_i B_i
z = Project(z_raw)
```

当前 latent 维度是 256，投影半径为 `sqrt(256)=16`：

```text
z = 16 × z_raw / ||z_raw||
```

### 3.5 使用固定 z 推理

Reward inference 得到一个固定的 `1×256` z。部署时每个控制周期都使用同一个 z：

```text
a_t = π(o_t, z_reward)
```

当前代码将这个 z 重复成 5000 帧：

```text
reward_locomotion.npy
shape = [5000, 256]
```

这和动作跟踪不同：动作跟踪每个时刻使用不同的 z，reward skill 在整个 episode 中通常保持同一个 z。

## 4. UFO 当前实现

当前入口：

```text
./run_reward_inference_onnx.sh
```

主要代码：

```text
humanoidverse/reward_inference_onnx.py
humanoidverse/mjlab_reward_relabel.py
humanoidverse/envs/g1_env_helper/rewards.py
humanoidverse/onnx_mujoco_sim.py
```

现有流程：

```text
读取 replay buffer
    ↓
读取非 terminal transition
    ↓
Backward ONNX 编码 next_obs
    ↓
MuJoCo 根据 qpos/qvel/action 重新计算 reward
    ↓
根据 reward 加权组合 B embedding
    ↓
投影成 256 维 z
    ↓
保存 reward_locomotion.pkl/.npy/.json
    ↓
Actor ONNX 使用固定 z 运行
```

这已经属于不更新网络参数的 zero-shot reward inference。

### 4.1 当前使用的加权公式

当前 ONNX 代码使用仓库已有的 `reward_wr_inference` 变体：

```text
w_i = softmax(10 × r_i)
z_raw = Σ r_i w_i B_i
z = Project(z_raw)
```

这比论文正文的简单加权更强调最高 reward 的少量样本。例如 `reward=1.0` 和 `reward=0.5` 的 softmax 权重比例约为：

```text
exp(10 × 1.0) / exp(10 × 0.5) = exp(5) ≈ 148
```

建议后续同时支持：

```text
paper_mean       # z = Project(Σ r_i B_i)
softmax_weight   # 当前实现
```

并在同一个任务上录制对比视频。

## 5. 当前实现与论文配置的主要差异

### 5.1 当前样本量偏少

当前默认：

```bash
UFO_REWARD_NUM_SAMPLES=10000
```

论文使用约：

```text
400000 ～ 600000
```

1 万条样本对普通前进可能够用，但对高速旋转、特殊蹲姿等稀疏 reward，可能没有足够的高分状态。

### 5.2 当前 buffer 与 Actor 不是同一个 run

当前 Actor：

```text
runs/新数据addlelay_onnx_153m_20260807
```

当前脚本默认 replay buffer：

```text
runs/xml有手base有imu_60m/checkpoint/buffers/train_rank_0
```

两者不是同一个训练 run。即使 observation 维度一致，状态分布也可能不一致，可能影响 reward z 的语义。

### 5.3 论文附录中 motion dataset 的效果更好

论文比较了：

- agent 在线 replay buffer；
- 训练 motion dataset。

附录报告 motion dataset 的 reward inference 效果更好，尤其是 LAFAN1。论文推测 replay buffer 中的 domain randomization 会让状态分布更杂，使 reward inference 更不稳定。

当前 Roban 数据入口：

```text
configs/data/roban_s22.yaml
```

当前训练数据目录：

```text
/home/thl/wt_wbc/UFO/humanoidverse/data/roban/named_roban_lafan_10s
```

因此，更合适的 reward inference 状态库应该优先从这批 motion 构建。

## 6. 推荐的正式实现：一次构建 B bank，多次查询任务

将 reward inference 拆成两个阶段。

### 6.1 阶段 A：构建状态与 embedding bank

从当前 Roban LAFAN motion dataset 采样约 40 万帧，生成：

```text
reward_inference/state_bank/
├── backward_embeddings.npy   # [N, 256]
├── qpos.npy
├── qvel.npy
├── reward_features.npz
└── metadata.json
```

`reward_features.npz` 可以预计算：

```text
pelvis_height
pelvis_up
base_linear_velocity
base_angular_velocity
left/right wrist height
contact state
```

40 万条 float32 256 维 embedding 约占：

```text
400000 × 256 × 4 bytes ≈ 410 MB
```

可以使用 NumPy memmap 和分块 ONNX inference，避免一次性占用过多内存。

构建 B bank 时需要保证使用当前模型的：

```text
runs/新数据addlelay_onnx_153m_20260807/exported/backward_encoder.onnx
```

### 6.2 阶段 B：查询任意 reward

用户提交任务：

```bash
./run_reward_inference_onnx.sh \
  --tasks rotate-z--5-0.5
```

任务查询只执行：

```text
读取 reward features
    ↓
计算 reward vector r
    ↓
读取 B bank
    ↓
矩阵乘法生成 z
    ↓
投影并保存
```

不再重复：

- 读取大型 replay HDF5；
- 运行 backward ONNX；
- 从 qpos/qvel 重建全部 MuJoCo 状态。

任务不变时，继续直接读取当前已经实现的最终 z 缓存；任务变化时，也只需要重新进行 reward 加权。

## 7. Few-shot latent adaptation：不微调网络，只搜索 z

当 zero-shot z 效果不够好时，论文使用 CEM 等无梯度优化方法搜索更好的 z。

保持以下参数冻结：

```text
Actor：冻结
Backward Encoder：冻结
Critic：不需要部署
```

只优化：

```text
z ∈ R^256
```

### 7.1 CEM 初始化

使用 zero-shot latent 作为初值：

```text
mu_0 = z_zero_shot
sigma_0 = 初始搜索标准差
```

### 7.2 每轮优化

1. 采样 K 个候选 latent：

```text
z_k ~ Normal(mu, sigma)
```

2. 投影回半径为 16 的 latent 球面：

```text
z_k = 16 × z_k / ||z_k||
```

3. 每个候选 z 在并行仿真中运行固定 horizon，例如 500 个 policy step：

```text
J(z_k) = Σ task_reward(s_t) - safety_penalty(s_t, a_t)
```

4. 选择得分最高的前 5%～10% elite。

5. 更新采样分布：

```text
mu = mean(z_elite)
sigma = std(z_elite)
```

6. 重复若干轮。论文单姿态实验使用 20 轮 CEM。

最终得到：

```text
z_star
```

部署时只加载 `z_star`，不需要继续执行 CEM，也不修改网络权重。

### 7.3 安全建议

CEM 搜索应该在与目标硬件尽量对齐的仿真环境中执行，不建议直接在实物机器人上随机搜索 256 维 z。推荐流程：

```text
zero-shot z
    ↓
仿真中 CEM 优化
    ↓
筛选稳定且满足安全约束的 z_star
    ↓
保存固定 latent
    ↓
部署到实物
```

## 8. 轨迹级 latent adaptation

论文对轨迹跟踪还提出了优化一段 latent sequence：

```text
z_t, z_{t+1}, ..., z_{t+H-1}
```

论文使用类似 DIAL-MPC 的双重退火无梯度优化：

```text
粒子数 N = 2048
beta_1 = 0.85
beta_2 = 0.9
优化轮数 M = 6
```

这比固定 z 的 CEM 复杂得多，因为优化变量从一个 256 维向量变为一段 256 维序列。建议先完成 reward B bank 和固定 z CEM，再实现轨迹级优化。

## 9. 推荐实施顺序

1. 从当前 Roban LAFAN motion dataset 构建可复用的 B bank。
2. 将默认 reward inference 样本量提升到约 40 万。
3. 增加 `paper_mean` 与 `softmax_weight` 两种聚合模式。
4. 对相同 reward 分别录制视频并对比。
5. 支持不同 mini-batch/seed 生成多个候选 z。
6. 在 MuJoCo 中自动评价候选 z 并选择最佳结果。
7. 实现固定 z 的 CEM latent adaptation。
8. 最后再考虑 trajectory latent sequence 优化。

## 10. 当前命令

运行默认 reward：

```bash
cd /home/thl/wt_wbc/UFO
./run_reward_inference_onnx.sh
```

运行负方向 z 轴旋转：

```bash
cd /home/thl/wt_wbc/UFO
./run_reward_inference_onnx.sh \
  --tasks rotate-z--5-0.5
```

任务和相关配置不变时，会复用：

```text
runs/新数据addlelay_onnx_153m_20260807/reward_inference/reward_locomotion.pkl
runs/新数据addlelay_onnx_153m_20260807/reward_inference/reward_locomotion.npy
runs/新数据addlelay_onnx_153m_20260807/reward_inference/reward_locomotion.json
```

强制重新计算：

```bash
cd /home/thl/wt_wbc/UFO
UFO_REWARD_REUSE_LATENT=false ./run_reward_inference_onnx.sh \
  --tasks rotate-z--5-0.5
```

## 11. 关键结论

- 当前 ONNX reward inference 已经是不修改网络参数的 zero-shot 方法。
- 当前效果上限主要受状态数据覆盖率、状态分布是否与模型匹配、样本量和 reward 定义影响。
- 当前 1 万条、来自另一个 run 的 replay buffer，与论文主实验条件差距较大。
- 优先从当前训练使用的 Roban LAFAN motion 构建 B bank，比直接调 Actor 或重新训练更符合论文方法。
- 如果 zero-shot 仍不够好，可以在仿真中通过 CEM 优化 z，而不是微调 Actor 权重。
