# BFM-Zero ManagerOnly 与 UFO Agents / 数据优先级对比报告

对比日期：2026-07-14

对比范围：

- 当前代码：`/home/thl/wt_wbc/BFM-Zero-ManagerOnly/bfm/agents`
- UFO 代码：`/home/thl/wt_wbc/UFO/humanoidverse/agents`
- 为追踪数据 priority，额外检查了两边的训练 workspace/hook、motion library、训练配置和 expert loader。

对比方法：逐个枚举 Python 文件；同相对路径文件先做字节比较，再检查 unified diff；对名称不同但职责对应的环境、评估和数据加载模块做语义对比。

## 1. 结论摘要

1. 当前目录有 31 个 Python 文件，UFO 有 45 个；共同路径 26 个，其中 20 个完全相同、6 个有差异；当前独有 5 个，UFO 独有 19 个。
2. 对用户给出的 `--agent fb` 命令，UFO 实际构建的是 `FBcprAuxAgent`，不是基础 `FBAgent`。它与当前 `build_bfmzero_aux_agent_config()` 的核心模型结构、损失和关键超参数基本一致。
3. 对这条 FB 训练路径，agents 核心代码最重要的差异是 UFO 增加了分布式梯度 all-reduce/average；这使 8 个进程上的 optimizer 更新保持一致。当前 agents 中没有这部分。
4. UFO 还新增了 GCR-RL、GCR-RL + discriminator、GCR-RL + auxiliary critic，以及 TeCH/TLDR 路线。这些是新算法能力，但不会因为 `--agent fb` 自动启用。
5. 两边 effective priority 公式相同：根据每条 motion 的 tracking EMD 提高困难动作的采样概率，并同时更新在线 motion sampler 和 expert trajectory sampler。
6. priority 的主要差异不在 agent buffer，而在训练编排和 motion library：UFO 支持多数据源固定混合权重、rank 0 评估后广播；当前实现是单进程、单数据源全局归一化。
7. 两边都存在同一个潜在索引问题：给在线 motion sampler 传入的是 expert buffer index，而接口期待 global motion id。只有当 expert buffer 顺序与 motion id 完全一致时才正确。

## 2. 逐文件结果

### 2.1 同路径且完全相同：20 个

| 文件 | 结果 |
|---|---|
| `__init__.py` | 完全相同 |
| `base_model.py` | 完全相同 |
| `buffers/transition.py` | 完全相同 |
| `envs/utils/gym_spaces.py` | 完全相同 |
| `evaluations/base.py` | 完全相同 |
| `fb/__init__.py` | 完全相同 |
| `fb/model.py` | 完全相同 |
| `fb_cpr/__init__.py` | 完全相同 |
| `fb_cpr/configs.py` | 完全相同 |
| `fb_cpr/model.py` | 完全相同 |
| `fb_cpr_aux/__init__.py` | 完全相同 |
| `fb_cpr_aux/model.py` | 完全相同 |
| `misc/__init__.py` | 完全相同 |
| `misc/loggers.py` | 完全相同 |
| `misc/zbuffer.py` | 完全相同 |
| `nn_filters.py` | 完全相同 |
| `nn_models.py` | 完全相同 |
| `normalizers.py` | 完全相同 |
| `pytree_utils.py` | 完全相同 |
| `utils.py` | 完全相同 |

### 2.2 同路径但有差异：6 个

| 文件 | 具体差异 | 训练影响 |
|---|---|---|
| `base.py` | 仅一处注释措辞变化。 | 无运行时影响。 |
| `buffers/trajectory.py` | 当前版通过 `_maybe_compile` 支持 `BFM_DISABLE_TORCH_COMPILE` / `HUMANOIDVERSE_DISABLE_TORCH_COMPILE`；UFO 直接 `torch.compile`。采样与 priority 实现相同。 | 当前版更容易禁用 compile 进行排错；priority 无差异。 |
| `fb/agent.py` | UFO 在 FB loss backward 后平均 forward/backward 网络梯度，在 actor backward 后平均 actor 梯度。 | 八卡训练的关键差异；单卡数值逻辑不变。 |
| `fb_cpr/agent.py` | UFO 对 discriminator、critic、actor 梯度做跨 rank 平均。 | 保证各 rank optimizer 状态和参数同步。 |
| `fb_cpr_aux/agent.py` | UFO 对 auxiliary critic 和该类覆写的 actor update 做跨 rank 梯度平均。 | 保证 auxiliary/actor 多卡同步。 |
| `nn_filter_models.py` | assertion 文本由 “humenv observations only” 改为 “dictionary observations only”。 | 只影响报错信息。 |

### 2.3 当前代码独有：5 个

| 文件 | 职责 / 与 UFO 的关系 |
|---|---|
| `buffers/load_data.py` | 旧式 humenv H5 expert trajectory 与 buffer 加载工具；UFO agents 目录没有该文件。当前 manager 训练主要走 `standalone_expert.py`，不是这里的 H5 路径。 |
| `envs/bfmzero_manager_isaac.py` | 当前 IsaacLab ManagerBasedEnv 适配层；对应 UFO 的 `envs/humanoidverse_mjlab.py`，但后端和接口组织不同。 |
| `evaluations/bfmzero_manager.py` | 当前 manager 环境的 tracking/goal/reward 评估；对应 UFO 的 MJLab 评估模块。 |
| `fb/huggingface.py` | `FBModel` 的 Hugging Face Hub mixin。 |
| `fb_cpr/huggingface.py` | 旧 checkpoint config/state-dict 到当前 `FBcprModel` 的兼容加载与 Hugging Face Hub mixin。 |

### 2.4 UFO 独有：19 个

| 文件 | 职责 / 新能力 |
|---|---|
| `envs/expert_motion_loader.py` | 从 humanoidverse motion 数据构造 expert trajectory buffer，并附加 `motion_ids`、`file_names`；当前对应逻辑位于 agents 目录外的 `standalone_expert.py`。 |
| `envs/humanoidverse_mjlab.py` | MJLab/MuJoCo-Warp 环境适配器，包含机器人配置、观测、奖励、reset、motion loading 等更完整的后端接入。 |
| `envs/utils/history_handler.py` | 管理 actor/critic 历史观测窗口。 |
| `evaluations/humanoidverse_mjlab.py` | MJLab tracking、goal/reward inference 与逐 motion EMD 评估。 |
| `gcr_rl/__init__.py` | GCR-RL 导出。 |
| `gcr_rl/agent.py` | Goal-conditioned representation RL：goal encoder、contrastive pretraining、mixed-z rollout、actor/critic 更新。 |
| `gcr_rl/model.py` | GCR-RL goal encoder、actor、critic 和 latent geometry；支持 hypersphere/Poincaré 相关表示逻辑。 |
| `gcr_rl_dist/__init__.py` | GCR-RL discriminator 版本导出。 |
| `gcr_rl_dist/agent.py` | 在 GCR-RL 上增加 discriminator、WGAN gradient penalty 和 discriminator reward。 |
| `gcr_rl_dist/model.py` | 给 GCR-RL 模型增加 discriminator。 |
| `gcr_rl_dist_aux/__init__.py` | GCR-RL discriminator + aux 版本导出。 |
| `gcr_rl_dist_aux/agent.py` | 增加 auxiliary critic、aux reward 与相应 actor objective。 |
| `gcr_rl_dist_aux/model.py` | 增加 auxiliary critic 网络。 |
| `load_utils.py` | 统一按 config 加载 FB、FB-CPR、FB-CPR-Aux、GCR-RL-Aux 和 TLDR agent/checkpoint。 |
| `presets/__init__.py` | agent preset 导出。 |
| `presets/fb.py` | 集中定义 FB 命令实际使用的 `FBcprAuxAgent` 结构、损失参数和训练 runtime。 |
| `presets/tldr.py` | TeCH/TLDR preset：TE pretrain、dual regularization、softplus objective、goal encoder 训练策略。 |
| `tldr_dist_aux/__init__.py` | TeCH/TLDR agent 导出。 |
| `tldr_dist_aux/agent.py` | TeCH 的内部实现；保留 TLDR 名称用于 checkpoint/config 兼容，包含 temporal-distance/TE pretraining、dual lambda 和 goal encoder schedule/freeze。 |

## 3. `--agent fb` 到底训练什么

UFO 的 `presets/fb.py::build_fb_agent()` 返回 `FBcprAuxAgentConfig`。因此命令中的 `fb` 是 preset 名，不等于只使用 `humanoidverse/agents/fb/agent.py`。

有效算法栈为：

`FB representation + actor` → `CPR discriminator + critic` → `auxiliary critic/rewards`

与当前 `bfm/manager_training/standalone_config.py::build_bfmzero_aux_agent_config()` 对比，以下关键部分相同或实质相同：

- latent dimension 256，normalized z；
- forward/critic/actor 为 2048 hidden、6 层 residual；
- backward encoder 为 256 hidden、1 层；
- batch size 1024，discount 0.98；
- expert ASM ratio 0.6、relabel ratio 0.8；
- mixed rollout、expert trajectory rollout、CPR discriminator、aux critic；
- 默认 FB preset 的 aux reward 集合和缩放与当前配置一致。

UFO FB 路径值得迁移的直接改进：

- 分布式梯度平均：对 8 GPU 是必要能力，不是可忽略的性能小改动。
- 集中 preset：agent 网络/训练 runtime 不再散落在入口和环境代码中。
- `lr_scale`、`clip_grad_norm` 和 `cartwheel_aux_safe` 开关：方便稳定性和动作专项实验。
- 通用 checkpoint loader。

TeCH/GCR-RL 是另一个 agent 路线，需要用对应 preset 才会生效，不应算作当前 FB 命令已经获得的 feature。

## 4. 数据 priority 的完整链路

两边的有效配置都是：

- `prioritization_min_val = 0.5`
- `prioritization_max_val = 2.0`
- `prioritization_scale = 2.0`
- `prioritization_mode = "exp"`

对 motion `i`，原始 tracking 指标为 `EMD_i`，未归一化 priority 为：

```text
p_i = 2 ** (2 * clamp(EMD_i, 0.5, 2.0))
```

示例：

| EMD | 未归一化 priority |
|---:|---:|
| ≤ 0.5 | 2 |
| 1.0 | 4 |
| ≥ 2.0 | 16 |

因此最困难区间相对最容易区间最多约 8 倍权重。之后各 sampler 再归一化成概率。

链路如下：

1. tracking evaluation 对每条 motion 计算 EMD；两边 priority 读取的都是 `metric["emd"]`，核心含义是 joint-position tracking EMD。
2. training hook/workspace 对 EMD clamp、scale、exp。
3. 同一组 priority 更新在线环境的 motion sampler，影响后续 reset/load 时选择哪条 reference motion。
4. 同一组 priority 更新 expert `TrajectoryDictBuffer`，影响 CPR/FB 更新时抽到哪条 demonstration sequence。
5. `TrajectoryDictBuffer.sample()` 按 priority 调用 `torch.multinomial(..., replacement=True)`，然后在选中的 trajectory 内采样序列起点。

## 5. 两份 priority 实现有无区别

### 相同点

- EMD 来源和 effective 变换公式相同。
- 都同时更新 online motion sampler 和 expert trajectory sampler。
- 两边 `agents/buffers/trajectory.py` 的 priority 更新、归一化和 multinomial 采样逻辑相同；唯一 diff 是 compile 包装。
- 都让 EMD 越大的困难 motion 获得更高采样概率。

### 不同点

| 维度 | 当前 ManagerOnly | UFO |
|---|---|---|
| 执行位置 | `standalone_hooks.py` 直接评估并更新。 | `workspace.py` 仅 rank 0 评估，随后向所有 rank 广播 payload。 |
| 多卡一致性 | 当前训练路径没有 priority broadcast。 | 每个 rank 都更新自己的 env sampler 和 expert buffer。 |
| 数据源 | 当前 manager 配置为单 motion path；named NPZ sampler 全局归一化。 | 支持多个 data path 和 `data_mix_weights`。 |
| 多源归一化 | 无 source 概念。 | 先在每个 source 内按 priority 归一化，再乘固定 source weight；priority 只改变源内 motion 分布，不改变源间比例。 |
| 文件名校验 | named NPZ 实现严格检查 motion id 对应文件名。 | 不匹配时主要给 warning，容错更宽但更容易掩盖映射错误。 |
| 默认评估周期 | 当前默认约每 1,024,000 env steps。 | UFO FB preset 为每 3,200,000 global env steps。 |
| 评估后环境处理 | 当前 shared-env hook 保存/恢复部分 episode 状态，并标记评估前 transition。 | UFO 分布式 workspace 在评估后统一 reset，各 rank 执行同步流程。 |

注意：UFO `TrainConfig` 类本身还保留 `max=5, mode=bin` 的通用默认值，但用户给出的 FB 训练构建流程在 `train.py` 中覆盖成 `max=2, mode=exp`；所以比较该命令时，两边 effective 配置相同。

## 6. 发现的风险与可能 bug

### P0：motion id 与 expert buffer index 混用

两边优先级代码都先通过 `index_in_buffer[motion_id]` 得到 expert trajectory index `idxs`。这个 `idxs` 应传给 `expert_slicer.update_priorities()`；但它同时被传给环境的 `update_sampling_weight_by_id(... motions_id=idxs ...)`。

在线 motion library 的接口期待 global motion id，而不是 expert buffer index。只有 expert trajectory 的存储顺序与 global motion id 完全一一对应时，两者才碰巧相等。UFO 代码甚至已经构建了 `motions_id` 列表，但最终环境更新仍传 `idxs`。

建议修正为：

- online env sampler：传 `motions_id`；
- expert trajectory sampler：传 `idxs`；
- 加一个故意打乱 expert buffer 顺序的单元测试。

### P0：当前 checkout 的 PKL motion loader 不完整

当前 manager provider 对非 `.npz` 数据分支导入 `.light_motion_lib.BFMZeroLightMotionLib`，但本 checkout 的 `bfm/manager_envs/mdp` 中没有 `light_motion_lib.py`。而当前默认/示例数据路径仍可能是 `.pkl`。

这意味着在该 checkout 中使用用户命令所示的 `lafan_29dof_10s-clipped.pkl` 路径存在导入失败风险。应恢复该模块、迁移 UFO 的 PKL loader，或明确把当前默认数据切换到已支持的 named NPZ。

### P1：partial evaluation 与 trajectory priority assertion 冲突

两边 `TrajectoryDictBuffer.update_priorities()` 接受 `idxs`，但同时断言：

```python
len(priorities) == len(self.priorities)
```

因此如果只评估部分 motions（例如启用 `max_eval_motions`），即使提供了正确 `idxs` 也会失败。若要支持部分评估，应把断言改成 `len(priorities) == len(idxs)`，并明确未评估 trajectory 是保留旧权重、设 floor，还是重新统一归一化。

### P1：UFO 的文件名 mismatch 只 warning

多源数据下，motion 顺序错误会直接把困难度分配给错误动作。建议将 warning 升级为默认 error，并仅在显式兼容模式下降级。

### P2：priority 动态范围有限且没有新鲜度机制

当前公式把 EMD 压到 `[0.5, 2.0]`，概率动态范围最多 8 倍，稳定但较保守；且只在评估周期刷新。可以实验 EMA、priority floor、importance-sampling correction 或 success/failure 与 EMD 混合指标，但应先修复索引和 loader 风险。

## 7. 建议的优化顺序

1. 先修 motion id / buffer index 分离，并补 mapping 测试；这是 correctness 问题。
2. 确认当前 `.pkl` loader 可用，否则 8 卡长训练可能在环境创建阶段就失败。
3. 若当前代码要支持 8 GPU，迁移 UFO 的 gradient average、rank-0 evaluation 和 priority broadcast，并验证 optimizer step 后各 rank 参数一致。
4. 若要混合多个动作数据集，迁移 UFO 的 source-stratified priority；保留固定 source weights，避免某一困难数据源吞掉全部采样量。
5. 在 FB baseline 稳定后，再独立做 TeCH/GCR-RL ablation。不要同时更换算法、priority 和环境后端，否则难以归因收益。

## 8. 关键代码位置

当前代码：

- `bfm/manager_training/standalone_hooks.py`：EMD 转 priority，以及更新两个 sampler。
- `bfm/manager_training/standalone_config.py`：effective priority 参数和 FB-CPR-Aux 配置。
- `bfm/agents/buffers/trajectory.py`：expert trajectory priority 采样。
- `bfm/manager_envs/mdp/named_npz_motion_lib.py`：单源 online motion priority。
- `bfm/agents/evaluations/bfmzero_manager.py`：逐 motion tracking EMD。
- `bfm/manager_training/standalone_expert.py`：expert buffer 与 motion id/file name metadata。

UFO：

- `/home/thl/wt_wbc/UFO/humanoidverse/training/workspace.py`：rank-0 evaluation、priority 计算和广播。
- `/home/thl/wt_wbc/UFO/humanoidverse/train.py`：FB 命令 effective priority 参数。
- `/home/thl/wt_wbc/UFO/humanoidverse/agents/buffers/trajectory.py`：expert trajectory priority 采样。
- `/home/thl/wt_wbc/UFO/humanoidverse/utils/motion_lib/motion_lib_base.py`：多源 source-mixed online priority。
- `/home/thl/wt_wbc/UFO/humanoidverse/agents/evaluations/humanoidverse_mjlab.py`：逐 motion tracking EMD。
- `/home/thl/wt_wbc/UFO/humanoidverse/agents/envs/expert_motion_loader.py`：expert buffer metadata。
