## Reference observation 与 simulator observation 的 `base_ang_vel` 不一致

### 问题描述

当前 simulator observation 中的 `base_ang_vel` 会经过：

```python
self.base_ang_vel = quat_rotate_inverse(
    self.base_quat,
    root_vel_w[:, 3:6],
    w_last=True,
)
```

随后在 `_raw_actor_obs()` 中应用：

```python
_apply_obs_scale_noise("base_ang_vel", self.base_ang_vel)
```

因此 simulator 输入实际为：

```text
local-frame base angular velocity * obs_scales["base_ang_vel"]
```

当前配置中：

```yaml
base_ang_vel: 0.25
```

但是 expert/reference observation 在 `expert_motion_loader.py` 中使用：

```python
ref_body_angular_vels = motion_res["body_ang_vel_t"]
ref_ang_vel = ref_body_angular_vels[:, 0]
```

并直接拼接进：

```python
state = torch.cat([
    ref_dof_pos,
    ref_dof_vel,
    projected_gravity,
    ref_ang_vel,
], dim=-1)
```

这里没有应用 `obs_scales["base_ang_vel"]`。

此外，`motion_res["body_ang_vel_t"]` 来源于 motion data 的 pose，通过 forward kinematics 和相邻帧旋转差分计算得到，表示 world-frame angular velocity；而 simulator 中的 `base_ang_vel` 是经过 `quat_rotate_inverse()` 转换后的 local/base-frame angular velocity。

因此当前两条数据路径存在两个潜在不一致：

```text
simulator:
    local-frame angular velocity * 0.25

reference:
    world-frame angular velocity
```

### 影响

这会导致 simulator trajectory 和 expert/reference trajectory 的 observation 分布不一致，尤其影响：

- expert discriminator 的输入分布；
- FB backward encoder / TeCH goal encoder；
- reference motion 与 simulator motion 的 latent 对齐；
- imitation tracking 和训练稳定性。

虽然 train/expert observation 后续会经过 BatchNormNormalizer，但这只能部分缓解数值分布差异，不能修正 frame 和 preprocessing 语义不一致的问题。

### 建议修复

在构造 expert/reference observation 时，复用 simulator observation 的处理逻辑：

1. 将 reference root angular velocity 从 world frame 转换到 local/base frame；
2. 应用相同的 observation scale；
3. 不添加 simulator observation noise。

示意代码：

```python
ref_ang_vel_world = ref_body_angular_vels[:, 0]
ref_ang_vel = quat_rotate_inverse(
    base_quat,
    ref_ang_vel_world,
    w_last=True,
)

ref_ang_vel = ref_ang_vel * float(
    env.config.obs.obs_scales.get("base_ang_vel", 1.0)
)
```

同时建议检查并统一 reference 路径中的：

- `dof_pos`
- `dof_vel`
- `projected_gravity`
- `history_actor`

确保它们与 simulator observation 使用相同的 scale 和坐标系约定。

### 相关代码

- `humanoidverse/agents/envs/humanoidverse_mjlab.py`
- `humanoidverse/agents/envs/expert_motion_loader.py`
- `humanoidverse/utils/motion_lib/motion_lib_base.py`
- `humanoidverse/utils/motion_lib/torch_humanoid_batch.py`
