# roban_s22_handball 资产说明

## 1. 资产定位

本资产继承自 `kuavo-ros-control` 的 `dev` 分支中
`biped_s17` 模型，是一个面向 MuJoCo 和 URDF 使用的 Kuavo S17 子版本。
该子版本采用球手、固定头部，并删除不参与机器人本体动力学的定位小球。

当前权威模型文件为：

- URDF：`urdf/biped_s17_verified_handball_fixedhead.urdf`
- MuJoCo XML：`xml/biped_s17_verified_handball_fixedhead.xml`

## 2. 参数选取原则

Kuavo 上游的 S17 URDF 与 MuJoCo XML 并非所有字段都完全一致。本资产采用
以下规则：

1. 同一个参数在 URDF 和 XML 中都有定义时，原则上以 Kuavo URDF 为准。
2. 参数只存在于 Kuavo XML 时，以 Kuavo XML 为补充来源。
3. 对已经确认的冲突项，采用下表中的明确选择。

| 冲突或缺失项 | 本资产采用的数值或来源 |
|---|---|
| `leg_l6_link`、`leg_r6_link` 质量和惯量 | Kuavo URDF 版本 |
| `zhead_2_link` 质量和惯量 | Kuavo URDF 版本 |
| `waist_yaw_joint` 力矩限制 | joint 与 motor 均为 `±80 N·m` |
| `waist_yaw_joint` 阻尼 | Kuavo XML：`damping="0.2"` |
| `waist_yaw_joint` 摩擦 | Kuavo XML：`frictionloss="0.0"` |
| 活动关节电枢惯量 | Kuavo XML：`armature="0.003"` |
| URDF 中的关节速度限制 | 保留在 URDF；MJCF 没有直接等价的 joint 字段 |
| `camera_base`、`head_radar` 的微小质量 | 跟随 Kuavo XML，在 XML 中忽略 |

Kuavo URDF 将腰关节 effort 定义为 `±80 N·m`；Kuavo XML 的 motor
`ctrlrange` 是 `±80`，但 joint `actuatorfrcrange` 是 `±50 N·m`。
本子版本根据项目内部的电机峰值力矩记录，统一采用 `±80 N·m`。相关内部
记录：[S17 电机参数调研](https://bcn9fa1lvktb.feishu.cn/wiki/QEnYwYN79i7tFgkUTfVc909unqd)。

## 3. 球手结构

当前训练 XML 将棍状手刚性合并到前臂 link：

```text
zarm_l4_link（包含 l_handball.STL）
zarm_r4_link（包含 r_handball.STL）
```

对应视觉 mesh 为：

```text
l_forearm.STL + l_handball.STL
r_forearm.STL + r_handball.STL
```

`l_handball.STL` 和 `r_handball.STL` 只是前臂 body 上的第二个视觉 geom，
不再建立独立的 `l_handball` / `r_handball` body。手部质量、质心和惯量
已按 `biped_s17_fixedheadjoint.xml` 的合并参数计入 `zarm_l4_link` 和
`zarm_r4_link`。

## 4. 相对 Kuavo 上游的有意修改

1. 将躯干设为根 link：
   - 本资产的 `base_link` 对应 Kuavo 的 `torso`。
   - 本资产的 `waist_yaw_link` 对应 Kuavo 的 `base_link`。
   - MuJoCo 中的 IMU 仍位于躯干上。
2. 删除左右脚底共 12 个无质量、仅带 5 mm 碰撞球的定位 link。
3. 删除左右手末端的 `zarm_l7_end_effector` 和
   `zarm_r7_end_effector`，以及对应的 5 mm 定位球。
4. 将 `zhead_1_joint` 和 `zhead_2_joint` 设置为固定关节；


## 5. URDF 与 MuJoCo XML 的对齐状态

截至 2026-07-29，已完成以下核验：

| 项目 | 核验结果 |
|---|---|
| 共同物理刚体 | 26 个 |
| 共同刚体的质量、质心和惯量 | 全部一致 |
| 活动关节 | 21 个 |
| 活动关节的轴、角度范围、力矩、阻尼和摩擦 | 按本 README 的规则一致 |
| 视觉 mesh 引用数量 | URDF/XML 均为 28 |
| 碰撞体数量 | URDF/XML 均为 37 |
| 37 个碰撞体的形状、尺寸和局部位姿 | 逐项一致 |

两份模型的源文件总质量分别为：

```text
URDF：40.18604309 kg
XML ：40.18603209 kg
差值：0.000011 kg（11 mg）
```

这 11 mg 来自 URDF 中的：

- `camera_base`：`0.000010 kg`
- `head_radar`：`0.000001 kg`

两者在 URDF 中均为正质量、零惯量定义。为跟随 Kuavo XML 并避免在
MuJoCo 中引入不合理的正质量零惯量刚体，XML 将它们作为无质量固定坐标系
处理；这是已确认的有意差异。

## 6. 删除了无关需要维护的文件

强化学习侧目前没有 Gazebo 和ros使用需求，因此旧的
`biped_s17_gazebo.urdf` 和 `biped_s17_gazebo.xacro` 已删除，不再维护。
目录中的 `launch/gazebo.launch`被同步删除。
`drake`文件夹被删除
`rviz`文件夹被删除
`CMakeLists.txt`文件被删除，

---
本文档核验日期：2026-07-29
