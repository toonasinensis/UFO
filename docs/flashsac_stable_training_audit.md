# FlashSAC 稳定训练机制与 UFO 对照审计

> 审计日期：2026-08-14  
> UFO 范围：`right_roban` 分支，`HEAD=1e10f54`，并包含审计时工作区中的未提交修改  
> UFO 训练入口：当前 `--agent fb`，实际构建 `FBcprAuxAgent`  
> FlashSAC 范围：[arXiv:2604.04539v2](https://arxiv.org/html/2604.04539v2)，官方实现 `87edc906`

`--agent fb` 到当前 preset 的映射可见 [`agents/presets/__init__.py`](../humanoidverse/agents/presets/__init__.py#L25-L38)，实际配置类型可见 [`agents/presets/fb.py`](../humanoidverse/agents/presets/fb.py#L73-L78)。

## 结论先行

UFO 当前确实使用了 residual connection，也有输入归一化、双 critic、target network 等稳定措施；但它**没有实现 FlashSAC 的完整稳定训练栈**。

把论文机制拆成 7 个可核查的原子项后，严格同款实现是 **0/7**：

1. `d → 4d → d` inverted residual：没有。
2. critic/policy head 前的 post RMSNorm：没有。
3. 每个非线性前的内部 BatchNorm：没有。
4. current/next 拼成同一批次的 cross-batch value prediction：没有。
5. categorical distributional critic：没有。
6. 基于 discounted return 方差和最大幅值的 adaptive reward scaling：没有同款；只有 auxiliary reward 的另一种 EMA 缩放。
7. 每次 `optimizer.step()` 后的权重单位球投影：没有。

因此准确说法是：

> UFO 当前是普通的 pre-LayerNorm residual MLP，加上 FB/CPR 自己的 target、双路悲观估计、表示约束和输入归一化；不能称为复现了 FlashSAC 的稳定网络。

这并不等于“UFO 没有稳定措施”或“UFO 一定不稳定”。两者算法也不同：FlashSAC 是 SAC，UFO 当前 `fb` 是 FB + CPR + auxiliary critic。论文机制可以借鉴，但不能直接一一等同。

## 1. FlashSAC 认为不稳定从哪里来

论文把核心问题归结为 bootstrapped critic：下一状态上的估计误差进入当前 Bellman target，经过重复更新后可能递归放大。状态/动作维度越高、模型越大，这个问题越明显。

FlashSAC 的目标不是只裁一个梯度，而是联合控制：

- weight norm；
- feature norm；
- gradient norm；
- critic loss landscape 的 condition number。

来源：[论文 §4.2 Stable Training](https://arxiv.org/html/2604.04539v2#S4.SS2)。

## 2. FlashSAC 的稳定训练机制

### 2.1 Inverted residual backbone

官方实现的一个 block 是：

```text
x
└─ Linear(d → 4d)
   → BatchNorm → ReLU
   → Linear(4d → d)
   → BatchNorm → ReLU
   → + x
```

可写成：

```text
y = x + ReLU(BN(W2 · ReLU(BN(W1 · x))))
```

其中 `W1` 扩维，`W2` 投影回原宽度。4 倍 expansion 来自[官方 `FlashSACBlock`](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC/layer.py#L82-L98)。扩维提供容量，residual connection 改善深层网络的梯度传播。

### 2.2 Post RMSNorm

最后一个 residual block 后、policy/value head 前增加 RMSNorm：

```text
residual blocks → RMSNorm → prediction head
```

论文给出的作用是直接限制每个样本进入 head 前的 feature norm，减少 OOD 输入产生无界激活并污染 bootstrap target 的风险。官方 actor 和 double critic 都这样做，见[网络实现](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC/network.py#L16-L97)。

### 2.3 Pre-activation BatchNorm

Replay buffer 混合了不同训练阶段 policy 采集的数据，分布持续变化。FlashSAC 在每个 ReLU 前放 BatchNorm，精确顺序是：

```text
Linear → BatchNorm → ReLU
```

这里的“pre-activation”是 BN 在激活函数之前，不是 BN 一定在 Linear 之前。论文明确选择 BatchNorm 而非 LayerNorm，理由是利用大 replay batch 的统计量平滑 loss landscape、降低有效条件数。

### 2.4 Cross-batch value prediction

这是内部 BatchNorm 的配套机制，不能割裂看待。

如果分别前向计算 current Q 与 next/target Q，它们会使用不同的 batch statistics。FlashSAC 先拼接：

```text
obs_all = concat(current_obs, next_obs)       # 2B
act_all = concat(current_action, next_action) # 2B
```

再让 online critic 和 target critic 都在这个 `2B` 批次上前向，最后分别取 current half 和 next half。这样 Bellman 两端基于同一种 current/next 混合分布计算 BN 统计。官方实现见 [`update_critic`](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC/update.py#L185-L231)。

因此只把 UFO 的 LayerNorm 换成 BatchNorm，却不改 critic 的 current/next 前向方式，并不是完整复现。

### 2.5 Distributional critic

FlashSAC 不直接回归一个标量 Q，而是在固定 support 上预测 categorical distribution：

```text
support = [G_min, G_max]
network output = atom logits
loss = projected Bellman target 的 cross entropy
```

论文 GPU 配置为：

- support：`[-5, 5]`；
- atoms：101；
- critics：2。

两个 critic 仍根据期望 Q 选择较小者，对应 clipped double Q。官方 categorical head 与 target projection 分别见 [`EnsembleCategoricalValue`](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC/layer.py#L245-L271) 和 [`_compute_categorical_td_target`](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC/update.py#L29-L72)。

论文认为 distributional cross-entropy objective 比标量 MSE 对 noisy target 更不敏感，并能改善优化地形。

### 2.6 Adaptive reward scaling

固定 support 要求 return 尺度不能长期超出边界，否则 target 会堆在 `G_min/G_max` 上。论文使用：

\[
\bar r_t =
\frac{r_t}
{\max\left(
\sqrt{\sigma^2_{t,G}+\epsilon},
G_{t,\max}/G_{\max}
\right)}
\]

其中：

- `σ²(t,G)` 是 discounted return 的运行方差；
- `G(t,max)` 是观测到的 discounted return 最大绝对值；
- `G_max` 是 categorical support 的正边界。

它不减 reward 均值，而是从“方差尺度”和“保证最大 return 能放进 support 所需的尺度”中取更严格者。官方实现见 [`RewardNormalizer`](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/utils/reward_normalization.py#L7-L114)。

Distributional critic 和 adaptive reward scaling 是耦合设计；论文消融也把二者作为一个增量，不能据此拆分两者各自的独立收益。

### 2.7 每步权重投影

这里的 weight normalization 不是 PyTorch `weight_norm` 重参数化，也不是 L2 weight decay。FlashSAC 在每次 `optimizer.step()` 后显式执行：

\[
w_i \leftarrow \frac{w_i}{\lVert w_i\rVert_2}
\]

即每个 Linear 输出单元对应的权重行投影到单位球。BatchNorm 的 `(γ, β)` 联合向量、RMSNorm 的 `γ` 也被投影到规定范数 `√d`。实现见[单位权重和归一化参数](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC/layer.py#L9-L69)，actor 和 critic 都在优化器更新后执行投影，见 [`update_actor`](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC/update.py#L126-L144) 与 [`update_critic`](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC/update.py#L232-L248)。

它直接阻止网络依靠不断增大权重幅值编码信息，以降低 Q 方差和 bootstrap error amplification。

### 2.8 “控制梯度范数”不等于 gradient clipping

这是一个容易误读的地方：论文说完整设计让 gradient norms 保持有界，但论文和官方实现都没有把 `clip_grad_norm_` 列为 FlashSAC 稳定训练组件。

更准确的解释是：

- 权重范数由单位球投影直接约束；
- 最终特征范数由 RMSNorm 直接约束；
- 中间激活由 BatchNorm 控制；
- residual、distributional objective 和上述约束共同改善梯度传播；
- gradient norm 是论文消融里观测到的稳定性指标，而非单独的梯度裁剪操作。

## 3. UFO 逐项对照

| FlashSAC 原子机制 | UFO 当前状态 | 判断依据 |
|---|---|---|
| `d → 4d → d` inverted residual | 无 | UFO 是单个等宽 `Linear(d,d)` 加 skip |
| post RMSNorm | 无同款 | 输出 Block 有 LayerNorm，但没有 head 前 RMSNorm |
| 每个激活前的内部 BN | 无 | residual block 使用 LayerNorm；BN 只用于原始 observation |
| cross-batch value prediction | 无 | current/next 分开更新输入统计、分开做 critic forward |
| categorical distributional critic | 无 | main/aux critic 都输出标量 1，并使用 MSE |
| Flash adaptive reward scaling | 无同款 | 只有 aux reward 的即时 reward EMA 标准差缩放 |
| optimizer 后权重单位球投影 | 无 | optimizer step 后直接结束或做 target EMA |
| 显式 gradient clipping | 可选但默认关闭 | 不是 FlashSAC 机制；且 UFO 开启后也不覆盖 critic |

### 3.1 UFO residual block 不是 inverted residual

UFO 当前 block 位于 [`nn_models.py`](../humanoidverse/agents/nn_models.py#L429-L437)：

```text
y = x + Mish(Linear_d→d(LayerNorm(x)))
```

它有 skip connection，但只有一层 `d → d` Linear，没有 FlashSAC 的 `d → 4d → d` 扩维和投影。主干的构建见 [`ResidualForwardMap`](../humanoidverse/agents/nn_models.py#L461-L499) 与 [`ResidualActor`](../humanoidverse/agents/nn_models.py#L510-L537)。

当前 preset 给 actor、F、main critic、aux critic 选择了 residual 模型，宽度 2048，见 [`presets/fb.py`](../humanoidverse/agents/presets/fb.py#L82-L138)。“宽度大、层数多、带 residual”仍不代表 block 与 FlashSAC 相同。

### 3.2 UFO 的 BatchNorm 只是输入归一化

UFO 对 `state`、`privileged_state`、`last_action`、`history_actor` 使用：

```python
nn.BatchNorm1d(..., affine=False, momentum=0.01)
```

配置与实现见 [`presets/fb.py`](../humanoidverse/agents/presets/fb.py#L140-L149) 和 [`normalizers.py`](../humanoidverse/agents/normalizers.py#L11-L25)。残差块内部仍然是 LayerNorm。

训练时 current obs 和 next obs 先分别调用 normalizer 更新 running statistics，然后在 eval mode 下分别归一化，见 [`fb_cpr_aux/agent.py`](../humanoidverse/agents/fb_cpr_aux/agent.py#L87-L110)。这不是把 `(s,a)` 与 `(s',a')` 拼成一个 `2B` critic batch，也没有解决块内 BN 的 Bellman 两端统计一致性问题。

### 3.3 UFO critic 是 scalar ensemble，不是 distributional critic

Main critic 构建为 `output_dim=1`，见 [`fb_cpr/model.py`](../humanoidverse/agents/fb_cpr/model.py#L34-L52)；aux critic 同样输出 1，见 [`fb_cpr_aux/model.py`](../humanoidverse/agents/fb_cpr_aux/model.py#L30-L48)。两者都构造 scalar Bellman target 并使用 MSE：

- [main critic update](../humanoidverse/agents/fb_cpr/agent.py#L369-L397)；
- [aux critic update](../humanoidverse/agents/fb_cpr_aux/agent.py#L213-L242)。

`num_parallel=2` 表示两套标量 critic 参数，不表示 101 个 categorical atoms。

### 3.4 UFO 的 aux reward normalization 与 FlashSAC 不同

UFO 只对 auxiliary reward 启用了 `translate=False, scale=True`，见 [`presets/fb.py`](../humanoidverse/agents/presets/fb.py#L150-L155)。实现对**即时 aux reward 本身**维护 EMA mean/mean-square，再除以 EMA standard deviation，见 [`nn_models.py`](../humanoidverse/agents/nn_models.py#L694-L743)。

它没有：

- discounted return 统计；
- 最大绝对 return `G(t,max)`；
- categorical support `G_max`；
- 两个分母取 max 的保护。

Main critic 使用的 discriminator reward 没有经过这套 EMA normalizer，见 [`fb_cpr/agent.py`](../humanoidverse/agents/fb_cpr/agent.py#L369-L391)。因此这只能算“也做了 reward scale 处理”，不能算实现 FlashSAC adaptive reward scaling。

### 3.5 UFO 没有每步权重投影

UFO 初始化时会调用正交初始化，但这只发生在训练初始化，见 [`nn_models.py`](../humanoidverse/agents/nn_models.py#L61-L72) 和 [`fb/agent.py`](../humanoidverse/agents/fb/agent.py#L91-L95)。

F/B、main critic、aux critic、actor 的 `optimizer.step()` 后都没有逐权重行的单位范数投影：

- [F/B optimizer](../humanoidverse/agents/fb/agent.py#L272-L281)；
- [main critic optimizer](../humanoidverse/agents/fb_cpr/agent.py#L389-L397)；
- [aux critic optimizer](../humanoidverse/agents/fb_cpr_aux/agent.py#L234-L242)；
- [actor optimizer](../humanoidverse/agents/fb_cpr_aux/agent.py#L286-L292)。

`weight_decay=0.0`，见 [`presets/fb.py`](../humanoidverse/agents/presets/fb.py#L156-L163)；即使改成非零 weight decay，也仍不等价于 FlashSAC 的硬投影。

### 3.6 UFO 的 gradient clipping 当前默认关闭，且不保护 critic

CLI/preset 默认 `clip_grad_norm=0.0`，见 [`train.py`](../humanoidverse/train.py#L90-L117) 和 [`presets/fb.py`](../humanoidverse/agents/presets/fb.py#L30-L36)。数值只有大于 0 才会转成有效开关。

即使显式启用，当前实现也只裁剪：

- forward map；
- backward map；
- actor。

Main critic、aux critic、discriminator 的参数梯度没有调用 `clip_grad_norm_`。所以它不能直接保护最受 bootstrap 误差影响的两个 scalar critic，也不能算 FlashSAC 的等价实现。

## 4. UFO 自己已经有的稳定措施

这些机制确实有价值，但应称为 UFO/FB 自己的设计或通用 off-policy 稳定器。

### 4.1 Target networks 与 Polyak/EMA 更新

UFO 对 F、B、main critic、aux critic 建立 target copy，并按：

```text
target ← (1 - τ) target + τ online
```

更新。实现见 [`_soft_update_params`](../humanoidverse/agents/nn_models.py#L80-L89) 和 [`fb_cpr_aux/agent.py`](../humanoidverse/agents/fb_cpr_aux/agent.py#L189-L209)。当前：

- F/B：`τ=0.01`；
- main/aux critic：`τ=0.005`。

这是常规 off-policy 稳定机制。FlashSAC 也有 target critic，但论文把它归为 SAC 基础，而不是 §4.2 的新增机制。

### 4.2 双 critic 与悲观估计

F、main critic、aux critic 都配置 `num_parallel=2`。UFO 聚合公式是：

```text
mean(preds) - λ · average_pairwise_abs_disagreement(preds)
```

实现见 [`get_targets_uncertainty`](../humanoidverse/agents/fb/agent.py#L331-L346)。当 critic 数量为 2 且 `λ=0.5` 时，它在数学上恰好等于：

```text
min(Q1, Q2)
```

当前 main critic、aux critic 和 actor 相关悲观系数为 0.5，因此与 clipped double Q 的取 min 效果一致；但 `fb_pessimism_penalty=0.0`，FB target 使用两路 F 的均值而不是 min，配置见 [`presets/fb.py`](../humanoidverse/agents/presets/fb.py#L156-L190)。

### 4.3 B/z 的固定范数与 B 正交约束

UFO 会把 z/B 投影到半径 `√d` 的球面，见 [`fb/model.py`](../humanoidverse/agents/fb/model.py#L119-L126)；同时 FB loss 对 B 加 orthonormality 约束，见 [`fb/agent.py`](../humanoidverse/agents/fb/agent.py#L248-L253)。当前 `ortho_coef=100`。

这可以抑制 latent 表示的尺度漂移或塌缩，但它约束的是 FB latent，不是 FlashSAC critic hidden backbone 的 post RMSNorm，也不是网络权重投影。

### 4.4 Discriminator gradient penalty

Discriminator 使用 WGAN-style input-gradient penalty，当前系数 10，见 [`fb_cpr/agent.py`](../humanoidverse/agents/fb_cpr/agent.py#L272-L367)。它约束 discriminator 对插值输入的梯度，不是对 optimizer parameter gradient 做 clipping，也不直接约束 main/aux critic。

### 4.5 有界 action 与 reward 防护

- Actor mean 经过 `tanh`，最终 sample clamp 到 `[-1+1e-6, 1-1e-6]`，见 [`nn_models.py`](../humanoidverse/agents/nn_models.py#L510-L537) 和 [`TruncatedNormal`](../humanoidverse/agents/nn_models.py#L662-L683)。
- 训练采样噪声还会按 `stddev_clip=0.3` 裁剪；policy std 固定为 0.05。
- Discriminator probability 在取 log-odds reward 前 clamp 到 `[1e-7, 1-1e-7]`，因此该 reward 数值大致限制在 `±16.12`，见 [`nn_models.py`](../humanoidverse/agents/nn_models.py#L365-L369)。

这些是数值防护，但不是 FlashSAC §4.2 的 feature/weight/critic objective 设计。

### 4.6 NaN/Inf fail-fast

训练工作区默认 `fail_on_nonfinite=True`，会检查 reset observation 和每次 agent update 返回的 metrics；发现非有限值会抛 `FloatingPointError`，见 [`workspace.py`](../humanoidverse/training/workspace.py#L254-L343) 与 [`workspace.py`](../humanoidverse/training/workspace.py#L923-L947)。

完整 model parameter/buffer 和 rollout 周期检查的默认间隔都是 0，因此默认关闭。该机制是“尽早发现并中止”，不会跳过坏 batch，也不会自动恢复训练。

### 4.7 正确的 timeout bootstrap 语义

UFO 的 discount 只因 `terminated` 归零，不因 `truncated/timeout` 归零，见 [`fb_cpr_aux/agent.py`](../humanoidverse/agents/fb_cpr_aux/agent.py#L87-L97)。这避免把时间上限错误地当作 MDP 终止，属于 target 语义正确性，不是 FlashSAC 特有机制。

## 5. 论文消融到底支持什么结论

论文 §6.3 按如下顺序从普通 MLP 增量加入组件：

```text
MLP
+ residual blocks
+ BatchNorm
+ post RMSNorm
+ distributional critic / reward scaling
+ weight normalization
= full FlashSAC
```

论文报告：随着组件加入，weight、feature、gradient norm 的无控制增长逐渐消失，critic condition number 单调下降，完整模型最低，最终任务性能也提高。Weight normalization 单项增益相对较小，但在样本不足时提高鲁棒性，因此保留。

来源：[论文 §6.3 Architectural Ablation](https://arxiv.org/html/2604.04539v2#S6.SS3)。

需要限制解读范围：

- Cross-batch prediction 没有单独消融，应视作 BatchNorm 的配套实现。
- Distributional critic 与 reward scaling 被绑定加入，不能由该图判断二者各自的独立贡献。
- 消融在四个 IsaacLab 任务上完成，不等于所有 60+ 任务都逐项消融。

## 6. 如果要把 FlashSAC 稳定设计迁移到 UFO

建议做成新的 opt-in architecture/config，并保留当前实现作为基线；不要直接覆盖 `ResidualBlock`，否则旧 checkpoint 结构不兼容，也无法做可靠 A/B。

推荐依赖顺序：

1. **先补可观测性**：记录每个网络的 weight norm、hidden feature norm、parameter gradient norm、Q/target 范围和 nonfinite 位置。
2. **Inverted residual + post RMSNorm**：先对 actor、main/aux critic 做独立 backbone；F/B 的 matrix objective 需要单独判断如何映射。
3. **内部 BN + cross-batch 一起实现**：不能只替换 LayerNorm。双 critic 的张量是 `[ensemble, batch, feature]`，BN 必须只在正确的 batch 维统计，不能把 ensemble 当 batch/channel。
4. **Distributional critic + adaptive reward scaling 一起实现**：main/aux critic 从 scalar MSE 改为 categorical CE；fixed support 必须与 reward/return scale 联调。
5. **每步权重投影**：应发生在各 optimizer step 后、target EMA 前，并覆盖 Linear、parallel Linear、BN/RMSNorm 参数。
6. **逐项 A/B**：使用相同数据、seed、环境数、update schedule 与 action mapping；不要一次全部修改后把收益归因给某一个组件。

特别注意：UFO 四卡训练会平均各 rank 梯度和 floating buffers。若引入块内 BatchNorm，需要明确是 per-rank BN、同步 BN，还是像现有 floating-buffer 同步那样只同步 running statistics；三者训练语义不同。

另有一个迁移前应处理的配置问题：`FBcprAuxModel` 虽然声明了 `archi.aux_critic`，构建 auxiliary critic 时实际使用的是 `cfg.archi.critic`，见 [`fb_cpr_aux/model.py`](../humanoidverse/agents/fb_cpr_aux/model.py#L30-L39)。当前 main/aux 两套配置相同，所以现有行为没有差异；但以后单独修改 `aux_critic` 配置会被忽略。

## 7. 不应混为一谈的内容

- `compile/JIT`、CUDA graph、AMP 是性能/数值格式选项，不是 FlashSAC §4.2 的稳定组件。UFO 当前 FB preset 的 `amp=False`。
- 10M replay buffer、大 batch、低 UTD、少更新属于 FlashSAC §4.1 scaling。它们也会影响稳定性，但不是 §4.2 的网络/损失栈。
- unified entropy target 和 noise repetition 属于 §4.3 exploration。
- FlashSAC 的 learnable policy std/entropy temperature 与 UFO 的固定 `std=0.05` 不同，但这是 SAC/exploration 差异，不是本审计的 §4.2 核心。

## 8. 复现时的官方材料差异

截至本次审计，论文和官方配置存在两处值得记录的差异：

1. §4.1 正文/项目页描述每 1024 个新 transitions 做 2 次更新，而论文 Table 9 写 `UTD=2/2048`。
2. 论文 GPU Table 9 写 `n-step=1`，部分官方 GPU 启动配置使用 `n_step=3`。

因此严格论文复现应固定论文版本和超参表；复现官方代码行为则应固定 Git commit 和实际启动配置。本文关于 §4.2 稳定机制的有无判断不依赖这两处差异。

## 9. 最终判断

UFO 当前已经有一套自己的稳定措施，但 FlashSAC 的核心价值正是把 residual、内部 BatchNorm、cross-batch、post RMSNorm、distributional critic/reward scaling、每步权重投影组合成一个完整系统。UFO 目前只在“有 residual connection”和“也做某些归一化/双 critic”这一高层上相似，严格实现层面没有对齐。

如果后续目标是验证 FlashSAC 稳定栈对 UFO 是否有效，最可靠的下一步不是直接宣称替换完成，而是新增一个可选 backbone/critic 路径，按论文消融顺序做同组 A/B。
