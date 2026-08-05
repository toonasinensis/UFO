# Codex / 训练操作约束

本文件是给后续 Codex 会话看的项目级操作说明。处理训练任务前先检查这里。

## 强制规则：每次训练重建 motion cache

所有使用 `--data-manifest` 的训练命令都必须显式包含：

```bash
--rebuild-motion-cache
```

原因：当前 manifest cache 不记录源 NPZ 文件列表、数量、mtime 或内容哈希。只要同名
PKL 存在，源数据即使由 1307 条改成 1222 条也会复用旧缓存。

`run_train.sh` 已加入兜底：检测到 `--data-manifest` 且命令未写该参数时，会自动添加
`--rebuild-motion-cache`。即便如此，Codex 在生成、记录或执行训练命令时仍必须把参数
明确写出来，便于审计。

Roban 训练还必须显式指定当前数据目录，不能依赖旧 BFM 目录的默认值：

```bash
env \
  UFO_ROBAN_MOTION_DIR=/home/thl/wt_wbc/UFO/humanoidverse/data/roban/named_roban_lafan_10s \
  ./run_train.sh \
    --data-manifest configs/data/roban_s22.yaml \
    --rebuild-motion-cache \
    ...
```

启动训练前必须在日志中核对：

- `path=` 指向本次指定的数据目录；
- `motions=` 等于源目录实际 NPZ 数量；
- 本次启动不得显示旧缓存的 `(reused)`。

## Roban 随机化

Roban 随机化集中配置在：

```text
humanoidverse/data/robots/roban_s22_handball/roban_s22_handball/config/domain_randomization.yaml
```

配置中的 `enabled: true` 是总开关；`--disable-dr` 始终具有最高优先级。不要把 Roban
body 名称和随机范围散落或硬编码进环境代码。
