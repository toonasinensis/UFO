# Roban 训练

当前正式训练使用 10s 数据，wandb entity 固定为 `xiechunyang1-hajimi`，group 放到 `ufo_fb_g1`。

远端启动命令：

```bash
cd /home/xiechunyang/wt_ws/wt_wbc/UFO
mkdir -p runs/roban_s22_fb_10s_gpu0_20260727_1719
nohup env \
  WANDB_ENTITY=xiechunyang1-hajimi \
  WANDB_PROJECT=hajimi \
  WANDB_GROUP=ufo_fb_g1 \
  UFO_PYTHON=.venv/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
  ./run_train.sh \
    --agent fb \
    --gpu-ids single \
    --data-manifest configs/data/roban_s22.yaml \
    --work-dir runs/roban_s22_fb_10s_gpu0_why_eval_g \
    --use-wandb \
    --wandb-run-name roban_s22_fb_10s_gpu0_why_eval_g \
  > runs/roban_s22_fb_10s_gpu0_20260727_1719/server_train.log 2>&1 < /dev/null &




#本地训练命令
cd /home/xiechunyang/wt_ws/wt_wbc/UFO

env \
  WANDB_ENTITY=xiechunyang1-hajimi \
  WANDB_PROJECT=hajimi \
  WANDB_GROUP=ufo_fb_g1 \
  UFO_PYTHON=.venv/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
  ./run_train.sh \
    --agent fb \
    --gpu-ids single \
    --data-manifest configs/data/roban_s22.yaml \
    --work-dir runs/roban_s22_fb_10s_gpu0_why_eval_g \
    --use-wandb \
    --wandb-run-name roban_s22_fb_10s_gpu0_why_eval_g
```


wandb:

```text
https://wandb.ai/xiechunyang1-hajimi/hajimi/runs/i2vhz857
```

# Roban play（本地实时窗口）

这是用下载到本地的 checkpoint 实时播放，不保存 mp4。

```bash


cd /home/thl/wt_wbc/UFO
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m humanoidverse.tracking_inference \
  --model-folder ./runs/biped_17_torso_8kg_20260728_144404 \
  --data-manifest configs/data/roban_s22_play.yaml \
  --dataset roban_lafan \
  --device cpu \
  --rebuild-motion-cache \
  --headless false \
  --disable-dr false \
  --live-view true \
  --show-reference true \
  --save-mp4 false \
  --motion-list 0 \
  --max-steps 40000 \
  --log-every-steps 50 \
  --export-onnx true


```
