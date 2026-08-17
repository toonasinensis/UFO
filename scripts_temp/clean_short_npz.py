#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量筛选并清理时长小于指定阈值的 .npz 文件。

默认只检查，不删除：
    python clean_short_npz.py /path/to/data

确认输出无误后，真正删除：
    python clean_short_npz.py /path/to/data --delete

例如删除小于 5 秒的数据：
    python clean_short_npz.py /path/to/data --min-duration 5 --delete
"""

import argparse
from pathlib import Path
import numpy as np


def get_npz_duration(npz_path: Path):
    """
    返回:
        duration: 时长（秒）
        num_frames: 帧数
        fps: 帧率

    针对当前数据格式：
        - fps 存在 npz["fps"]
        - 时间维通常是 data / joint_pos / body_pos_w 的第 0 维
        - duration = (num_frames - 1) / fps
    """
    with np.load(npz_path, allow_pickle=False) as f:
        if "fps" not in f:
            raise KeyError("缺少 'fps' 字段")

        fps = float(f["fps"])
        if fps <= 0:
            raise ValueError(f"非法 fps: {fps}")

        # 优先用 data；如果没有，再尝试其他常见时间序列字段
        frame_keys = [
            "data",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ]

        num_frames = None
        used_key = None

        for key in frame_keys:
            if key in f:
                arr = f[key]
                if arr.ndim >= 1:
                    num_frames = arr.shape[0]
                    used_key = key
                    break

        if num_frames is None:
            raise KeyError("找不到可用于计算帧数的数据字段")

        if num_frames <= 1:
            duration = 0.0
        else:
            duration = (num_frames - 1) / fps

        return duration, num_frames, fps, used_key


def main():
    parser = argparse.ArgumentParser(
        description="删除时长小于指定秒数的 NPZ 数据"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="包含 .npz 文件的数据文件夹"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="最小时长，单位秒；低于该值的文件会被筛掉，默认 5 秒"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="真正删除文件；不加此参数时仅预览"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子文件夹"
    )

    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"不是文件夹: {folder}")

    if args.recursive:
        npz_files = sorted(folder.rglob("*.npz"))
    else:
        npz_files = sorted(folder.glob("*.npz"))

    print(f"数据目录: {folder}")
    print(f"NPZ 数量: {len(npz_files)}")
    print(f"筛选阈值: < {args.min_duration:.3f} s")
    print(f"模式: {'真正删除' if args.delete else '仅预览，不删除'}")
    print("-" * 80)

    short_files = []
    kept_files = []
    error_files = []

    for npz_path in npz_files:
        try:
            duration, frames, fps, used_key = get_npz_duration(npz_path)

            if duration < args.min_duration:
                short_files.append(npz_path)

                print(
                    f"[SHORT] {npz_path.name} | "
                    f"{duration:.3f}s | frames={frames} | fps={fps:g} | key={used_key}"
                )

                if args.delete:
                    npz_path.unlink()
                    print(f"        -> 已删除")
            else:
                kept_files.append(npz_path)

        except Exception as e:
            error_files.append((npz_path, str(e)))
            print(f"[ERROR] {npz_path.name} | {e}")

    print("-" * 80)
    print(f"总文件数:       {len(npz_files)}")
    print(f"正常保留:       {len(kept_files)}")
    print(f"小于阈值:       {len(short_files)}")
    print(f"读取失败:       {len(error_files)}")

    if short_files and not args.delete:
        print()
        print("当前是预览模式，没有删除任何文件。")
        print("确认列表没问题后，增加 --delete 参数即可真正删除。")


if __name__ == "__main__":
    main()
