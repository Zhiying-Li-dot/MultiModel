#!/usr/bin/env python3
"""
镜头检测工具

使用 PySceneDetect 检测视频中的镜头切换。

安装依赖:
    pip install scenedetect[opencv]

用法:
    python detect_shots.py video.mp4
    python detect_shots.py video.mp4 --output shots.json --threshold 27.0
"""

import argparse
import json
from pathlib import Path

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
except ImportError:
    print("请安装 scenedetect: pip install scenedetect[opencv]")
    exit(1)


def detect_shots(video_path: str, threshold: float = 27.0) -> list:
    """检测视频中的镜头边界

    Args:
        video_path: 视频文件路径
        threshold: 检测阈值，越低越敏感 (默认 27.0)

    Returns:
        镜头列表，每个镜头包含起止帧和时间
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # 检测镜头
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    # 获取视频信息
    fps = video.frame_rate
    total_frames = video.duration.get_frames()
    duration = video.duration.get_seconds()

    # 转换为结构化数据
    shots = []
    for i, (start, end) in enumerate(scene_list):
        shots.append({
            "shot_id": i + 1,
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames() - 1,  # 包含最后一帧
            "start_time": round(start.get_seconds(), 2),
            "end_time": round(end.get_seconds(), 2),
            "duration": round(end.get_seconds() - start.get_seconds(), 2),
        })

    return {
        "video_path": video_path,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration": round(duration, 2),
        "threshold": threshold,
        "total_shots": len(shots),
        "shots": shots,
    }


def main():
    parser = argparse.ArgumentParser(description="检测视频镜头边界")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--output", "-o", help="输出 JSON 文件")
    parser.add_argument("--threshold", "-t", type=float, default=27.0,
                        help="检测阈值 (默认 27.0，越低越敏感)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        return 1

    print(f"Detecting shots in {video_path.name}...")
    result = detect_shots(str(video_path), args.threshold)

    # 打印结果
    print(f"\nFound {result['total_shots']} shot(s):")
    print("-" * 60)
    for shot in result["shots"]:
        print(f"  Shot {shot['shot_id']}: "
              f"{shot['start_time']:.1f}s - {shot['end_time']:.1f}s "
              f"({shot['duration']:.1f}s, frames {shot['start_frame']}-{shot['end_frame']})")
    print("-" * 60)

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")
    else:
        print("\nJSON output:")
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
