#!/usr/bin/env python3
"""
提取视频第一帧
"""

import argparse
from pathlib import Path

import cv2


def extract_first_frame(video_path: str, output_path: str) -> bool:
    """
    从视频中提取第一帧。

    Args:
        video_path: 输入视频路径
        output_path: 输出图片路径

    Returns:
        是否成功
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Cannot read first frame")
        return False

    # 保存图片
    cv2.imwrite(output_path, frame)
    print(f"Extracted first frame to {output_path}")

    # 打印信息
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract first frame from video")
    parser.add_argument("--video", "-v", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", required=True, help="输出图片路径")

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = extract_first_frame(args.video, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
