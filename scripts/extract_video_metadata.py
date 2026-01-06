#!/usr/bin/env python3
"""
视频元数据提取工具

用法:
    python extract_video_metadata.py video.mp4
    python extract_video_metadata.py videos/ --output metadata.json
"""

import argparse
import json
import subprocess
from pathlib import Path


def extract_metadata(video_path: str) -> dict:
    """使用 ffprobe 提取视频元数据"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    # 找到视频流
    video_stream = None
    for stream in data.get("streams", []):
        if stream["codec_type"] == "video":
            video_stream = stream
            break

    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    # 提取关键信息
    width = video_stream["width"]
    height = video_stream["height"]

    # 解析帧率 (可能是 "24/1" 或 "24000/1001" 格式)
    fps_str = video_stream.get("r_frame_rate", "24/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    # 时长
    duration = float(data["format"].get("duration", 0))

    # 总帧数
    total_frames = int(video_stream.get("nb_frames", int(duration * fps)))

    return {
        "path": video_path,
        "duration_sec": round(duration, 2),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "codec": video_stream.get("codec_name", "unknown"),
        "bitrate": data["format"].get("bit_rate", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description="提取视频元数据")
    parser.add_argument("input", help="视频文件或目录")
    parser.add_argument("--output", "-o", help="输出 JSON 文件")
    parser.add_argument("--recursive", "-r", action="store_true", help="递归搜索目录")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # 单个文件
        metadata = extract_metadata(str(input_path))
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

    elif input_path.is_dir():
        # 目录
        pattern = "**/*.mp4" if args.recursive else "*.mp4"
        videos = list(input_path.glob(pattern))

        results = []
        for video in sorted(videos):
            try:
                metadata = extract_metadata(str(video))
                results.append(metadata)
                print(f"[OK] {video.name}: {metadata['resolution']}, {metadata['duration_sec']}s")
            except Exception as e:
                print(f"[FAIL] {video.name}: {e}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nSaved {len(results)} entries to {args.output}")
    else:
        print(f"Error: {input_path} not found")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
