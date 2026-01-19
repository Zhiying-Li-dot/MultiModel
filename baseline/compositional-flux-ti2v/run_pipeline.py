#!/usr/bin/env python3
"""
PVTT 组合式方法完整 Pipeline

Stage 1: Flux.2 Dev Edit (图像编辑) - 本地运行
Stage 2: Wan2.1 TI2V (视频生成) - 5090 服务器运行
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_stage1_extract_frame(config: dict, config_dir: Path) -> str:
    """提取模板视频第一帧"""
    video_path = config_dir / config["template"]["video_path"]
    output_path = config_dir / "results" / "template_frame1.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 0: 提取模板视频第一帧")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(config_dir / "scripts" / "extract_frame.py"),
        "--video", str(video_path),
        "--output", str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Failed to extract first frame")

    return str(output_path)


def run_stage1_flux_edit(config: dict, config_dir: Path, template_frame: str) -> str:
    """Stage 1: Flux.2 Dev Edit 图像编辑"""
    product_image = config_dir / config["product"]["image_path"]
    output_path = config_dir / config["output"]["first_frame"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flux_config = config.get("flux_edit", {})

    print("\n" + "=" * 60)
    print("Stage 1: Flux.2 Dev Edit 图像编辑")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(config_dir / "scripts" / "flux_edit.py"),
        "--template-frame", template_frame,
        "--product-image", str(product_image),
        "--source-object", config["template"]["source_object"],
        "--target-object", config["product"]["target_object"],
        "--output", str(output_path),
        "--guidance-scale", str(flux_config.get("guidance_scale", 3.5)),
        "--steps", str(flux_config.get("num_inference_steps", 28)),
        "--width", str(flux_config.get("width", 832)),
        "--height", str(flux_config.get("height", 480)),
    ]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Stage 1 failed")

    return str(output_path)


def run_stage2_ti2v(config: dict, config_dir: Path, first_frame: str) -> str:
    """Stage 2: Wan2.1 TI2V 视频生成 (需要在 5090 服务器运行)"""
    output_path = config_dir / config["output"]["video"]
    ti2v_config = config.get("ti2v", {})

    print("\n" + "=" * 60)
    print("Stage 2: Wan2.1 TI2V 视频生成")
    print("=" * 60)

    print(f"\n请在 5090 服务器上运行以下命令：")
    print("-" * 60)

    # 构建 prompt
    motion_prompt = ti2v_config.get("motion_prompt", "")
    full_prompt = f"{config['product']['target_prompt']} {motion_prompt}"

    print(f"""
# 1. 上传第一帧到服务器
scp {first_frame} 5090:~/pvtt/inputs/

# 2. 在服务器上运行 TI2V
ssh 5090
cd ~/pvtt/baseline/compositional-flux-ti2v
python scripts/ti2v_generate.py \\
    --first-frame ~/pvtt/inputs/{Path(first_frame).name} \\
    --prompt "{full_prompt}" \\
    --output {output_path} \\
    --num-frames {ti2v_config.get('num_frames', 49)} \\
    --guidance-scale {ti2v_config.get('guidance_scale', 5.0)}

# 3. 下载结果
scp 5090:{output_path} {output_path}
""")
    print("-" * 60)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="PVTT Compositional Pipeline")
    parser.add_argument("--config", "-c", required=True, help="配置文件路径")
    parser.add_argument("--stage1-only", action="store_true", help="只运行 Stage 1")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1

    config = load_config(config_path)
    config_dir = Path(__file__).parent

    try:
        # Stage 0: 提取第一帧
        template_frame = run_stage1_extract_frame(config, config_dir)

        # Stage 1: Flux.2 图像编辑
        target_frame = run_stage1_flux_edit(config, config_dir, template_frame)

        print(f"\n✅ Stage 1 完成！目标第一帧: {target_frame}")

        if not args.stage1_only:
            # Stage 2: TI2V 视频生成 (打印指令)
            run_stage2_ti2v(config, config_dir, target_frame)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
