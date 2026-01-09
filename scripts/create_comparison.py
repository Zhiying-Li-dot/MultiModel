#!/usr/bin/env python3
"""Create 4-way comparison video: Source | Reference | Baseline | RefDrop"""

from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np
from pathlib import Path

# Paths
BASE = Path("/Users/verypro/research/pvtt")
SOURCE_VIDEO = BASE / "experiments/results/bracelet_shot1_480p.mp4"
REF_IMAGE = BASE / "data/pvtt-benchmark/images/jewelry/JEWE005_source.jpg"
BASELINE_VIDEO = BASE / "experiments/results/flowedit-wan2.1/test02_flowedit_pearl_baseline.mp4"
REFDROP_VIDEO = BASE / "experiments/results/flowedit-wan2.1/test02_flowedit_pearl_c0.05.mp4"
OUTPUT = BASE / "experiments/results/flowedit-wan2.1/test02_4way_comparison.mp4"

def load_video_frames(video_path):
    """Load all frames from a video using imageio."""
    reader = imageio.get_reader(str(video_path))
    frames = [Image.fromarray(frame) for frame in reader]
    reader.close()
    return frames

def resize_to_height(img, target_height):
    """Resize image maintaining aspect ratio to match height."""
    w, h = img.size
    scale = target_height / h
    new_w = int(w * scale)
    return img.resize((new_w, target_height), Image.LANCZOS)

def add_label(img, label):
    """Add text label to image at top center."""
    draw = ImageDraw.Draw(img)

    # Use default font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()

    # Get text size
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position at top center
    x = (img.width - text_w) // 2
    y = 10

    # Draw background
    padding = 5
    draw.rectangle(
        [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
        fill=(0, 0, 0)
    )

    # Draw text
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return img

def main():
    print("Creating 4-way comparison video...")

    print("Loading source video...")
    source_frames = load_video_frames(SOURCE_VIDEO)
    print(f"  Got {len(source_frames)} frames")

    print("Loading baseline video...")
    baseline_frames = load_video_frames(BASELINE_VIDEO)
    print(f"  Got {len(baseline_frames)} frames")

    print("Loading refdrop video...")
    refdrop_frames = load_video_frames(REFDROP_VIDEO)
    print(f"  Got {len(refdrop_frames)} frames")

    # Load reference image
    ref_img = Image.open(REF_IMAGE)
    print(f"Reference image size: {ref_img.size}")

    # Get target dimensions
    target_height = baseline_frames[0].height
    num_frames = min(len(source_frames), len(baseline_frames), len(refdrop_frames))
    print(f"Target height: {target_height}, frames to process: {num_frames}")

    # Resize reference image
    ref_resized = resize_to_height(ref_img, target_height)
    ref_labeled = add_label(ref_resized.copy(), "Reference")

    # Get widths
    src_sample = resize_to_height(source_frames[0], target_height)
    total_width = src_sample.width + ref_labeled.width + baseline_frames[0].width + refdrop_frames[0].width

    print(f"Output size: {total_width}x{target_height}")
    print("Creating combined frames...")

    # Create output video
    writer = imageio.get_writer(
        str(OUTPUT),
        fps=16,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'
    )

    for i in range(num_frames):
        # Get and resize source
        src_frame = resize_to_height(source_frames[i], target_height)
        src_labeled = add_label(src_frame.copy(), "Source")

        # Baseline and refdrop
        base_labeled = add_label(baseline_frames[i].copy(), "Baseline")
        refdrop_labeled = add_label(refdrop_frames[i].copy(), "RefDrop c=0.05")

        # Combine horizontally
        combined = Image.new('RGB', (total_width, target_height))
        x_offset = 0
        for img in [src_labeled, ref_labeled, base_labeled, refdrop_labeled]:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width

        # Write frame
        writer.append_data(np.array(combined))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_frames}")

    writer.close()
    print(f"\nSaved to: {OUTPUT}")

if __name__ == "__main__":
    main()
