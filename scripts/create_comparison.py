#!/usr/bin/env python3
"""Create 4-way comparison video: Source | Reference | Clean RefDrop | Noisy RefDrop"""

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path

# Paths
BASE = Path("/Users/verypro/research/pvtt")
SOURCE_VIDEO = BASE / "baseline/flowedit-wan2.1/videos/bracelet_shot1.mp4"
REF_IMAGE = BASE / "data/pvtt-benchmark/images/jewelry/JEWE005_source.jpg"
CLEAN_VIDEO = BASE / "experiments/results/flowedit-wan2.1/test02_flowedit_pearl_c0.05.mp4"
NOISY_VIDEO = BASE / "experiments/results/flowedit-wan2.1/test02_noisy_refdrop_c0.05.mp4"
OUTPUT = BASE / "experiments/results/flowedit-wan2.1/test02_clean_vs_noisy_comparison.mp4"

def load_video_frames(video_path):
    """Load all frames from a video using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
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
    print("Creating 4-way comparison video: Source | Target | Clean | Noisy")

    print("Loading source video...")
    source_frames = load_video_frames(SOURCE_VIDEO)
    print(f"  Got {len(source_frames)} frames")

    print("Loading clean refdrop video...")
    clean_frames = load_video_frames(CLEAN_VIDEO)
    print(f"  Got {len(clean_frames)} frames")

    print("Loading noisy refdrop video...")
    noisy_frames = load_video_frames(NOISY_VIDEO)
    print(f"  Got {len(noisy_frames)} frames")

    # Load reference image
    ref_img = Image.open(REF_IMAGE)
    print(f"Reference image size: {ref_img.size}")

    # Get target dimensions
    target_height = clean_frames[0].height
    num_frames = min(len(source_frames), len(clean_frames), len(noisy_frames))
    print(f"Target height: {target_height}, frames to process: {num_frames}")

    # Resize reference image
    ref_resized = resize_to_height(ref_img, target_height)
    ref_labeled = add_label(ref_resized.copy(), "Target Image")

    # Get widths
    src_sample = resize_to_height(source_frames[0], target_height)
    total_width = src_sample.width + ref_labeled.width + clean_frames[0].width + noisy_frames[0].width

    print(f"Output size: {total_width}x{target_height}")
    print("Creating combined frames...")

    # Create output video with OpenCV (use avc1 codec for better compatibility)
    # Try different codecs for Mac compatibility
    output_avi = str(OUTPUT).replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_avi, fourcc, 16, (total_width, target_height))

    for i in range(num_frames):
        # Get and resize source
        src_frame = resize_to_height(source_frames[i], target_height)
        src_labeled = add_label(src_frame.copy(), "Source")

        # Clean and noisy refdrop
        clean_labeled = add_label(clean_frames[i].copy(), "Clean RefDrop")
        noisy_labeled = add_label(noisy_frames[i].copy(), "Noisy RefDrop")

        # Combine horizontally
        combined = Image.new('RGB', (total_width, target_height))
        x_offset = 0
        for img in [src_labeled, ref_labeled, clean_labeled, noisy_labeled]:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width

        # Write frame (convert RGB to BGR for OpenCV)
        frame_bgr = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_frames}")

    writer.release()
    print(f"\nSaved to: {output_avi}")

if __name__ == "__main__":
    main()
