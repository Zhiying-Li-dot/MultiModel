#!/usr/bin/env python3
"""Create 3-way comparison: Original Video | Target Image | Generated Video"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Paths
BASE = Path("/Users/verypro/research/pvtt")
ORIG_VIDEO = BASE / "baseline/flowedit-wan2.1/videos/bracelet_shot1.mp4"
TARGET_IMAGE = BASE / "data/pvtt-benchmark/images/jewelry/JEWE005_source.jpg"
GEN_VIDEO = BASE / "experiments/results/compositional/target_video_v2.mp4"
OUTPUT = BASE / "experiments/results/compositional/comparison_3way.mp4"

# Target height for all
TARGET_H = 480

def add_label(frame, text):
    """Add text label to frame."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()

    # Draw with outline
    x, y = 10, 10
    for dx, dy in [(-2,0), (2,0), (0,-2), (0,2)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def resize_to_height(frame, target_h):
    """Resize frame to target height, maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h))

def main():
    # Open videos
    cap_orig = cv2.VideoCapture(str(ORIG_VIDEO))
    cap_gen = cv2.VideoCapture(str(GEN_VIDEO))

    # Load target image
    target_img = cv2.imread(str(TARGET_IMAGE))
    target_img = resize_to_height(target_img, TARGET_H)
    target_img = add_label(target_img, "Target Product")

    # Get video info
    orig_fps = cap_orig.get(cv2.CAP_PROP_FPS)
    gen_fps = cap_gen.get(cv2.CAP_PROP_FPS)
    gen_frames = int(cap_gen.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original: {orig_fps} fps")
    print(f"Generated: {gen_fps} fps, {gen_frames} frames")

    # Read first frames to get dimensions
    ret, orig_frame = cap_orig.read()
    ret, gen_frame = cap_gen.read()
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_gen.set(cv2.CAP_PROP_POS_FRAMES, 0)

    orig_frame = resize_to_height(orig_frame, TARGET_H)
    gen_frame = resize_to_height(gen_frame, TARGET_H)

    total_w = orig_frame.shape[1] + target_img.shape[1] + gen_frame.shape[1]
    print(f"Output size: {total_w}x{TARGET_H}")

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(OUTPUT).replace('.mp4', '.avi'), fourcc, 16, (total_w, TARGET_H))

    frame_idx = 0
    while frame_idx < gen_frames:
        ret_orig, orig_frame = cap_orig.read()
        ret_gen, gen_frame = cap_gen.read()

        if not ret_gen:
            break

        # If original video is shorter, loop it
        if not ret_orig:
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_orig, orig_frame = cap_orig.read()

        # Resize
        orig_frame = resize_to_height(orig_frame, TARGET_H)
        gen_frame = resize_to_height(gen_frame, TARGET_H)

        # Add labels
        orig_labeled = add_label(orig_frame, "Original Video")
        gen_labeled = add_label(gen_frame, "Generated Video")

        # Combine horizontally
        combined = np.hstack([orig_labeled, target_img, gen_labeled])
        out.write(combined)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{gen_frames}")

    cap_orig.release()
    cap_gen.release()
    out.release()

    print(f"\nSaved to: {str(OUTPUT).replace('.mp4', '.avi')}")

if __name__ == "__main__":
    main()
