#!/usr/bin/env python3
"""Create 4-way comparison: Original | Flux Generated | Pure FlowEdit | Compositional (TI2V)"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os

# Server paths
HOME = os.path.expanduser("~")
BASE = Path(HOME) / "pvtt"
ORIG_VIDEO = BASE / "baseline/flowedit-wan2.1/videos/bracelet_shot1.mp4"
FLUX_FRAME = BASE / "experiments/results/compositional/target_frame1.png"
FLOWEDIT_VIDEO = BASE / "experiments/results/compositional/test02_pure_flowedit_pearl.mp4"
TI2V_VIDEO = BASE / "experiments/results/compositional/target_video_v2.mp4"
OUTPUT = BASE / "experiments/results/compositional/comparison_4way.avi"

TARGET_H = 480

def add_label(frame, text):
    """Add text label to frame."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

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
    print(f"Original video: {ORIG_VIDEO}")
    print(f"Flux frame: {FLUX_FRAME}")
    print(f"FlowEdit video: {FLOWEDIT_VIDEO}")
    print(f"TI2V video: {TI2V_VIDEO}")

    # Open videos
    cap_orig = cv2.VideoCapture(str(ORIG_VIDEO))
    cap_flowedit = cv2.VideoCapture(str(FLOWEDIT_VIDEO))
    cap_ti2v = cv2.VideoCapture(str(TI2V_VIDEO))

    # Load Flux generated frame (static image)
    flux_frame = cv2.imread(str(FLUX_FRAME))
    flux_frame = resize_to_height(flux_frame, TARGET_H)
    flux_frame = add_label(flux_frame, "Flux.2 Frame")

    # Get video info
    flowedit_frames = int(cap_flowedit.get(cv2.CAP_PROP_FRAME_COUNT))
    ti2v_frames = int(cap_ti2v.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = max(flowedit_frames, ti2v_frames)

    print(f"FlowEdit frames: {flowedit_frames}")
    print(f"TI2V frames: {ti2v_frames}")
    print(f"Using {max_frames} frames")

    # Read first frames to get dimensions
    ret, orig_frame = cap_orig.read()
    ret, flowedit_frame = cap_flowedit.read()
    ret, ti2v_frame = cap_ti2v.read()
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_flowedit.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_ti2v.set(cv2.CAP_PROP_POS_FRAMES, 0)

    orig_frame = resize_to_height(orig_frame, TARGET_H)
    flowedit_frame = resize_to_height(flowedit_frame, TARGET_H)
    ti2v_frame = resize_to_height(ti2v_frame, TARGET_H)

    # Calculate total width (2x2 grid layout)
    frame_w = orig_frame.shape[1]
    total_w = frame_w * 2
    total_h = TARGET_H * 2

    print(f"Output size: {total_w}x{total_h}")

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(OUTPUT), fourcc, 16, (total_w, total_h))

    frame_idx = 0
    while frame_idx < max_frames:
        ret_orig, orig_frame = cap_orig.read()
        ret_flowedit, flowedit_frame = cap_flowedit.read()
        ret_ti2v, ti2v_frame = cap_ti2v.read()

        # Loop original video if shorter
        if not ret_orig:
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_orig, orig_frame = cap_orig.read()

        # Handle end of FlowEdit video
        if not ret_flowedit:
            cap_flowedit.set(cv2.CAP_PROP_POS_FRAMES, flowedit_frames - 1)
            ret_flowedit, flowedit_frame = cap_flowedit.read()

        # Handle end of TI2V video
        if not ret_ti2v:
            cap_ti2v.set(cv2.CAP_PROP_POS_FRAMES, ti2v_frames - 1)
            ret_ti2v, ti2v_frame = cap_ti2v.read()

        # Resize
        orig_frame = resize_to_height(orig_frame, TARGET_H)
        flowedit_frame = resize_to_height(flowedit_frame, TARGET_H)
        ti2v_frame = resize_to_height(ti2v_frame, TARGET_H)

        # Ensure all frames have same width
        target_w = frame_w
        orig_frame = cv2.resize(orig_frame, (target_w, TARGET_H))
        flowedit_frame = cv2.resize(flowedit_frame, (target_w, TARGET_H))
        ti2v_frame = cv2.resize(ti2v_frame, (target_w, TARGET_H))
        flux_resized = cv2.resize(flux_frame, (target_w, TARGET_H))

        # Add labels
        orig_labeled = add_label(orig_frame, "Original")
        flowedit_labeled = add_label(flowedit_frame, "Pure FlowEdit")
        ti2v_labeled = add_label(ti2v_frame, "Flux + TI2V")

        # 2x2 grid: [Original | Flux Frame]
        #           [FlowEdit | TI2V      ]
        top_row = np.hstack([orig_labeled, flux_resized])
        bottom_row = np.hstack([flowedit_labeled, ti2v_labeled])
        combined = np.vstack([top_row, bottom_row])

        out.write(combined)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{max_frames}")

    cap_orig.release()
    cap_flowedit.release()
    cap_ti2v.release()
    out.release()

    print(f"\nSaved to: {OUTPUT}")

if __name__ == "__main__":
    main()
