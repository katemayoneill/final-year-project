#!/usr/bin/env python3
"""
stage 2: pose estimation on selected frames.

usage: python3 pose_estimate.py <video_selection_log.json>
Output: output_v2/<stem>/<stem>_keypoints.json
         output_v2/<stem>/<stem>_preprocessing_steps/
"""
import json
import os
import sys
import time

import cv2
import numpy as np

from utils import print_progress, video_stem

try:
    from openpose import pyopenpose as op
except ImportError:
    import pyopenpose as op

OPENPOSE_MODELS = os.environ.get("OPENPOSE_MODELS", "/openpose/models/")
NET_RESOLUTION = "656x368"

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

UNSHARP_SIGMA = 3
UNSHARP_STRENGTH = 0.5

ROI_MARGIN_FRAC = 0.05

APPROX_HW = 2.0

STEP_H = 400
LABEL_H = 28 
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.55
LABEL_THICK = 1

COLOUR_CYCLIST = (0, 255, 0)
COLOUR_NO_CYCLIST = (0, 0, 255)


JOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]


def median_cyclist_hw(frames):
    """return median height/width ratio from frames that have a cyclist."""
    ratios = []
    for f in frames:
        box = f.get("cyclist_box")
        if box:
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w > 0:
                ratios.append(h / w)
    if not ratios:
        return APPROX_HW
    return sorted(ratios)[len(ratios) // 2]


def estimate_cyclist_box(entry, aspect):
    """estimate cyclist box from wheel positions when yolo fails to detect cyclist."""
    fw_box = entry["fw_box"]
    bw_box = entry["bw_box"]
    x_left = min(fw_box[0], bw_box[0])
    x_right = max(fw_box[2], bw_box[2])
    wheel_bottom = max(fw_box[3], bw_box[3])
    width = x_right - x_left
    height = int(width * aspect)
    return [x_left, wheel_bottom - height, x_right, wheel_bottom]


def compute_roi(box, frame_h, frame_w):
    """return padded crop region for a cyclist box."""
    cx1, cy1, cx2, cy2 = box
    margin = int(ROI_MARGIN_FRAC * max(cx2 - cx1, cy2 - cy1))
    x1 = max(0, cx1 - margin)
    y1 = max(0, cy1 - margin)
    x2 = min(frame_w, cx2 + margin)
    y2 = min(frame_h, cy2 + margin)
    return x1, y1, x2, y2


def apply_clahe(img):
    """apply CLAHE to LAB L-channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def apply_unsharp(img, strength=UNSHARP_STRENGTH):
    """apply unsharp mask."""
    blur = cv2.GaussianBlur(img, (0, 0), UNSHARP_SIGMA)
    return cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)


def square_pad(img):
    """pad img to a square."""
    h, w = img.shape[:2]
    s = max(h, w)
    pad_top  = (s - h) // 2
    pad_left = (s - w) // 2
    padded = cv2.copyMakeBorder(
        img, pad_top, s - h - pad_top, pad_left, s - w - pad_left,
        cv2.BORDER_CONSTANT, value=0,
    )
    return padded, pad_left, pad_top


def make_panel(img, label):
    """return panel image used in preprocessin montage."""
    h, w = img.shape[:2]
    new_w = max(1, int(w * STEP_H / h))
    panel = cv2.resize(img, (new_w, STEP_H), interpolation=cv2.INTER_AREA)
    bar = np.full((LABEL_H, new_w, 3), 240, dtype=np.uint8)
    tw, th = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICK)[0]
    tx = max(0, (new_w - tw) // 2)
    ty = (LABEL_H + th) // 2
    cv2.putText(bar, label, (tx, ty), LABEL_FONT, LABEL_SCALE, (30, 30, 30), LABEL_THICK, cv2.LINE_AA)
    return np.vstack([panel, bar])


def save_step_montage(steps, out_path):
    """save horizontal strip of labelled preprocessing panels to out path."""
    panels = [make_panel(img, label) for label, img in steps]
    montage = np.hstack(panels)
    cv2.imwrite(out_path, montage)


def main():
    """run OpenPose with preprocessing on selected frames; write keypoints.json."""
    if len(sys.argv) < 2:
        print("usage: python3 pose_estimate.py <video_selection_log.json>")
        sys.exit(1)

    log_path = sys.argv[1]
    with open(log_path) as f:
        log = json.load(f)

    stem = video_stem(log_path, "_selection_log")
    out_dir = os.path.join("output_v2", stem)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, stem)
    out_path = base + "_keypoints.json"
    steps_dir = base + "_preprocessing_steps"
    os.makedirs(steps_dir, exist_ok=True)

    params  = {"model_folder": OPENPOSE_MODELS, "net_resolution": NET_RESOLUTION}
    wrapper = op.WrapperPython()
    wrapper.configure(params)
    wrapper.start()
    all_entries = log["selected_frames"]
    approx_aspect = median_cyclist_hw(all_entries)
    print(f"OpenPose ready (net_resolution={NET_RESOLUTION}). processing {len(all_entries)} frames....")
    print(f"fallback cyclist aspect ratio (h/w): {approx_aspect:.2f}")

    frames_out = []
    inference_times = []
    joint_conf_sums = {name: 0.0 for name in JOINTS}
    joint_conf_count = 0

    for i, entry in enumerate(all_entries):
        frame = cv2.imread(entry["frame_file"])
        if frame is None:
            print(f"[SKIP] couldn't read {entry['frame_file']}")
            continue

        fh, fw = frame.shape[:2]

        has_cyclist_box = bool(entry.get("cyclist_box"))
        if not has_cyclist_box:
            print(f"[NO CYCLIST] no cyclist_box for frame {entry['frame_idx']}, estimating from wheel boxes")
        cyclist_box = entry["cyclist_box"] if has_cyclist_box else estimate_cyclist_box(entry, approx_aspect)

        x1, y1, x2, y2 = compute_roi(cyclist_box, fh, fw)
        annotated = frame.copy()
        cx1, cy1, cx2, cy2 = cyclist_box
        box_colour  =  COLOUR_CYCLIST if has_cyclist_box else COLOUR_NO_CYCLIST
        step1_label = "1. original + cyclist box" if has_cyclist_box else "1. original + estimated box"
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), box_colour, 3)
        step_original = annotated

        crop = frame[y1:y2, x1:x2].copy()

        crop_clahe = apply_clahe(crop)

        crop_sharp = apply_unsharp(crop_clahe)

        padded, pad_left, pad_top = square_pad(crop_sharp)

        save_step_montage(
            [
                (step1_label, step_original),
                ("2. ROI crop", crop),
                ("3. CLAHE", crop_clahe),
                ("4. Unsharp mask", crop_sharp),
                ("5. Square pad", padded),
            ],
            os.path.join(steps_dir, f"frame_{entry['frame_idx']:06d}_steps.jpg"),
        )

        datum = op.Datum()
        datum.cvInputData = padded
        t0 = time.time()
        wrapper.emplaceAndPop(op.VectorDatum([datum]))
        ms = (time.time() - t0) * 1000
        inference_times.append(ms)

        keypoints = []
        joint_conf = {}
        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            kp = datum.poseKeypoints[0]

            for j, name in enumerate(JOINTS):
                px, py, c = float(kp[j][0]), float(kp[j][1]), float(kp[j][2])
                ox = px - pad_left + x1
                oy = py - pad_top  + y1
                keypoints.append([round(ox, 2), round(oy, 2), round(c, 4)])
                joint_conf[name] = round(c, 4)
                joint_conf_sums[name] += c
            joint_conf_count += 1

        frames_out.append({
            "frame_idx": entry["frame_idx"],
            "timestamp": entry["timestamp"],
            "frame_file": entry["frame_file"],
            "inference_time_ms": round(ms, 1),
            "keypoints": keypoints,
            "joint_confidences": joint_conf,
        })

        print_progress(i + 1, len(log["selected_frames"]), f"  {ms:.0f}ms")

    print()

    avg_infer = sum(inference_times) / len(inference_times) if inference_times else 0

    avg_joint_conf = (
        {name: round(joint_conf_sums[name] / joint_conf_count, 4) for name in JOINTS}
        if joint_conf_count > 0 else {}
    )

    output = {
        "video": log["video"],
        "frames": frames_out,
        "metrics": {
            "frames_processed": len(frames_out),
            "avg_inference_time_ms": round(avg_infer, 1),
            "avg_joint_confidence": avg_joint_conf,
            "preprocessing": {
                "roi_crop": True,
                "clahe": True,
                "unsharp_mask": True,
                "square_pad": True,
                "net_resolution": NET_RESOLUTION,
            },
        }
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    m = output["metrics"]
    print(f"frames processed: {m['frames_processed']}")
    print(f"averag inference time: {m['avg_inference_time_ms']}ms/frame")
    print(f"step montages: {steps_dir}/")
    print(f"keypoints saved: {out_path}")


if __name__ == "__main__":
    main()
