#!/usr/bin/env python3
"""
Pipeline V2 — Stage 2: Pose estimation on selected frames.
Reads frame images listed in selection_log.json, runs OpenPose on each,
and saves 25-joint Body25 keypoints to keypoints.json.

Preprocessing applied before OpenPose (vs Pipeline 1 baseline):
  1. ROI crop to YOLO cyclist bounding box
  2. CLAHE on LAB L-channel (lighting normalisation)
  3. Unsharp mask (counteracts lateral motion blur)
  4. Square-pad before inference (preserves aspect ratio)
  5. net_resolution=656x368 (better joint localisation on small subjects)

Keypoints are transformed back to original-frame coordinates so downstream
stages (seat_height, rpm, annotate_output) work without modification.

Usage: python3 pose_estimate.py <video_selection_log.json>
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
NET_RESOLUTION  = "656x368"

STEP_H      = 400   # height each panel is resized to in the montage
LABEL_H     = 28    # pixel height of the label bar beneath each panel
LABEL_FONT  = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.55
LABEL_THICK = 1

JOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]


def compute_roi(entry, frame_h, frame_w):
    """Return (x1, y1, x2, y2) crop region from the cyclist bounding box."""
    cx1, cy1, cx2, cy2 = entry["cyclist_box"]
    margin = int(0.05 * max(cx2 - cx1, cy2 - cy1))
    x1 = max(0, cx1 - margin)
    y1 = max(0, cy1 - margin)
    x2 = min(frame_w, cx2 + margin)
    y2 = min(frame_h, cy2 + margin)
    return x1, y1, x2, y2


def apply_clahe(img):
    """CLAHE on the L channel of LAB colour space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def apply_unsharp(img, strength=0.5):
    """Unsharp mask to counteract motion blur."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)


def square_pad(img):
    """Pad img to square with black bars. Returns (padded, pad_left, pad_top)."""
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
    """Resize img to STEP_H height and attach a white label bar below it."""
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
    """
    steps: list of (label, img) in pipeline order.
    Saves a horizontal strip with labelled panels to out_path.
    """
    panels = [make_panel(img, label) for label, img in steps]
    montage = np.hstack(panels)
    cv2.imwrite(out_path, montage)


def main():
    """Parse arguments, run OpenPose with preprocessing on selected frames, write keypoints.json."""
    if len(sys.argv) < 2:
        print("Usage: python3 pose_estimate.py <video_selection_log.json>")
        sys.exit(1)

    log_path = sys.argv[1]
    with open(log_path) as f:
        log = json.load(f)

    stem      = video_stem(log_path, "_selection_log")
    out_dir   = os.path.join("output_v2", stem)
    os.makedirs(out_dir, exist_ok=True)
    base      = os.path.join(out_dir, stem)
    out_path  = base + "_keypoints.json"
    steps_dir = base + "_preprocessing_steps"
    os.makedirs(steps_dir, exist_ok=True)

    params  = {"model_folder": OPENPOSE_MODELS, "net_resolution": NET_RESOLUTION}
    wrapper = op.WrapperPython()
    wrapper.configure(params)
    wrapper.start()
    print(f"[Pipeline V2] OpenPose ready (net_resolution={NET_RESOLUTION}). Processing {len(log['selected_frames'])} frames...")

    frames_out       = []
    inference_times  = []
    joint_conf_sums  = {name: 0.0 for name in JOINT_NAMES}
    joint_conf_count = 0

    for i, entry in enumerate(log["selected_frames"]):
        frame = cv2.imread(entry["frame_file"])
        if frame is None:
            print(f"  [WARN] Could not read {entry['frame_file']}, skipping")
            continue

        if not entry.get("cyclist_box"):
            print(f"  [WARN] No cyclist_box for frame {entry['frame_idx']}, skipping")
            continue

        fh, fw = frame.shape[:2]

        # Step 1 — original frame with cyclist box highlighted
        x1, y1, x2, y2 = compute_roi(entry, fh, fw)
        annotated = frame.copy()
        cx1, cy1, cx2, cy2 = entry["cyclist_box"]
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
        step_original = annotated

        # Step 2 — ROI crop (raw)
        crop_raw = frame[y1:y2, x1:x2].copy()

        # Step 3 — after CLAHE
        crop_clahe = apply_clahe(crop_raw)

        # Step 4 — after unsharp mask
        crop_sharp = apply_unsharp(crop_clahe)

        # Step 5 — square padded (what OpenPose sees)
        padded, pad_left, pad_top = square_pad(crop_sharp)

        save_step_montage(
            [
                ("1. Original + cyclist box", step_original),
                ("2. ROI crop",               crop_raw),
                ("3. CLAHE",                  crop_clahe),
                ("4. Unsharp mask",           crop_sharp),
                ("5. Square pad",             padded),
            ],
            os.path.join(steps_dir, f"frame_{entry['frame_idx']:06d}_steps.jpg"),
        )

        datum             = op.Datum()
        datum.cvInputData = padded
        t0                = time.time()
        wrapper.emplaceAndPop(op.VectorDatum([datum]))
        ms = (time.time() - t0) * 1000
        inference_times.append(ms)

        keypoints  = []
        joint_conf = {}
        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            kp = datum.poseKeypoints[0]
            for j, name in enumerate(JOINT_NAMES):
                px, py, c = float(kp[j][0]), float(kp[j][1]), float(kp[j][2])
                # Transform back: padded-square coords → crop coords → original-frame coords
                ox = px - pad_left + x1
                oy = py - pad_top  + y1
                keypoints.append([round(ox, 2), round(oy, 2), round(c, 4)])
                joint_conf[name] = round(c, 4)
                joint_conf_sums[name] += c
            joint_conf_count += 1

        frames_out.append({
            "frame_idx":         entry["frame_idx"],
            "timestamp":         entry["timestamp"],
            "frame_file":        entry["frame_file"],
            "inference_time_ms": round(ms, 1),
            "keypoints":         keypoints,
            "joint_confidences": joint_conf,
        })

        print_progress(i + 1, len(log["selected_frames"]), f"  {ms:.0f}ms")

    print()

    avg_infer = sum(inference_times) / len(inference_times) if inference_times else 0

    avg_joint_conf = (
        {name: round(joint_conf_sums[name] / joint_conf_count, 4) for name in JOINT_NAMES}
        if joint_conf_count > 0 else {}
    )

    output = {
        "video":  log["video"],
        "frames": frames_out,
        "metrics": {
            "frames_processed":      len(frames_out),
            "avg_inference_time_ms": round(avg_infer, 1),
            "avg_joint_confidence":  avg_joint_conf,
            "preprocessing": {
                "roi_crop":       True,
                "clahe":          True,
                "unsharp_mask":   True,
                "square_pad":     True,
                "net_resolution": NET_RESOLUTION,
            },
        }
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    m = output["metrics"]
    print(f"Frames processed   : {m['frames_processed']}")
    print(f"Avg inference time : {m['avg_inference_time_ms']}ms/frame")
    print(f"Step montages      → {steps_dir}/")
    print(f"Keypoints saved    → {out_path}")


if __name__ == "__main__":
    main()
