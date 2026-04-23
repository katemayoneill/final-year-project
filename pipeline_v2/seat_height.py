#!/usr/bin/env python3
"""
Pipeline V2 — Stage 4: Seat height assessment from keypoints.
Per-frame: computes Hip→Knee→Ankle and Shoulder→Hip→Knee angles for both sides.
Summary peak: mean of validated bottom-of-stroke angles from knee_analysis.json
(Stage 3). Falls back to max(knee_angles) if fewer than 2 peaks were detected.

Optimal knee angle at bottom of stroke: 145–155 degrees
  too_low  : peak < 145 (seat too low — power loss, knee stress)
  optimal  : 145 <= peak <= 155
  too_high : peak > 155 (seat too high — over-extension risk)

Usage: python3 seat_height.py <video_keypoints.json>
Output: output_v2/<stem>/<stem>_assessment.json
"""
import json
import math
import os
import sys

from utils import calc_angle, video_stem

KNEE_OPTIMAL_LOW  = 145.0
KNEE_OPTIMAL_HIGH = 155.0
CONF_MIN          = 0.1

JOINT = {
    "Nose": 0, "Neck": 1,
    "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7,
    "MidHip": 8,
    "RHip": 9,  "RKnee": 10, "RAnkle": 11,
    "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18,
    "LBigToe": 19, "LSmallToe": 20, "LHeel": 21,
    "RBigToe": 22, "RSmallToe": 23, "RHeel": 24,
}


def get_joint(keypoints, name):
    """Return (x, y) for named joint if confidence >= CONF_MIN and position is valid, else None."""
    idx = JOINT[name]
    if not keypoints or idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    return (x, y) if c >= CONF_MIN and x > 0 and y > 0 else None


def main():
    """Parse arguments, assess seat height using knee_analysis peaks, write assessment.json."""
    if len(sys.argv) < 2:
        print("Usage: python3 seat_height.py <video_keypoints.json>")
        sys.exit(1)

    kp_path = sys.argv[1]
    with open(kp_path) as f:
        data = json.load(f)

    stem     = video_stem(kp_path, "_keypoints")
    out_dir  = os.path.join("output_v2", stem)
    os.makedirs(out_dir, exist_ok=True)
    base     = os.path.join(out_dir, stem)
    out_path = base + "_assessment.json"
    ka_path  = base + "_knee_analysis.json"

    ka        = None
    direction = None
    knee_used = "right"

    try:
        with open(ka_path) as f:
            ka = json.load(f)
        direction = ka.get("direction")
        knee_used = ka.get("knee_used", "right")
    except FileNotFoundError:
        print("Warning: knee_analysis.json not found — run knee_analysis.py first for accurate peak detection")

    if direction:
        print(f"Direction of travel    : {direction}  →  using {knee_used} knee (camera-facing side)")
    else:
        print("Direction of travel    : unknown — defaulting to right knee")

    # Per-frame angle computation for all joints (both sides, used by annotate_output.py)
    frames_out      = []
    knee_angles_all = []  # camera-facing knee angle per frame, for fallback peak

    for entry in data["frames"]:
        kp = entry.get("keypoints", [])

        r_hip   = get_joint(kp, "RHip")
        r_knee  = get_joint(kp, "RKnee")
        r_ankle = get_joint(kp, "RAnkle")
        l_hip   = get_joint(kp, "LHip")
        l_knee  = get_joint(kp, "LKnee")
        l_ankle = get_joint(kp, "LAnkle")
        r_sh    = get_joint(kp, "RShoulder")
        l_sh    = get_joint(kp, "LShoulder")

        r_knee_angle = calc_angle(r_hip, r_knee, r_ankle) if r_hip and r_knee and r_ankle else None
        l_knee_angle = calc_angle(l_hip, l_knee, l_ankle) if l_hip and l_knee and l_ankle else None
        r_hip_angle  = calc_angle(r_sh, r_hip, r_knee)    if r_sh and r_hip and r_knee    else None
        l_hip_angle  = calc_angle(l_sh, l_hip, l_knee)    if l_sh and l_hip and l_knee    else None

        frames_out.append({
            "frame_idx":        entry["frame_idx"],
            "timestamp":        entry["timestamp"],
            "right_knee_angle": round(r_knee_angle, 2) if r_knee_angle is not None else None,
            "left_knee_angle":  round(l_knee_angle, 2) if l_knee_angle is not None else None,
            "right_hip_angle":  round(r_hip_angle, 2)  if r_hip_angle  is not None else None,
            "left_hip_angle":   round(l_hip_angle, 2)  if l_hip_angle  is not None else None,
        })

        camera_angle = l_knee_angle if knee_used == "left" else r_knee_angle
        if camera_angle is not None:
            knee_angles_all.append(camera_angle)

    # Peak angle: mean across validated bottom-of-stroke peaks from all runs that
    # have >= 2 detected peaks. Falls back to max() of raw series when no run
    # has enough peaks for confident cycle detection.
    peak_angle        = None
    peak_angle_method = "insufficient_data"

    if ka:
        usable_peaks = [
            p
            for run in ka.get("runs", [])
            if len(run.get("peaks", [])) >= 2
            for p in run["peaks"]
        ]
        if usable_peaks:
            peak_angles       = [p["angle"] for p in usable_peaks]
            peak_angle        = sum(peak_angles) / len(peak_angles)
            peak_angle_method = "peak_mean"

    if peak_angle is None and knee_angles_all:
        peak_angle        = max(knee_angles_all)
        peak_angle_method = "max_fallback"

    verdict        = "insufficient_data"
    verdict_detail = "Not enough valid knee angle measurements."
    mean_angle     = None
    std_angle      = None

    if knee_angles_all:
        mean_angle = sum(knee_angles_all) / len(knee_angles_all)
        variance   = sum((a - mean_angle) ** 2 for a in knee_angles_all) / len(knee_angles_all)
        std_angle  = math.sqrt(variance)

    if peak_angle is not None:
        if peak_angle < KNEE_OPTIMAL_LOW:
            verdict = "too_low"
            verdict_detail = (
                f"Peak knee extension {peak_angle:.1f}° is below {KNEE_OPTIMAL_LOW}°. "
                "Seat is likely too low — raise saddle height."
            )
        elif peak_angle > KNEE_OPTIMAL_HIGH:
            verdict = "too_high"
            verdict_detail = (
                f"Peak knee extension {peak_angle:.1f}° exceeds {KNEE_OPTIMAL_HIGH}°. "
                "Seat is likely too high — risk of over-extension, lower saddle height."
            )
        else:
            verdict = "optimal"
            verdict_detail = (
                f"Peak knee extension {peak_angle:.1f}° is within the optimal range "
                f"({KNEE_OPTIMAL_LOW}–{KNEE_OPTIMAL_HIGH}°)."
            )

    output = {
        "video":  data["video"],
        "frames": frames_out,
        "summary": {
            "knee_angles_count":  len(knee_angles_all),
            "knee_angle_mean":    round(mean_angle, 2) if mean_angle is not None else None,
            "knee_angle_std":     round(std_angle, 2)  if std_angle  is not None else None,
            "knee_angle_peak":    round(peak_angle, 2) if peak_angle is not None else None,
            "peak_angle_method":  peak_angle_method,
            "optimal_range":      [KNEE_OPTIMAL_LOW, KNEE_OPTIMAL_HIGH],
            "verdict":            verdict,
            "verdict_detail":     verdict_detail,
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    s = output["summary"]
    print(f"Frames with knee angles : {s['knee_angles_count']}")
    print(f"Mean knee angle         : {s['knee_angle_mean']}°")
    print(f"Std deviation           : {s['knee_angle_std']}°")
    print(f"Peak extension          : {s['knee_angle_peak']}°  [{peak_angle_method}]")
    print(f"Verdict                 : {s['verdict'].upper()}")
    print(f"Detail                  : {s['verdict_detail']}")
    print(f"Assessment saved        → {out_path}")


if __name__ == "__main__":
    main()
