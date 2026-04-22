#!/usr/bin/env python3
"""
Stage 3: Seat height assessment from keypoints.
Computes knee angle (Hip -> Knee -> Ankle) across selected frames using
the law of cosines. Assesses the peak extension angle (bottom of pedal stroke).

Optimal knee angle at bottom of stroke: 145-155 degrees
  too_low  : peak < 145 (seat too low — power loss, knee stress)
  optimal  : 145 <= peak <= 155
  too_high : peak > 155 (seat too high — over-extension risk)

Usage: python3 seat_height.py <video_keypoints.json>
Output: <video>_assessment.json
"""
import json
import math
import os
import sys

KNEE_OPTIMAL_LOW  = 145.0
KNEE_OPTIMAL_HIGH = 155.0
CONF_MIN          = 0.1

JOINT = {
    "Nose": 0, "Neck": 1,
    "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7,
    "MidHip": 8,
    "RHip": 9, "RKnee": 10, "RAnkle": 11,
    "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18,
    "LBigToe": 19, "LSmallToe": 20, "LHeel": 21,
    "RBigToe": 22, "RSmallToe": 23, "RHeel": 24,
}


def calc_angle(A, B, C):
    """Law of cosines: returns angle at joint B in degrees."""
    a2 = (B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2
    b2 = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
    c2 = (A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2
    denom = 2 * math.sqrt(a2 * b2)
    if denom < 1e-6:
        return None
    cos_val = max(-1.0, min(1.0, (a2 + b2 - c2) / denom))
    return math.degrees(math.acos(cos_val))


def get_joint(keypoints, name):
    idx = JOINT[name]
    if not keypoints or idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    return (x, y) if c >= CONF_MIN and x > 0 and y > 0 else None

def detect_direction(selection_log_path):
    """
    Return 'left' or 'right' (direction cyclist is travelling) by comparing
    median front_wheel vs back_wheel x-centre across all selected frames.
    front_wheel to the right of back_wheel  → moving right.
    front_wheel to the left  of back_wheel  → moving left.
    Returns None if log not found or insufficient data.
    """
    if not os.path.exists(selection_log_path):
        return None
    with open(selection_log_path) as f:
        log = json.load(f)

    deltas = []
    for entry in log.get("selected_frames", []):
        fb = entry.get("front_wheel_box")
        bb = entry.get("back_wheel_box")
        if fb and bb:
            fx = (fb[0] + fb[2]) / 2
            bx = (bb[0] + bb[2]) / 2
            deltas.append(fx - bx)

    if not deltas:
        return None
    deltas.sort()
    median = deltas[len(deltas) // 2]
    return "right" if median > 0 else "left"


if len(sys.argv) < 2:
    print("Usage: python3 seat_height.py <video_keypoints.json>")
    sys.exit(1)

kp_path = sys.argv[1]
with open(kp_path) as f:
    data = json.load(f)

stem     = os.path.splitext(os.path.basename(kp_path))[0].replace("_keypoints", "")
out_dir  = os.path.join("output", stem)
os.makedirs(out_dir, exist_ok=True)
base     = os.path.join(out_dir, stem)
out_path = base + "_assessment.json"
log_path = base + "_selection_log.json"

direction = detect_direction(log_path)

if direction:
    print(f"Direction of travel    : {direction}  →  using {direction} knee (camera-facing side)")
else:
    print("Direction of travel    : unknown (selection log missing) — defaulting to right knee")

frames_out = []
for entry in data["frames"]:
    kp = entry.get("keypoints", [])

    r_hip    = get_joint(kp, "RHip")
    r_knee   = get_joint(kp, "RKnee")
    r_ankle  = get_joint(kp, "RAnkle")
    l_hip    = get_joint(kp, "LHip")
    l_knee   = get_joint(kp, "LKnee")
    l_ankle  = get_joint(kp, "LAnkle")
    r_sh     = get_joint(kp, "RShoulder")
    l_sh     = get_joint(kp, "LShoulder")

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

# Prefer right knee; fall back to left
knee_angles = []
for f in frames_out:
    angle = f["right_knee_angle"] if (f["right_knee_angle"] is not None and direction == "right") else f["left_knee_angle"]
    if angle is not None:
        knee_angles.append(angle)

verdict        = "insufficient_data"
verdict_detail = "Not enough valid knee angle measurements."
mean_angle = std_angle = peak_angle = None

if knee_angles:
    mean_angle = sum(knee_angles) / len(knee_angles)
    variance   = sum((a - mean_angle) ** 2 for a in knee_angles) / len(knee_angles)
    std_angle  = math.sqrt(variance)
    peak_angle = max(knee_angles)  # maximum extension approximates bottom of pedal stroke

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
        "knee_angles_count": len(knee_angles),
        "knee_angle_mean":   round(mean_angle, 2) if mean_angle is not None else None,
        "knee_angle_std":    round(std_angle, 2)  if std_angle  is not None else None,
        "knee_angle_peak":   round(peak_angle, 2) if peak_angle is not None else None,
        "optimal_range":     [KNEE_OPTIMAL_LOW, KNEE_OPTIMAL_HIGH],
        "verdict":           verdict,
        "verdict_detail":    verdict_detail,
    }
}

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

s = output["summary"]
print(f"Frames with knee angles : {s['knee_angles_count']}")
print(f"Mean knee angle         : {s['knee_angle_mean']}°")
print(f"Std deviation           : {s['knee_angle_std']}°")
print(f"Peak extension          : {s['knee_angle_peak']}°")
print(f"Verdict                 : {s['verdict'].upper()}")
print(f"Detail                  : {s['verdict_detail']}")
print(f"Assessment saved        → {out_path}")
