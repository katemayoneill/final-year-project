#!/usr/bin/env python3
"""
stage 4: seat height assessment from knee_analysis.json.

usage: python3 seat_height.py <video_knee_analysis.json>
output: output_v2/<stem>/<stem>_assessment.json
"""
import json
import os
import sys

import numpy as np

from utils import video_stem

KNEE_OPTIMAL_LOW = 145.0
KNEE_OPTIMAL_HIGH = 155.0
PEAK_MEAN_MIN_PEAKS = 10 
SMOOTH_PERCENTILE = 80


def main():
    """assess seat height from knee_analysis peaks, write assessment.json."""
    if len(sys.argv) < 2:
        print("usage: python3 seat_height.py <video_knee_analysis.json>")
        sys.exit(1)

    ka_path = sys.argv[1]
    with open(ka_path) as f:
        ka = json.load(f)

    stem = video_stem(ka_path, "_knee_analysis")
    out_dir = os.path.join("output_v2", stem)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, stem + "_assessment.json")

    direction = ka.get("direction")
    knee_used = ka.get("knee_used", "right")

    if direction:
        print(f"direction of travel: {direction}  (using {knee_used} knee)")
    else:
        print("direction of travel: unknown (defaulting to right knee)")

    usable_peaks = [
        p
        for run in ka.get("runs", [])
        if len(run.get("peaks", [])) >= 2
        for p in run["peaks"]
    ]
    smoothed_angles = [
        ang
        for run in ka.get("runs", [])
        for _, ang in run.get("angle_series", [])
    ]

    peak_angle = None
    peak_angle_method = "insufficient_data"

    if len(usable_peaks) >= PEAK_MEAN_MIN_PEAKS:
        peak_angle = sum(p["angle"] for p in usable_peaks) / len(usable_peaks)
        peak_angle_method = "peak_mean"
    elif smoothed_angles:
        peak_angle = float(np.percentile(smoothed_angles, SMOOTH_PERCENTILE))
        peak_angle_method = "smooth_p80"

    verdict = "insufficient_data"
    verdict_detail = "not enough valid knee angle measurements."
    mean_angle = None
    std_angle = None

    if smoothed_angles:
        arr = np.array(smoothed_angles)
        mean_angle = float(arr.mean())
        std_angle  = float(arr.std())

    if peak_angle is not None:
        if peak_angle < KNEE_OPTIMAL_LOW:
            verdict = "too_low"
            verdict_detail = (
                f"peak knee extension {peak_angle:.1f}deg is below {KNEE_OPTIMAL_LOW}deg. "
                "raise the saddle!!"
            )
        elif peak_angle > KNEE_OPTIMAL_HIGH:
            verdict = "too_high"
            verdict_detail = (
                f"Peak knee extension {peak_angle:.1f}deg exceeds {KNEE_OPTIMAL_HIGH}deg. "
                "lower the saddle!!"
            )
        else:
            verdict = "optimal"
            verdict_detail = (
                f"peak knee extension {peak_angle:.1f}deg is within the optimal range "
                f"({KNEE_OPTIMAL_LOW}-{KNEE_OPTIMAL_HIGH}deg)."
                "saddle is at correct height!!"
            )

    output = {
        "video": ka["video"],
        "summary": {
            "knee_angles_count": ka.get("metrics", {}).get("frames_with_angle", 0),
            "knee_angle_mean": round(mean_angle, 2) if mean_angle is not None else None,
            "knee_angle_std": round(std_angle, 2)  if std_angle  is not None else None,
            "knee_angle_peak": round(peak_angle, 2) if peak_angle is not None else None,
            "peak_angle_method": peak_angle_method,
            "optimal_range": [KNEE_OPTIMAL_LOW, KNEE_OPTIMAL_HIGH],
            "verdict": verdict,
            "verdict_detail": verdict_detail,
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    s = output["summary"]
    print(f"frames with knee angles : {s['knee_angles_count']}")
    print(f"mean knee angle : {s['knee_angle_mean']}deg")
    print(f"std deviation : {s['knee_angle_std']}deg")
    print(f"peak extension : {s['knee_angle_peak']}deg  [{peak_angle_method}]")
    print(f"verdict : {s['verdict'].upper()}")
    print(f"detail : {s['verdict_detail']}")
    print(f"assessment saved : {out_path}")


if __name__ == "__main__":
    main()
