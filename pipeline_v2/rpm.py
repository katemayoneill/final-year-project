#!/usr/bin/env python3
"""
stage 5: cadence (RPM) from knee_analysis.json.

usage: python3 rpm.py <video_knee_analysis.json>
output: output_v2/<stem>/<stem>_rpm.json
"""
import json
import math
import os
import sys

from utils import video_stem


def main():
    """calculate cadence (RPM) from knee_analysis peaks or autocorrelation, writes rpm.json."""
    if len(sys.argv) < 2:
        print("usage: python3 rpm.py <video_knee_analysis.json>")
        sys.exit(1)

    ka_path = sys.argv[1]
    with open(ka_path) as f:
        ka = json.load(f)

    stem = video_stem(ka_path, "_knee_analysis")
    out_dir = os.path.join("output_v2", stem)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, stem + "_rpm.json")

    direction = ka.get("direction")
    knee_used = ka.get("knee_used", "right")
    runs = ka.get("runs", [])
    run_info = ka.get("best_run", {})

    if direction:
        print(f"direction: {direction}  (using {knee_used} knee, camera-facing side)")
    else:
        print("direction: unknown defaulting to right knee")

    all_periods = []
    per_run_rpms = []

    for run in runs:
        peaks = run.get("peaks", [])
        if len(peaks) >= 2:
            ts = [p["timestamp"] for p in peaks]
            periods = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            avg_run_period = sum(periods) / len(periods)
            per_run_rpms.append({
                "run_id": run["run_id"],
                "rpm": round(60.0 / avg_run_period, 1),
                "peak_count": len(peaks),
            })
            all_periods.extend(periods)

    cadence_rpm = None
    std_dev_rpm = None
    cycle_periods = []
    peak_method = "peak_detection"

    if all_periods:
        avg_period  = sum(all_periods) / len(all_periods)
        cadence_rpm = round(60.0 / avg_period, 1)
        cycle_periods = [round(p, 4) for p in all_periods]

        if len(all_periods) > 1:
            var   = sum((p - avg_period) ** 2 for p in all_periods) / len(all_periods)
            std_p = math.sqrt(var)
            if avg_period > std_p:
                std_dev_rpm = round(
                    (60.0 / (avg_period - std_p) - 60.0 / (avg_period + std_p)) / 2, 1
                )
    else:
        for run in runs:
            if run.get("peak_method") == "autocorrelation" and run.get("autocorr_period_sec"):
                autocorr_period = run["autocorr_period_sec"]
                cadence_rpm = round(60.0 / autocorr_period, 1)
                cycle_periods = [autocorr_period]
                peak_method = "autocorrelation"
                break

    peak_timestamps = [
        p["timestamp"]
        for run in runs
        if len(run.get("peaks", [])) >= 2
        for p in run["peaks"]
    ]
    angle_series = ka.get("angle_series", [])

    output = {
        "video": ka["video"],
        "direction": direction,
        "knee_used": knee_used,
        "cadence_rpm": cadence_rpm,
        "cycle_count": sum(len(r.get("peaks", [])) for r in runs if len(r.get("peaks", [])) >= 2),
        "cycle_timestamps": peak_timestamps,
        "cycle_periods_sec": cycle_periods,
        "std_dev_rpm": std_dev_rpm,
        "rpm_method": peak_method,
        "per_run_rpms": per_run_rpms,
        "best_run": run_info,
        "metrics": {
            "frames_with_angle": ka.get("metrics", {}).get("frames_with_angle", 0),
            "frames_in_best_run": run_info.get("frame_count", 0),
            "total_runs": len(runs),
            "usable_runs": len(per_run_rpms),
            "peaks_found": sum(len(r.get("peaks", [])) for r in runs),
            "time_span_sec": ka.get("metrics", {}).get("time_span_sec", 0),
        },

        "angle_series": angle_series,
        "peak_timestamps": peak_timestamps,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    m = output["metrics"]
    print(f"frames with knee angle: {m['frames_with_angle']}")
    print(f"runs (total / usable): {m['total_runs']} / {m['usable_runs']}")
    if per_run_rpms:
        parts = ", ".join(f"{r['rpm']} RPM (run {r['run_id']}, {r['peak_count']} peaks)" for r in per_run_rpms)
        print(f"per-run RPM: {parts}")
    if cadence_rpm is not None:
        suffix = f"  +/-{std_dev_rpm}" if std_dev_rpm else ""
        label = "mean across runs" if len(per_run_rpms) > 1 else peak_method
        print(f"Cadence: {cadence_rpm} RPM{suffix}  [{label}]")
    else:
        print("cadence: insufficient data")
    print(f"RPM saved: {out_path}")


if __name__ == "__main__":
    main()
