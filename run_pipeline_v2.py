#!/usr/bin/env python3
"""
Pipeline V2 — Full pipeline runner.
Runs all 6 stages in sequence and prints a summary after each one.
Outputs go to output_v2/<stem>/ (separate from Pipeline 1's output/<stem>/).

Stage 3 (knee_analysis) is the single source of truth for direction, knee
selection, smoothing, and pedal cycle peak detection. Stages 4 and 5 consume
its output rather than repeating the analysis independently.

Usage: python3 run_pipeline_v2.py <video.mp4> [model.pt]
"""
import json
import os
import subprocess
import sys
import time

PYTHON = sys.executable
HERE   = os.path.dirname(os.path.abspath(__file__))


def run(label, cmd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=HERE)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  [{label} completed in {elapsed:.1f}s]")


def load(path):
    with open(path) as f:
        return json.load(f)


def print_section(title):
    print(f"\n{'─'*60}")
    print(f"  RESULTS: {title}")
    print(f"{'─'*60}")


if len(sys.argv) < 2:
    print("Usage: python3 run_pipeline_v2.py <video.mp4> [model.pt]")
    sys.exit(1)

video   = sys.argv[1]
model   = sys.argv[2] if len(sys.argv) >= 3 else "best.pt"
stem    = os.path.splitext(os.path.basename(video))[0]
out_dir = os.path.join(HERE, "output_v2", stem)
os.makedirs(out_dir, exist_ok=True)
base    = os.path.join("output_v2", stem, stem)

log_path        = base + "_selection_log.json"
kp_path         = base + "_keypoints.json"
ka_path         = base + "_knee_analysis.json"
assessment_path = base + "_assessment.json"
rpm_path        = base + "_rpm.json"
final_path      = base + "_final.mp4"

print(f"\n[Pipeline V2]")
print(f"Input video    : {video}")
print(f"Model          : {model}")
print(f"Output dir     : output_v2/{stem}/")

# ── Stage 1: side-angle selection ──────────────────────────────────────────
run("Pipeline V2 — Stage 1: Side-angle frame selection",
    [PYTHON, os.path.join(HERE, "pipeline_v2", "side_angle_select.py"), video, model])

log = load(log_path)
m   = log["metrics"]
print_section("Side-angle selection")
print(f"  Frames processed : {m['frames_processed']}")
print(f"  Frames selected  : {m['frames_selected']}  ({m['selection_rate']:.1%})")
print(f"  Avg confidence   : {m['avg_confidence']:.3f}")
print(f"  Elapsed          : {m['elapsed_sec']}s")
if m['frames_selected'] == 0:
    print("\n  [WARN] No side-angle frames found — check model and video.")
    sys.exit(1)

# ── Stage 2: OpenPose pose estimation ──────────────────────────────────────
run("Pipeline V2 — Stage 2: Pose estimation (OpenPose)",
    [PYTHON, os.path.join(HERE, "pipeline_v2", "pose_estimate.py"), log_path])

kp = load(kp_path)
m  = kp["metrics"]
print_section("Pose estimation")
print(f"  Frames processed        : {m['frames_processed']}")
print(f"  Avg inference time      : {m['avg_inference_time_ms']}ms/frame")
low_conf = {k: v for k, v in m["avg_joint_confidence"].items() if v < 0.3}
if low_conf:
    print(f"  Low-confidence joints   : {', '.join(f'{k} ({v:.2f})' for k, v in low_conf.items())}")
key_joints = ["RKnee", "RHip", "RAnkle", "LKnee", "LHip", "LAnkle"]
print("  Key joint confidences   :")
for j in key_joints:
    conf = m["avg_joint_confidence"].get(j, 0)
    bar  = "█" * int(conf * 20)
    print(f"    {j:<12} {conf:.2f}  {bar}")

# ── Stage 3: knee cycle analysis ───────────────────────────────────────────
run("Pipeline V2 — Stage 3: Knee cycle analysis",
    [PYTHON, os.path.join(HERE, "pipeline_v2", "knee_analysis.py"), kp_path])

ka = load(ka_path)
km = ka["metrics"]
print_section("Knee cycle analysis")
print(f"  Direction              : {ka.get('direction', 'unknown')}  →  {ka.get('knee_used', '?')} knee")
print(f"  Frames with angle      : {km['frames_with_angle']}")
print(f"  Contiguous runs        : {ka['best_run']['total_runs']}  (best run: {ka['best_run']['frame_count']} frames, {ka['best_run']['duration_sec']}s)")
print(f"  Peaks detected         : {km['peaks_found']}  [method: {ka['peak_method']}]")
if ka.get("autocorr_period_sec"):
    print(f"  Autocorr period        : {ka['autocorr_period_sec']}s")

# ── Stage 4: seat height assessment ────────────────────────────────────────
run("Pipeline V2 — Stage 4: Seat height assessment",
    [PYTHON, os.path.join(HERE, "pipeline_v2", "seat_height.py"), kp_path])

assess = load(assessment_path)
s      = assess["summary"]
print_section("Seat height assessment")
print(f"  Frames with knee angles : {s['knee_angles_count']}")
print(f"  Mean knee angle         : {s['knee_angle_mean']}°")
print(f"  Std deviation           : {s['knee_angle_std']}°")
print(f"  Peak extension          : {s['knee_angle_peak']}°  [{s.get('peak_angle_method', '?')}]  (optimal: {s['optimal_range'][0]}–{s['optimal_range'][1]}°)")
verdict_symbols = {"optimal": "✓", "too_high": "↑", "too_low": "↓", "insufficient_data": "?"}
sym = verdict_symbols.get(s["verdict"], "?")
print(f"  Verdict                 : {sym}  {s['verdict'].upper()}")
print(f"  Detail                  : {s['verdict_detail']}")

# ── Stage 5: RPM ───────────────────────────────────────────────────────────
run("Pipeline V2 — Stage 5: RPM / cadence",
    [PYTHON, os.path.join(HERE, "pipeline_v2", "rpm.py"), ka_path])

rpm = load(rpm_path)
m   = rpm["metrics"]
print_section("RPM / cadence")
print(f"  Frames with knee angle  : {m['frames_with_angle']}")
print(f"  Time span               : {m['time_span_sec']}s")
print(f"  Peaks (pedal cycles)    : {rpm['cycle_count']}")
if rpm["cadence_rpm"] is not None:
    suffix = f"  ±{rpm['std_dev_rpm']} RPM" if rpm["std_dev_rpm"] else ""
    print(f"  Cadence                 : {rpm['cadence_rpm']} RPM{suffix}")
    if rpm["cycle_periods_sec"]:
        print(f"  Cycle periods           : {[f'{p:.2f}s' for p in rpm['cycle_periods_sec']]}")
else:
    print("  Cadence                 : insufficient data (need ≥2 peaks)")

# ── Stage 6: annotate output video ─────────────────────────────────────────
run("Pipeline V2 — Stage 6: Annotate output video",
    [PYTHON, os.path.join(HERE, "pipeline_v2", "annotate_output.py"),
     video, kp_path, assessment_path, rpm_path])

# ── Final summary ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  PIPELINE V2 COMPLETE")
print(f"{'='*60}")
print(f"  Input video      : {video}")
print(f"  Selection log    : {log_path}")
print(f"  Keypoints        : {kp_path}")
print(f"  Knee analysis    : {ka_path}")
print(f"  Assessment       : {assessment_path}")
print(f"  RPM data         : {rpm_path}")
print(f"  Annotated output : {final_path}")
print()
print(f"  Seat height      : {assess['summary']['verdict'].upper()} — {assess['summary']['verdict_detail']}")
if rpm["cadence_rpm"] is not None:
    print(f"  Cadence          : {rpm['cadence_rpm']} RPM")
else:
    print("  Cadence          : insufficient data")
print()
