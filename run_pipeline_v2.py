#!/usr/bin/env python3
"""
run full pipeline

usage: python3 run_pipeline_v2.py <video.mp4> [model.pt]
"""
import json
import os
import subprocess
import sys
import time

PYTHON = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "pipeline_v2")


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

log_path = base + "_selection_log.json"
kp_path = base + "_keypoints.json"
ka_path = base + "_knee_analysis.json"
assessment_path = base + "_assessment.json"
rpm_path = base + "_rpm.json"
final_path = base + "_final.mp4"

print(f"\n[PIPELINE]")
print(f"video: {video}")
print(f"model: {model}")
print(f"output dir: output_v2/{stem}/")

run("stage 1",
    [PYTHON, os.path.join(P2, "side_angle.py"), video, model])

log = load(log_path)
m   = log["metrics"]
print_section("Side-angle selection")
print(f"frames processed: {m['frames_processed']}")
print(f"frames selected: {m['frames_selected']}  ({m['selection_rate']:.1%})")
print(f"average confidence: {m['avg_confidence']:.3f}")
print(f"elapsed: {m['elapsed_sec']}s")
if m['frames_selected'] == 0:
    print("\n[WARN] no side-angle frames found.")
    sys.exit(1)

run("stage 2",
    [PYTHON, os.path.join(P2, "pose_estimate.py"), log_path])

kp = load(kp_path)
m = kp["metrics"]
print_section("pose estimation")
print(f"frames processed: {m['frames_processed']}")
print(f"avg inference time: {m['avg_inference_time_ms']}ms/frame")
low_conf = {k: v for k, v in m["avg_joint_confidence"].items() if v < 0.3}
if low_conf:
    print(f"low-confidence joints   : {', '.join(f'{k} ({v:.2f})' for k, v in low_conf.items())}")
key_joints = ["RKnee", "RHip", "RAnkle", "LKnee", "LHip", "LAnkle"]
print("joint confidences:")
for j in key_joints:
    conf = m["avg_joint_confidence"].get(j, 0)
    bar = "#" * int(conf * 20)
    print(f"    {j:<12} {conf:.2f}  {bar}")


run("stage 3",
    [PYTHON, os.path.join(P2, "knee_analysis.py"), kp_path])

ka = load(ka_path)
km = ka["metrics"]
print_section("knee cycle analysis")
print(f"direction: {ka.get('direction', 'unknown')}, {ka.get('knee_used', '?')} knee")
print(f"frames with angle: {km['frames_with_angle']}")
print(f"contiguous runs: {km['total_runs']}  (best run: {ka['best_run']['frame_count']} frames, {ka['best_run']['duration_sec']}s)")
print(f"peaks detected: {km['peaks_found']}  [method: {ka['peak_method']}]")
if ka.get("autocorr_period_sec"):
    print(f"autocorr period: {ka['autocorr_period_sec']}s")

run("stage 4",
    [PYTHON, os.path.join(P2, "seat_height.py"), ka_path])

assess = load(assessment_path)
s = assess["summary"]
print_section("Seat height assessment")
print(f"frames with knee angles: {s['knee_angles_count']}")
print(f"mean knee angle: {s['knee_angle_mean']}°")
print(f"ttd deviation: {s['knee_angle_std']}°")
print(f"peak extension: {s['knee_angle_peak']}°  [{s.get('peak_angle_method', '?')}]  (optimal: {s['optimal_range'][0]}–{s['optimal_range'][1]}°)")
sym = {"optimal": ":)", "too_high": "<", "too_low": ">"}.get(s["verdict"], "?")
print(f"verdict: {sym}  {s['verdict'].upper()}")
print(f"detail: {s['verdict_detail']}")

run("stage 5",
    [PYTHON, os.path.join(P2, "rpm.py"), ka_path])

rpm = load(rpm_path)
m = rpm["metrics"]
print_section("RPM / cadence")
print(f"frames with knee angle: {m['frames_with_angle']}")
print(f"time span: {m['time_span_sec']}s")
print(f"peaks (pedal cycles): {rpm['cycle_count']}")
if rpm["cadence_rpm"] is not None:
    suffix = f"  pm{rpm['std_dev_rpm']} RPM" if rpm["std_dev_rpm"] else ""
    print(f"cadence: {rpm['cadence_rpm']} RPM{suffix}")
    if rpm["cycle_periods_sec"]:
        print(f"cycle periods: {[f'{p:.2f}s' for p in rpm['cycle_periods_sec']]}")
else:
    print("cadence: insufficient data")

run("stage 6",
    [PYTHON, os.path.join(P2, "annotate_output.py"),
     video, kp_path, assessment_path, rpm_path])

print(f"\n{'='*60}")
print("COMPLETE")
print(f"{'='*60}")
print(f"video: {video}")
print(f"selection log: {log_path}")
print(f"keypoints: {kp_path}")
print(f"knee analysis: {ka_path}")
print(f"assessment: {assessment_path}")
print(f"RPM data: {rpm_path}")
print(f"annotated video: {final_path}")
print()
print(f"seat height: {s['verdict'].upper()} {s['verdict_detail']}")
if rpm["cadence_rpm"] is not None:
    print(f"cadence: {rpm['cadence_rpm']} RPM")
else:
    print("cadence: insufficient data")
print()
