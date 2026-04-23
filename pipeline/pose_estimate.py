#!/usr/bin/env python3
"""
Stage 2: Pose estimation on selected frames.
Reads frame images listed in selection_log.json, runs OpenPose on each,
and saves 25-joint Body25 keypoints to keypoints.json.

Usage: python3 pose_estimate.py <video_selection_log.json>
Output: <video>_keypoints.json
"""
import json
import os
import sys
import time

import cv2
try:
    from openpose import pyopenpose as op
except ImportError:
    import pyopenpose as op

OPENPOSE_MODELS = os.environ.get("OPENPOSE_MODELS", "/openpose/models/")
CONF_MIN        = 0.1

JOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]


def main():
    """Parse arguments, run OpenPose on selected frames, write keypoints.json."""
    if len(sys.argv) < 2:
        print("Usage: python3 pose_estimate.py <video_selection_log.json>")
        sys.exit(1)

    log_path = sys.argv[1]
    with open(log_path) as f:
        log = json.load(f)

    stem     = os.path.splitext(os.path.basename(log_path))[0].replace("_selection_log", "")
    out_dir  = os.path.join("output", stem)
    os.makedirs(out_dir, exist_ok=True)
    base     = os.path.join(out_dir, stem)
    out_path = base + "_keypoints.json"

    params  = {"model_folder": OPENPOSE_MODELS}
    wrapper = op.WrapperPython()
    wrapper.configure(params)
    wrapper.start()
    print(f"OpenPose ready. Processing {len(log['selected_frames'])} frames...")

    frames_out      = []
    inference_times = []

    for i, entry in enumerate(log["selected_frames"]):
        frame = cv2.imread(entry["frame_file"])
        if frame is None:
            print(f"  [WARN] Could not read {entry['frame_file']}, skipping")
            continue

        datum             = op.Datum()
        datum.cvInputData = frame
        t0                = time.time()
        wrapper.emplaceAndPop(op.VectorDatum([datum]))
        ms = (time.time() - t0) * 1000
        inference_times.append(ms)

        keypoints  = []
        joint_conf = {}
        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            kp = datum.poseKeypoints[0]
            for j, name in enumerate(JOINT_NAMES):
                x, y, c = float(kp[j][0]), float(kp[j][1]), float(kp[j][2])
                keypoints.append([round(x, 2), round(y, 2), round(c, 4)])
                joint_conf[name] = round(c, 4)

        frames_out.append({
            "frame_idx":         entry["frame_idx"],
            "timestamp":         entry["timestamp"],
            "frame_file":        entry["frame_file"],
            "inference_time_ms": round(ms, 1),
            "keypoints":         keypoints,
            "joint_confidences": joint_conf,
        })

        pct = (i + 1) / len(log["selected_frames"])
        bar = ('█' * int(pct * 40)).ljust(40)
        print(f'\r  [{bar}] {i+1}/{len(log["selected_frames"])} ({pct:.0%})  {ms:.0f}ms', end='', flush=True)

    print()

    avg_infer = sum(inference_times) / len(inference_times) if inference_times else 0

    avg_joint_conf = {}
    if frames_out:
        for name in JOINT_NAMES:
            vals = [f["joint_confidences"].get(name, 0) for f in frames_out if f["joint_confidences"]]
            avg_joint_conf[name] = round(sum(vals) / len(vals), 4) if vals else 0

    output = {
        "video":  log["video"],
        "frames": frames_out,
        "metrics": {
            "frames_processed":      len(frames_out),
            "avg_inference_time_ms": round(avg_infer, 1),
            "avg_joint_confidence":  avg_joint_conf,
        }
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    m = output["metrics"]
    print(f"Frames processed   : {m['frames_processed']}")
    print(f"Avg inference time : {m['avg_inference_time_ms']}ms/frame")
    print(f"Keypoints saved    → {out_path}")


if __name__ == "__main__":
    main()
