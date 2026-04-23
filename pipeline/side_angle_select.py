#!/usr/bin/env python3
"""
Stage 1: Side-angle frame selection.
Runs YOLO on every frame (cheap). Finds the longest strictly consecutive burst
of frames where both wheels are visible and near-square (true side-on view),
saves only those frames to disk, and outputs selection_log.json.

Usage: python3 side_angle_select.py <video.mp4> [model.pt]
Output: <video>_selected_frames/  +  <video>_selection_log.json
"""
import cv2
import json
import os
import sys
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip3 install ultralytics")
    sys.exit(1)

CONFIDENCE = 0.5
SQUARE_TOL = 0.15   # max deviation from aspect ratio 1.0
SIZE_TOL   = 0.20   # max fractional difference in wheel areas
MODEL_PATH = "best.pt"


def main():
    """Parse arguments, select side-angle frames via YOLO, write selection_log.json."""
    global MODEL_PATH

    if len(sys.argv) < 2:
        print("Usage: python3 side_angle_select.py <video.mp4> [model.pt]")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        MODEL_PATH = sys.argv[2]

    stem       = os.path.splitext(os.path.basename(input_path))[0]
    base       = os.path.join("output", stem, stem)
    frames_dir = base + "_selected_frames"
    log_path   = base + "_selection_log.json"

    os.makedirs(frames_dir, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap         = cv2.VideoCapture(input_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    selected  = []
    frame_idx = 0
    t0        = time.time()

    print(f"Processing {frame_count} frames at {fps:.1f} fps...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE, verbose=False)

        front_box = back_box = None
        front_conf = back_conf = 0.0

        for result in results:
            for box in result.boxes:
                cls   = int(box.cls[0])
                label = model.names[cls]
                conf  = float(box.conf[0])
                coords = list(map(int, box.xyxy[0]))
                if label == "front_wheel" and conf > front_conf:
                    front_box, front_conf = coords, conf
                elif label == "back_wheel" and conf > back_conf:
                    back_box, back_conf = coords, conf

        side_angle = False
        front_sq = back_sq = size_ratio = None

        if front_box and back_box:
            fw = front_box[2] - front_box[0]
            fh = front_box[3] - front_box[1]
            bw = back_box[2] - back_box[0]
            bh = back_box[3] - back_box[1]
            front_sq  = abs(1.0 - (fw / fh if fh > 0 else 0))
            back_sq   = abs(1.0 - (bw / bh if bh > 0 else 0))
            f_area    = fw * fh
            b_area    = bw * bh
            size_ratio = min(f_area, b_area) / max(f_area, b_area) if max(f_area, b_area) > 0 else 0
            if front_sq < SQUARE_TOL and back_sq < SQUARE_TOL and size_ratio > (1.0 - SIZE_TOL):
                side_angle = True

        if side_angle:
            fname = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(fname, frame)
            selected.append({
                "frame_idx":        frame_idx,
                "timestamp":        round(frame_idx / fps, 4),
                "frame_file":       fname,
                "front_wheel_conf": round(front_conf, 4),
                "back_wheel_conf":  round(back_conf, 4),
                "front_squareness": round(1.0 - front_sq, 4),
                "back_squareness":  round(1.0 - back_sq, 4),
                "size_match_ratio": round(size_ratio, 4),
                "front_wheel_box":  front_box,
                "back_wheel_box":   back_box,
            })

        frame_idx += 1
        pct = frame_idx / frame_count
        bar = ('█' * int(pct * 40)).ljust(40)
        print(f'\r  [{bar}] {frame_idx}/{frame_count} ({pct:.0%})', end='', flush=True)

    cap.release()
    elapsed = time.time() - t0
    print()

    # Split into strictly consecutive runs (no gaps allowed)
    runs = []
    if selected:
        current_run = [selected[0]]
        for prev, curr in zip(selected, selected[1:]):
            if curr["frame_idx"] - prev["frame_idx"] == 1:
                current_run.append(curr)
            else:
                runs.append(current_run)
                current_run = [curr]
        runs.append(current_run)

    best_run = max(runs, key=len) if runs else []
    best_idx = {e["frame_idx"] for e in best_run}

    # Delete saved frames that are not in the best run
    for entry in selected:
        if entry["frame_idx"] not in best_idx:
            try:
                os.remove(entry["frame_file"])
            except OSError:
                pass

    avg_conf = (
        sum((s["front_wheel_conf"] + s["back_wheel_conf"]) / 2 for s in best_run) / len(best_run)
        if best_run else 0.0
    )

    log = {
        "video":            input_path,
        "model":            MODEL_PATH,
        "fps":              fps,
        "total_frames":     frame_idx,
        "selected_frames":  best_run,
        "metrics": {
            "frames_processed":    frame_idx,
            "frames_selected":     len(best_run),
            "total_bursts":        len(runs),
            "selection_rate":      round(len(best_run) / frame_idx, 4) if frame_idx > 0 else 0,
            "avg_confidence":      round(avg_conf, 4),
            "elapsed_sec":         round(elapsed, 2),
        }
    }

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    m = log["metrics"]
    print(f"Frames processed : {m['frames_processed']}")
    print(f"Bursts found     : {m['total_bursts']}")
    print(f"Frames selected  : {m['frames_selected']}  ({m['selection_rate']:.1%})  [longest burst]")
    print(f"Avg confidence   : {m['avg_confidence']:.3f}")
    print(f"Elapsed          : {m['elapsed_sec']}s")
    print(f"Selection log    → {log_path}")
    print(f"Saved frames     → {frames_dir}/")


if __name__ == "__main__":
    main()
