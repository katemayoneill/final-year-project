#!/usr/bin/env python3
"""
Pipeline V2 — Stage 1: Side-angle frame selection.
Runs YOLO on every frame (cheap). Scores each consecutive qualifying burst by a
composite quality metric (squareness × size-match × normalised cyclist height ×
burst length), keeps all bursts scoring >= QUALITY_FRACTION of the best burst,
saves their frames to disk, and outputs selection_log.json.

Usage: python3 side_angle_select.py <video.mp4> [model.pt]
Output: output_v2/<stem>/<stem>_selected_frames/  +  output_v2/<stem>/<stem>_selection_log.json
"""
import cv2
import json
import os
import sys
import time

from utils import print_progress, video_stem

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip3 install ultralytics")
    sys.exit(1)

CONFIDENCE       = 0.5
SQUARE_TOL       = 0.15   # max deviation from aspect ratio 1.0
SIZE_TOL         = 0.20   # max fractional difference in wheel areas
MODEL_PATH       = "best.pt"
MIN_BURST_FRAMES = 5      # bursts shorter than this are always excluded
QUALITY_FRACTION = 0.5    # keep bursts scoring >= this fraction of the best burst


def burst_quality_score(run, frame_h):
    """
    Composite burst quality: length × mean squareness × mean size-match × mean
    normalised cyclist height. Rewards both duration and per-frame image quality
    (a taller cyclist bounding box gives OpenPose more resolution to work with).
    """
    n = len(run)
    if n == 0:
        return 0.0
    mean_sq   = sum((f["front_squareness"] + f["back_squareness"]) / 2 for f in run) / n
    mean_size = sum(f["size_match_ratio"] for f in run) / n
    heights   = [
        (f["cyclist_box"][3] - f["cyclist_box"][1]) / frame_h
        for f in run if f.get("cyclist_box")
    ]
    mean_ht = sum(heights) / len(heights) if heights else 0.5
    return n * mean_sq * mean_size * mean_ht


def main():
    """Parse arguments, select side-angle frames via YOLO (saves cyclist_box), write selection_log.json."""
    global MODEL_PATH

    if len(sys.argv) < 2:
        print("Usage: python3 side_angle_select.py <video.mp4> [model.pt]")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        MODEL_PATH = sys.argv[2]

    stem       = video_stem(input_path)
    base       = os.path.join("output_v2", stem, stem)
    frames_dir = base + "_selected_frames"
    log_path   = base + "_selection_log.json"

    os.makedirs(frames_dir, exist_ok=True)

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline V2] Loading model: {MODEL_PATH}  (device: {'cuda:0' if device == 0 else 'cpu'})")
    model = YOLO(MODEL_PATH)

    cap         = cv2.VideoCapture(input_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    selected  = []
    frame_idx = 0
    t0        = time.time()

    print(f"Processing {frame_count} frames at {fps:.1f} fps...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE, verbose=False, device=device)

        front_box = back_box = cyclist_box = None
        front_conf = back_conf = cyclist_conf = 0.0

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
                elif label == "cyclist" and conf > cyclist_conf:
                    cyclist_box, cyclist_conf = coords, conf

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
                "cyclist_box":      cyclist_box,
            })

        frame_idx += 1
        print_progress(frame_idx, frame_count)

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

    # Score every run; keep all bursts scoring >= QUALITY_FRACTION of best
    if runs:
        run_scores = [burst_quality_score(r, frame_h) for r in runs]
        best_score = max(run_scores)
        threshold  = QUALITY_FRACTION * best_score
        good_pairs = [
            (s, r) for s, r in zip(run_scores, runs)
            if s >= threshold and len(r) >= MIN_BURST_FRAMES
        ]
        if not good_pairs:
            # Fallback: keep the highest-scoring run regardless of length
            best_i = run_scores.index(best_score)
            good_pairs = [(best_score, runs[best_i])]
        good_pairs.sort(key=lambda x: x[1][0]["frame_idx"])  # temporal order
    else:
        good_pairs = []

    # Build final frame set from all good runs
    good_idx = {e["frame_idx"] for _, run in good_pairs for e in run}

    # Delete frames that are not in any selected burst
    for entry in selected:
        if entry["frame_idx"] not in good_idx:
            try:
                os.remove(entry["frame_file"])
            except OSError:
                pass

    selected_frames_flat = [e for e in selected if e["frame_idx"] in good_idx]

    selected_bursts = [
        {
            "burst_id":        i,
            "quality_score":   round(s, 4),
            "frame_idx_start": run[0]["frame_idx"],
            "frame_idx_end":   run[-1]["frame_idx"],
            "frame_count":     len(run),
        }
        for i, (s, run) in enumerate(good_pairs)
    ]

    avg_conf = (
        sum((f["front_wheel_conf"] + f["back_wheel_conf"]) / 2 for f in selected_frames_flat)
        / len(selected_frames_flat)
        if selected_frames_flat else 0.0
    )

    log = {
        "video":            input_path,
        "model":            MODEL_PATH,
        "fps":              fps,
        "total_frames":     frame_idx,
        "selected_frames":  selected_frames_flat,
        "selected_bursts":  selected_bursts,
        "metrics": {
            "frames_processed": frame_idx,
            "frames_selected":  len(selected_frames_flat),
            "total_bursts":     len(runs),
            "bursts_selected":  len(good_pairs),
            "selection_rate":   round(len(selected_frames_flat) / frame_idx, 4) if frame_idx > 0 else 0,
            "avg_confidence":   round(avg_conf, 4),
            "elapsed_sec":      round(elapsed, 2),
        }
    }

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    m = log["metrics"]
    print(f"Frames processed : {m['frames_processed']}")
    print(f"Bursts found     : {m['total_bursts']}")
    print(f"Bursts selected  : {m['bursts_selected']}  (quality >= {QUALITY_FRACTION:.0%} of best)")
    for i, (s, run) in enumerate(good_pairs):
        print(f"  Burst {i}        : frames {run[0]['frame_idx']}–{run[-1]['frame_idx']}  ({len(run)} frames, score={s:.3f})")
    print(f"Frames selected  : {m['frames_selected']}  ({m['selection_rate']:.1%})")
    print(f"Avg confidence   : {m['avg_confidence']:.3f}")
    print(f"Elapsed          : {m['elapsed_sec']}s")
    print(f"Selection log    → {log_path}")
    print(f"Saved frames     → {frames_dir}/")


if __name__ == "__main__":
    main()
