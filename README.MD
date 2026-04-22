# Cycling Posture Analysis System

Analyses real-world smartphone footage of a cyclist passing a camera. Detects side-on frames automatically, runs OpenPose on those frames to extract joint keypoints, then measures knee angles to assess seat height and estimate pedalling RPM.

## Pipeline

```
smartphone_video.mp4
  → side_angle_select.py  →  selected_frames/  +  selection_log.json
  → pose_estimate.py      →  keypoints.json          (RunPod GPU)
  → seat_height.py        →  assessment.json
  → rpm.py                →  rpm.json
  → annotate_output.py    →  _final.mp4
```

## Usage

### Run all stages at once

```bash
python3 run_pipeline.py <video.mp4> [best.pt]
```

### Run stages individually

```bash
# Stage 1 — side-angle frame selection (YOLO, CPU)
python3 pipeline/side_angle_select.py <video.mp4> [best.pt]

# Stage 2 — pose estimation (OpenPose, RunPod GPU)
export OPENPOSE_MODELS=/users/ugrad/oneilk10/openpose/models/
python3 pipeline/pose_estimate.py <video>_selection_log.json

# Stage 3 — seat height assessment
python3 pipeline/seat_height.py <video>_keypoints.json

# Stage 4 — RPM / cadence
python3 pipeline/rpm.py <video>_keypoints.json

# Stage 5 — annotated output video
python3 pipeline/annotate_output.py <video.mp4> <video>_keypoints.json <video>_assessment.json <video>_rpm.json
```

## Local Setup (msc-linux-sls-016)

```bash
source ~/fyp/linux/bin/activate
export OPENPOSE_MODELS=/users/ugrad/oneilk10/openpose/models/
```

Both are set automatically via `~/.bashrc`.

## File Transfer (Cloudflare R2)

Large files are transferred via R2 (HTTPS, port 443) — runpodctl is blocked on the college network.

```bash
# Upload a video to R2
python3 infra/upload.py <video.mp4>

# Download a file from R2
python3 infra/download.py <key> [dest]

# Push all pipeline scripts to R2 (run locally after any script change)
python3 infra/upload_scripts.py

# Pull latest scripts on pod
python3 /app/download_scripts.py
```

Credentials via `.env`: `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`.

## Wheel-Detection Experiments

Standalone scripts comparing four approaches for the report:

```bash
python3 experiments/wheel_detection/wheels.py   <video.mp4>   # YOLO bbox → circle
python3 experiments/wheel_detection/circles.py  <video.mp4>   # Hough circles
python3 experiments/wheel_detection/arcs.py     <video.mp4>   # RANSAC arc fitting
python3 experiments/wheel_detection/contours.py <video.mp4>   # contour circularity
```

## Inference Benchmark Experiments

```bash
python3 experiments/inference_benchmarks/infer_base.py       <video.mp4>
python3 experiments/inference_benchmarks/infer_opt1_fp16.py  <video.mp4>
python3 experiments/inference_benchmarks/infer_opt2_batch.py <video.mp4>
python3 experiments/inference_benchmarks/infer_opt3_skip.py  <video.mp4>
python3 experiments/inference_benchmarks/infer_opt4_all.py   <video.mp4>
```

## Seat Height Verdicts

| Verdict | Peak knee extension |
|---|---|
| `too_low` | < 145° |
| `optimal` | 145° – 155° |
| `too_high` | > 155° |
