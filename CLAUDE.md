# Final Year Project — Cycling Posture Analysis

## Project Overview
A cycling posture analysis system for real-world smartphone footage of a cyclist moving past a camera. Detects the moment the cyclist is at a true side-on angle, runs OpenPose only on those frames to extract joint keypoints, then measures angles to assess seat height and compute pedalling RPM.

**Research gap:** existing systems assume a bike mounted on a trainer (controlled environment). This system works on real-world pass-by footage.

**Why OpenPose:** Body25 skeleton is the standard in academic biomechanics literature and has been more thoroughly validated for lower-limb angle measurement than newer alternatives (MediaPipe, YOLOPose). Required by supervisor.

**Core computational insight:** YOLO side-angle gating runs at ~30fps on CPU; OpenPose takes ~1–2s per frame on GPU. Selecting only good side-angle frames (e.g. 67 from 900) reduces OpenPose calls ~13×.

## Pipeline
```
smartphone_video.mp4
  → side_angle_select.py  → output/<stem>/<stem>_selected_frames/
  │                          output/<stem>/<stem>_selection_log.json
  │   (YOLO on every frame — cheap, runs locally or on pod)
  │   Detects when both wheels visible + near-square aspect ratio
  │
  → pose_estimate.py      → output/<stem>/<stem>_keypoints.json
  │   (OpenPose only on selected frames — expensive, cloud GPU)
  │   25-joint Body25 keypoints per selected frame
  │
  → seat_height.py        → output/<stem>/<stem>_assessment.json
  │   (Law of cosines on hip/knee/ankle keypoints)
  │   Per-frame angles + verdict: optimal / too high / too low
  │
  → rpm.py                → output/<stem>/<stem>_rpm.json
  │   (Count knee-angle cycles per second from keypoints)
  │   Cadence estimate + cycle timestamps
  │
  → annotate_output.py    → output/<stem>/<stem>_final.mp4
      (Overlay skeleton + angles + verdict + RPM on original video)
```

All outputs for a given video land in `output/<stem>/` (e.g. `output/jennyb30/`). Each stage creates the directory if it doesn't exist, so stages can be run standalone without `run_pipeline.py`.

Each stage also outputs intermediate files so stages can be re-run independently and evaluated in isolation for the report.

## Stack
- **OpenPose** (`pyopenpose`) — Body25 model, 25-joint pose estimation
- **YOLO** (`ultralytics`) — Person + bicycle detection; custom model `yolo26s.pt` used by wheel-detection scripts
- **OpenCV** — Video I/O, annotation
- **NumPy** — Kalman filter, angle maths (law of cosines)
- **Docker** — Base: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **RunPod** — GPU cloud deployment target

## Docker Image
- `infra/Dockerfile` — build config; `infra/DOCKER_HUB.md` — push/pull instructions
- CUDA 11.8 + cuDNN 8 on Ubuntu 22.04
- OpenPose compiled from source with Python bindings (`BUILD_PYTHON=ON`), cuDNN + CUDA enabled
- GPU architectures: 60, 61, 62, 70, 72, 75, 80, 86, 89, 90
- On startup: run `infra/startup.sh` — installs Python deps, then fetches `download_scripts.py` from R2 via `download.py`, then pulls all pipeline scripts to `/app`
- OpenPose models in `/openpose/models/` — baked into the image at build time via `COPY models/`; NOT in git (too large)
- Scripts are NOT bundled in the image — pulled from R2 at startup so updates don't require a rebuild

## Key Constraints
- **Model weights** are excluded from git (`.gitignore` ignores `/models`)
- **Raw footage** (`videos/`) is gitignored — too large for git; transfer via R2
- **runpodctl** is blocked on the college network — cannot use it for file transfer
- **HTTP transfers** time out on large files — naive HTTP upload/download is not viable
- Target workflow: upload input video to pod → run pipeline → retrieve output files

## File Transfer
Large files are transferred via **Cloudflare R2** (S3-compatible, HTTPS/port 443):
- `infra/upload.py <file>` — uploads a single file to R2 (key = filename, no prefix)
- `infra/download.py <key> [dest]` — downloads a single file from R2
- `infra/upload_scripts.py` — uploads all pipeline scripts to R2 under `scripts/` prefix; run locally after any script change
- `infra/download_scripts.py [dest_dir]` — downloads all pipeline scripts from R2; run on pod to get latest versions
- Credentials via `.env`: `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
- `boto3.upload_file` handles multipart automatically — suitable for large videos

### Pod setup workflow
1. On first pod start, `infra/download.py` and `.env` must already be in `/app` (paste via web terminal or bake into image)
2. `infra/startup.sh` uses `download.py` to fetch `download_scripts.py` from R2, then runs it to pull all other scripts
3. After that, re-running `infra/startup.sh` always gets the latest script versions from R2

### Updating scripts
```bash
# Local — after editing any pipeline script:
python3 infra/upload_scripts.py

# Pod — to pull latest without restarting:
python3 /app/download_scripts.py
```

## Directory Structure

### `videos/` — raw footage (gitignored)
Organized by subject: `videos/<name>/<name><position><group>.mp4`
- `<position>`: `a` = controlled (trainer-mounted bike, used as ground truth); `b` = real-world pass-by footage
- `<group>`: `30` = subject asked to target 60 RPM; `60` = subject asked to target 90 RPM. This is a cadence condition label, **not** the recording frame rate. Actual RPM varies per subject and is measured in Kinovea.
- Example: `videos/jenny/jennyb30.mp4` — Jenny, real-world, target 60 RPM condition

Original `.MOV` files from iPhone are converted to H.264 MP4 with:
```bash
bash convert_videos.sh   # converts all .MOV under videos/ to .mp4; skips if .mp4 already exists
```
Uses NVENC if available, falls back to libx264. Output written next to the source file.

### `output/` — pipeline outputs (gitignored)
One sub-directory per processed video, named by the video stem:
```
output/
  jennyb30/
    jennyb30_selected_frames/     ← Stage 1 frame images
    jennyb30_selection_log.json   ← Stage 1
    jennyb30_keypoints.json       ← Stage 2
    jennyb30_assessment.json      ← Stage 3
    jennyb30_rpm.json             ← Stage 4
    jennyb30_final.mp4            ← Stage 5
  alexb60/
    ...
```

### `samples/` — reference pipeline output
`samples/jennyb30/` contains the full pipeline output for `jennyb30.MOV`:
- `jennyb30.MOV` — source video
- `jennyb30_selection_log.json`, `jennyb30_selected_frames/` — Stage 1 output
- `jennyb30_keypoints.json` — Stage 2 output
- `jennyb30_assessment.json`, `jennyb30_rpm.json` — Stages 3 & 4 output
- `jennyb30_final.mp4` — Stage 5 annotated video

Use this as a reference for expected intermediate file content and for testing stages 3–5 without re-running OpenPose.

## Scripts — Usage

### Main pipeline
Run all stages at once with the pipeline runner:

```bash
python3 run_pipeline.py <video.mp4> [best.pt]
```

Prints a summary after each stage and a final results block. Or run stages individually — stages 1 and 3–5 run on any machine; stage 2 requires the RunPod GPU pod.

```bash
# Stage 1 — side-angle frame selection (cheap, YOLO only)
python3 pipeline/side_angle_select.py  <video.mp4> [best.pt]
# → output/<stem>/<stem>_selected_frames/  +  output/<stem>/<stem>_selection_log.json

# Stage 2 — OpenPose on selected frames (expensive, RunPod GPU)
python3 pipeline/pose_estimate.py  output/<stem>/<stem>_selection_log.json
# → output/<stem>/<stem>_keypoints.json

# Stage 3 — seat height assessment
python3 pipeline/seat_height.py  output/<stem>/<stem>_keypoints.json
# → output/<stem>/<stem>_assessment.json

# Stage 4 — RPM / cadence
python3 pipeline/rpm.py  output/<stem>/<stem>_keypoints.json
# → output/<stem>/<stem>_rpm.json

# Stage 5 — annotated output video
python3 pipeline/annotate_output.py  <video.mp4>  output/<stem>/<stem>_keypoints.json  output/<stem>/<stem>_assessment.json  output/<stem>/<stem>_rpm.json
# → output/<stem>/<stem>_final.mp4
```

### Pipeline script details

| Script | Input | Output | Key logic |
|---|---|---|---|
| `side_angle_select.py` | video.mp4 | selection_log.json + frame images | YOLO best.pt; both wheels near-square (±15%) + similar area (±20%); selects only the longest strictly consecutive burst of qualifying frames |
| `pose_estimate.py` | selection_log.json | keypoints.json | OpenPose Body25; 25 joints per selected frame; records inference time/frame |
| `seat_height.py` | keypoints.json | assessment.json | Law of cosines Hip→Knee→Ankle; peak extension assessed against 145–155° optimal range |
| `rpm.py` | keypoints.json (+ selection_log.json auto-detected) | rpm.json | Direction of travel from front/back wheel x-positions → camera-facing knee selected; longest contiguous run of frames used; Savitzky-Golay smoothing; adaptive-prominence peak detection (scipy) with autocorrelation fallback; inter-peak period → RPM |
| `annotate_output.py` | video + 3 JSONs | _final.mp4 | Draws Body25 skeleton from keypoints; verdict banner + RPM overlay; non-selected frames pass through |

### Intermediate file schemas

**`<video>_selection_log.json`** — output of Stage 1
```json
{
  "video": "alexb60.MOV",
  "model": "best.pt",
  "fps": 59.3,
  "total_frames": 1195,
  "selected_frames": [
    {
      "frame_idx": 321,
      "timestamp": 5.4149,
      "frame_file": "output/alexb60/alexb60_selected_frames/frame_0321.jpg",
      "front_wheel_conf": 0.9421,
      "back_wheel_conf": 0.9187,
      "front_squareness": 0.9823,
      "back_squareness": 0.9711,
      "size_match_ratio": 0.9542,
      "front_wheel_box": [x1, y1, x2, y2],
      "back_wheel_box":  [x1, y1, x2, y2]
    }
  ],
  "metrics": {
    "frames_processed": 1195,
    "frames_selected": 73,
    "total_bursts": 14,
    "selection_rate": 0.0611,
    "avg_confidence": 0.935,
    "elapsed_sec": 18.09
  }
}
```

**`<video>_keypoints.json`** — output of Stage 2
```json
{
  "video": "alexb60.MOV",
  "frames": [
    {
      "frame_idx": 321,
      "timestamp": 5.4149,
      "frame_file": "output/alexb60/alexb60_selected_frames/frame_0321.jpg",
      "inference_time_ms": 130.1,
      "keypoints": [[x, y, conf], ...],   // 25 joints, Body25 order
      "joint_confidences": { "Nose": 0.92, "RKnee": 0.87, ... }
    }
  ],
  "metrics": {
    "frames_processed": 73,
    "avg_inference_time_ms": 130.1,
    "avg_joint_confidence": { "Nose": 0.91, ... }
  }
}
```

**`<video>_assessment.json`** — output of Stage 3
```json
{
  "video": "alexb60.MOV",
  "frames": [
    {
      "frame_idx": 321,
      "timestamp": 5.4149,
      "right_knee_angle": 142.31,
      "left_knee_angle": 138.54,
      "right_hip_angle": 87.12,
      "left_hip_angle": 82.45
    }
  ],
  "summary": {
    "knee_angles_count": 73,
    "knee_angle_mean": 111.26,
    "knee_angle_std": 28.62,
    "knee_angle_peak": 162.7,
    "optimal_range": [145.0, 155.0],
    "verdict": "too_high",
    "verdict_detail": "Peak knee extension 162.7° exceeds 155.0°. ..."
  }
}
```

**`<video>_rpm.json`** — output of Stage 4
```json
{
  "video": "alexb60.MOV",
  "direction": "right",
  "knee_used": "left",
  "cadence_rpm": 85.0,
  "cycle_count": 2,
  "cycle_timestamps": [5.63, 6.34],
  "cycle_periods_sec": [0.71],
  "std_dev_rpm": 4.2,
  "rpm_method": "peak_detection",
  "best_run": {
    "frame_idx_start": 321,
    "frame_idx_end": 393,
    "frame_count": 73,
    "duration_sec": 1.21,
    "total_runs": 1
  },
  "metrics": {
    "frames_with_angle": 73,
    "frames_in_best_run": 73,
    "peaks_found": 2,
    "time_span_sec": 1.21
  }
}
```

`rpm_method` is `"peak_detection"` when ≥2 peaks found; `"autocorrelation"` when fewer peaks are found and the autocorrelation fallback succeeds. When `"autocorrelation"` is used, `cycle_timestamps` is `[]` and `cycle_periods_sec` contains one estimated period.

### Model classes
`best.pt` has exactly three classes: `cyclist`, `front_wheel`, `back_wheel`.

The `front_wheel` / `back_wheel` separation serves two purposes:
1. Side-angle gating — both wheels must be visible and near-square
2. Direction of travel detection — if `front_wheel` center x > `back_wheel` center x the cyclist moves right; used by `rpm.py` to select the camera-facing (non-occluded) knee

### Seat height thresholds
- `too_low`  : peak knee extension < 145° (seat too low — power loss, knee stress)
- `optimal`  : 145° ≤ peak ≤ 155°
- `too_high` : peak knee extension > 155° (over-extension risk)

Peak extension = maximum knee angle observed across all selected frames ≈ bottom of pedal stroke.

### Wheel-detection research scripts
Standalone experimental scripts for the report's wheel-detection comparison chapter.
They all use `yolo26s.pt` (person + bike classes 0/1) and output `<input>_<suffix>.mp4`.

| Script | Approach | Output suffix |
|---|---|---|
| `wheels.py` | YOLO bounding box → draw circle from box extents | `_wheels` |
| `circles.py` | YOLO crop → Hough circle transform | `_circles` |
| `arcs.py` | YOLO crop → RANSAC partial-arc fitting | `_arcs` |
| `contours.py` | Canny edges → contour circularity filter (no YOLO) | `_contours` |

```bash
python3 experiments/wheel_detection/wheels.py   <video.mp4>
python3 experiments/wheel_detection/circles.py  <video.mp4>
python3 experiments/wheel_detection/arcs.py     <video.mp4>
python3 experiments/wheel_detection/contours.py <video.mp4>
```

### Inference benchmark scripts
Five standalone versions of the YOLO inference pipeline for performance comparison in the report.

| Script | What changes vs base |
|---|---|
| `infer_base.py` | Baseline — single frame, CPU libx264, no GPU hints |
| `infer_opt1_fp16.py` | CUDA device + FP16 half-precision (`half=True`) |
| `infer_opt2_batch.py` | Batched inference (`BATCH_SIZE=8`) |
| `infer_opt3_skip.py` | Frame skipping — detect every `DETECT_EVERY=2` frames |
| `infer_opt4_all.py` | All of the above + NVENC GPU encode |

```bash
python3 experiments/inference_benchmarks/infer_base.py       <video.mp4>
python3 experiments/inference_benchmarks/infer_opt1_fp16.py  <video.mp4>
python3 experiments/inference_benchmarks/infer_opt2_batch.py <video.mp4>
python3 experiments/inference_benchmarks/infer_opt3_skip.py  <video.mp4>
python3 experiments/inference_benchmarks/infer_opt4_all.py   <video.mp4>
```

## Pipeline Script Evaluation Metrics
Each script prints these on exit — quote directly in the report:

| Stage | Metrics printed |
|---|---|
| `side_angle_select.py` | frames processed, bursts found, frames selected (longest burst), selection rate, avg confidence, elapsed time |
| `pose_estimate.py` | frames processed, avg inference time/frame, per-joint avg confidence |
| `seat_height.py` | knee angle count, mean, std dev, peak, verdict + detail string |
| `rpm.py` | frames with angle, contiguous runs found, best run (frame range + duration), peak count, cadence RPM ± std dev, direction of travel, knee used, RPM method (peak_detection / autocorrelation) |

## Local Lab Machine Setup (msc-linux-sls-016)

The pipeline runs directly on the lab machine without Docker. It has an NVIDIA RTX A4000 (16GB, sm_86) with CUDA 12.6.

### Environment
- OpenPose built from source at `~/openpose/build/`
- Python venv at `~/fyp/linux/` — activate with `source ~/fyp/linux/bin/activate`
- cuDNN installed via pip (`nvidia-cudnn-cu12`) at `~/fyp/linux/lib/python3.12/site-packages/nvidia/cudnn/`
- Models at `~/openpose/models/` (copied from `~/fyp/final-year-project/models/`)
- `~/.bashrc` sets PYTHONPATH, LD_LIBRARY_PATH, and activates the venv automatically

### Required env vars (set in ~/.bashrc)
```bash
export PYTHONPATH=/users/ugrad/oneilk10/openpose/build/python/openpose:$PYTHONPATH
export LD_LIBRARY_PATH=/users/ugrad/oneilk10/openpose/build/src/openpose:/users/ugrad/oneilk10/fyp/linux/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export OPENPOSE_MODELS=/users/ugrad/oneilk10/openpose/models/
```

`OPENPOSE_MODELS` is critical — `pose_estimate.py` defaults to `/openpose/models/` (the Docker path) and silently returns empty keypoints if it isn't set.

### Build notes
- pybind11 submodule updated to v2.11.1 (bundled v2.3 incompatible with Python 3.12)
- Line in `CMakeLists.txt` that resets pybind11 submodule was commented out to prevent it reverting
- `3rdparty/caffe/src/caffe/util/io.cpp` patched: `SetTotalBytesLimit` call reduced to 1 argument (protobuf 3.21+ compatibility)
- Body25 model must be a valid download — a corrupted `pose_iter_584000.caffemodel` causes silent no-detection

## Evaluation Pipeline

Ground truth for RPM is measured manually in **Kinovea** from the `a` (trainer/controlled) videos.
`<name>a<group>` videos are controlled (trainer-mounted); `<name>b<group>` are real-world pass-by footage.
`<group>` is `30` (target 60 RPM) or `60` (target 90 RPM) — a cadence condition label, not frame rate.

### Files
- `evaluation/ground_truth.csv` — one row per video: `video,true_rpm`. Fill in Kinovea RPM values here.
- `evaluation/evaluate.py` — runs both evaluations and prints a report table.

### Usage
```bash
# From project root:
python3 evaluation/evaluate.py
# Optional overrides:
python3 evaluation/evaluate.py --gt evaluation/ground_truth.csv --videos output/
```

The script searches `output/` recursively for `<stem>_rpm.json` and `<stem>_assessment.json` files.

### RPM evaluation output
Per-video table: true RPM, predicted RPM, absolute error, % error, method used.
Aggregate stats (MAE, RMSE, mean % error) broken down by:
- Condition `a` (trainer) vs `b` (real-world)
- Cadence group: `30` (target 60 RPM) vs `60` (target 90 RPM)
- RPM method: `peak_detection` vs `autocorrelation`

### Seat height evaluation output
Automatically pairs `<name>a<group>` ↔ `<name>b<group>` for each subject.
Reports per-pair verdict agreement (`too_low` / `optimal` / `too_high`) and peak angle delta between conditions.

## TODOs
- [ ] Choose best wheel-detection approach from research scripts and document rationale in report
- [ ] Fill in evaluation/ground_truth.csv with Kinovea RPM measurements
- [ ] Preprocessing before OpenPose (in `pipeline/pose_estimate.py`):
  - [ ] ROI crop to YOLO cyclist bounding box (highest priority — fixes small subject in large frame)
  - [ ] Motion deblur / unsharp mask to counteract lateral motion blur from pass-by footage
  - [ ] CLAHE on LAB L-channel for variable outdoor lighting
  - [ ] Pad crop to square before resizing to net_resolution (avoid aspect ratio distortion)
  - [ ] Tune net_resolution (try `656x368` or `736x368` for better joint localisation on small subjects)

## Final Year Project Report
A detailed written report is required covering approach and findings. It should document:
- **Motivation & problem statement** — why camera angle matters for cycling posture analysis; the challenge of detecting a true side-on view automatically; real-world vs. trainer-mounted distinction
- **Literature / related work** — pose estimation (OpenPose vs. MediaPipe/YOLOPose); object detection (YOLO); classical CV wheel-detection methods (Hough, RANSAC, contour filtering); existing cycling analysis systems
- **System design** — the full 5-stage pipeline; computational justification for YOLO gating before OpenPose; why intermediate JSON files enable modular evaluation
- **Wheel-detection experiments** — comparison of the four approaches (`wheels.py`, `circles.py`, `arcs.py`, `contours.py`): methodology, results, failure cases, rationale for chosen approach
- **Side-angle detection** — the bounding-box squareness + size-match heuristic; quantitative evaluation (selection rate, false positive/negative analysis)
- **Pose estimation & angle measurement** — joint keypoint extraction, law-of-cosines angle calculation, seat height thresholds and verdict logic
- **RPM calculation** — knee-angle cycle counting methodology; direction-of-travel detection from front/back wheel positions; contiguous-run selection to avoid inter-window noise; accuracy vs. ground truth
- **Computational evaluation** — frames selected vs. total, OpenPose calls saved, inference time/frame, end-to-end latency
- **Results & evaluation** — qualitative and quantitative assessment on real cycling video; seat height verdict accuracy; RPM accuracy
- **Conclusions & future work** — limitations, what would improve accuracy, potential real-world deployment
