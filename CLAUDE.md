# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Final Year Project — Cycling Posture Analysis

## Project Overview
A cycling posture analysis system for real-world smartphone footage of a cyclist moving past a camera. Detects the moment the cyclist is at a true side-on angle, runs OpenPose only on those frames to extract joint keypoints, then measures angles to assess seat height and compute pedalling RPM.

**Research gap:** existing systems assume a bike mounted on a trainer (controlled environment). This system works on real-world pass-by footage.

**Why OpenPose:** Body25 skeleton is the standard in academic biomechanics literature and has been more thoroughly validated for lower-limb angle measurement than newer alternatives (MediaPipe, YOLOPose). Required by supervisor.

**Core computational insight:** YOLO side-angle gating runs at ~30fps on CPU; OpenPose takes ~1–2s per frame on GPU. Selecting only good side-angle frames (e.g. 67 from 900) reduces OpenPose calls ~13×.

## Stack
- **OpenPose** (`pyopenpose`) — Body25 model, 25-joint pose estimation
- **YOLO** (`ultralytics`) — Person + bicycle detection; custom model `yolo26s.pt` used by wheel-detection scripts
- **OpenCV** — Video I/O, annotation
- **NumPy** — Kalman filter, angle maths (law of cosines)
- **Docker** — Base: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **RunPod** — GPU cloud deployment target

## Pipelines

There are two parallel pipelines. **Pipeline 1** is the stable baseline; **Pipeline V2** is the working copy used for experiments and improvements. Run both on the same video and compare outputs.

> **IMPORTANT: Do not modify any scripts in `pipeline/` (Pipeline 1).** Pipeline 1 is frozen as the comparison baseline. All improvements and fixes go into `pipeline_v2/` only.

| | Pipeline 1 | Pipeline V2 |
|---|---|---|
| Scripts | `pipeline/` | `pipeline_v2/` |
| Runner | `run_pipeline.py` | `run_pipeline_v2.py` |
| Outputs | `output/<stem>/` | `output_v2/<stem>/` |

```bash
python3 run_pipeline.py    <video.mp4>   # Pipeline 1 (baseline)
python3 run_pipeline_v2.py <video.mp4>   # Pipeline V2 (experimental)
```

**Pipeline 1** follows a 5-stage structure. **Pipeline V2** has 6 stages — it adds a dedicated knee analysis stage (Stage 3) as the single source of truth for direction detection, knee selection, smoothing, and peak detection. Stages 4 and 5 in V2 consume its output rather than repeating the analysis independently.

**Pipeline 1 (5 stages):**
```
smartphone_video.mp4
  → side_angle_select.py  → output/<stem>/<stem>_selected_frames/
  │                          output/<stem>/<stem>_selection_log.json
  → pose_estimate.py      → output/<stem>/<stem>_keypoints.json
  → seat_height.py        → output/<stem>/<stem>_assessment.json
  → rpm.py                → output/<stem>/<stem>_rpm.json
  → annotate_output.py    → output/<stem>/<stem>_final.mp4
```

**Pipeline V2 (6 stages):**
```
smartphone_video.mp4
  → side_angle.py         → output_v2/<stem>/<stem>_selected_frames/
  │                          output_v2/<stem>/<stem>_selection_log.json
  │   (YOLO on every frame — cheap, runs locally or on pod)
  │   Detects when both wheels visible + near-square aspect ratio;
  │   quality-weighted burst selection keeps all bursts scoring >=50%
  │   of best (score = len × squareness × size-match × norm. height)
  │
  → pose_estimate.py      → output_v2/<stem>/<stem>_keypoints.json
  │   (OpenPose only on selected frames — expensive, cloud GPU)
  │   25-joint Body25 keypoints per selected frame; ROI crop + CLAHE +
  │   unsharp mask + square pad preprocessing; frames missing cyclist_box
  │   fall back to wheel-position crop at median aspect ratio (not skipped)
  │
  → knee_analysis.py      → output_v2/<stem>/<stem>_knee_analysis.json
  │   (Single source of truth for pedal cycle analysis)
  │   Per-burst direction detection (load_direction_map); moving right →
  │   right knee, moving left → left knee; processes every contiguous run
  │   independently — Savitzky-Golay smoothing, adaptive-prominence peak
  │   detection (scipy), autocorrelation fallback per run; outputs runs[]
  │   array + aggregated peaks + angle_series
  │
  → seat_height.py        → output_v2/<stem>/<stem>_assessment.json
  │   (Reads knee_analysis.json only — no keypoints re-read)
  │   peak_mean  : used when >=10 validated peaks exist across usable runs
  │               (long/trainer recordings with many complete cycles)
  │   smooth_p80 : 80th percentile of the SG-smoothed angle series; used
  │               for short real-world pass-by windows with <10 peaks —
  │               clips perspective-distortion inflation without trainer data
  │
  → rpm.py                → output_v2/<stem>/<stem>_rpm.json
  │   (Reads knee_analysis.json only)
  │   Pools all inter-peak periods from usable runs → single mean cadence
  │   RPM; autocorr fallback from first run with a valid period
  │
  → annotate_output.py    → output_v2/<stem>/<stem>_final.mp4
      (Overlay skeleton + angles + verdict + RPM on original video)
```

Each stage creates its output directory if it doesn't exist, so stages can be run standalone without the runner. Each stage also outputs intermediate files so stages can be re-run independently and evaluated in isolation for the report.

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

### `output/` — Pipeline 1 outputs (gitignored)
### `output_v2/` — Pipeline V2 outputs (gitignored)
Both follow the same structure. One sub-directory per processed video, named by the video stem:
```
output/          ← Pipeline 1 (5 stages)
  jennyb30/
    jennyb30_selected_frames/         ← Stage 1
    jennyb30_selection_log.json       ← Stage 1
    jennyb30_keypoints.json           ← Stage 2
    jennyb30_assessment.json          ← Stage 3
    jennyb30_rpm.json                 ← Stage 4
    jennyb30_final.mp4                ← Stage 5

output_v2/       ← Pipeline V2 (6 stages)
  jennyb30/
    jennyb30_selected_frames/         ← Stage 1
    jennyb30_selection_log.json       ← Stage 1
    jennyb30_keypoints.json           ← Stage 2
    jennyb30_preprocessing_steps/     ← Stage 2 — per-frame montages showing each preprocessing step
    jennyb30_knee_analysis.json       ← Stage 3
    jennyb30_assessment.json          ← Stage 4
    jennyb30_rpm.json                 ← Stage 5
    jennyb30_final.mp4                ← Stage 6
  alexb60/
    ...
```

### `samples/` — reference pipeline output
`samples/jennyb30/` contains the full pipeline output for `jennyb30.MOV`:
- `jennyb30.MOV` — source video
- `jennyb30_selection_log.json`, `jennyb30_selected_frames/` — Stage 1 output
- `jennyb30_keypoints.json` — Stage 2 output
- `jennyb30_assessment.json`, `jennyb30_rpm.json` — Stages 3 & 4 output (Pipeline 1 format)
- `jennyb30_final.mp4` — Stage 5 annotated video

Use this as a reference for Pipeline 1 intermediate file formats and for testing stages 3–5 without re-running OpenPose. Note: no `_knee_analysis.json` in samples as that is V2-only.

## Scripts — Usage

### Running the pipelines
Run all stages at once with the pipeline runner. All stages except Stage 2 run on any machine; Stage 2 (OpenPose) requires the RunPod GPU pod.

```bash
# Pipeline 1 (baseline — do not modify these scripts)
python3 run_pipeline.py <video.mp4> [best.pt]

# Pipeline V2 (experimental — make changes here)
python3 run_pipeline_v2.py <video.mp4> [best.pt]
```

Both runners print a summary after each stage and a final results block.

**Pipeline 1 — individual stages:**
```bash
python3 pipeline/side_angle_select.py  <video.mp4> [best.pt]
# → output/<stem>/<stem>_selected_frames/  +  output/<stem>/<stem>_selection_log.json

python3 pipeline/pose_estimate.py  output/<stem>/<stem>_selection_log.json
# → output/<stem>/<stem>_keypoints.json

python3 pipeline/seat_height.py  output/<stem>/<stem>_keypoints.json
# → output/<stem>/<stem>_assessment.json

python3 pipeline/rpm.py  output/<stem>/<stem>_keypoints.json
# → output/<stem>/<stem>_rpm.json

python3 pipeline/annotate_output.py  <video.mp4>  output/<stem>/<stem>_keypoints.json  output/<stem>/<stem>_assessment.json  output/<stem>/<stem>_rpm.json
# → output/<stem>/<stem>_final.mp4
```

**Pipeline V2 — individual stages:**
```bash
python3 pipeline_v2/side_angle.py  <video.mp4> [best.pt]
# → output_v2/<stem>/<stem>_selected_frames/  +  output_v2/<stem>/<stem>_selection_log.json

python3 pipeline_v2/pose_estimate.py  output_v2/<stem>/<stem>_selection_log.json
# → output_v2/<stem>/<stem>_keypoints.json

python3 pipeline_v2/knee_analysis.py  output_v2/<stem>/<stem>_keypoints.json
# → output_v2/<stem>/<stem>_knee_analysis.json

python3 pipeline_v2/seat_height.py  output_v2/<stem>/<stem>_knee_analysis.json
# → output_v2/<stem>/<stem>_assessment.json

python3 pipeline_v2/rpm.py  output_v2/<stem>/<stem>_knee_analysis.json
# → output_v2/<stem>/<stem>_rpm.json

python3 pipeline_v2/annotate_output.py  <video.mp4>  output_v2/<stem>/<stem>_keypoints.json  output_v2/<stem>/<stem>_assessment.json  output_v2/<stem>/<stem>_rpm.json
# → output_v2/<stem>/<stem>_final.mp4
```

### Pipeline script details

Pipeline 1:

| Script | Input | Output | Key logic |
|---|---|---|---|
| `side_angle_select.py` | video.mp4 | selection_log.json + frame images | YOLO best.pt; both wheels near-square (±15%) + similar area (±20%); selects only the longest strictly consecutive burst of qualifying frames |
| `pose_estimate.py` | selection_log.json | keypoints.json | OpenPose Body25; 25 joints per selected frame; records inference time/frame |
| `seat_height.py` | keypoints.json | assessment.json | Law of cosines Hip→Knee→Ankle; peak = max angle across all frames; assessed against 145–155° optimal range |
| `rpm.py` | keypoints.json (+ selection_log.json auto-detected) | rpm.json | Direction detection, knee selection, contiguous-run selection, Savitzky-Golay smoothing, adaptive-prominence peak detection, autocorrelation fallback; inter-peak period → RPM |
| `annotate_output.py` | video + 3 JSONs | _final.mp4 | Draws Body25 skeleton from keypoints; verdict banner + RPM overlay; non-selected frames pass through |

Pipeline V2 (differences from Pipeline 1):

All pipeline_v2 scripts import shared helpers from **`pipeline_v2/utils.py`** (`CONF_MIN`, `calc_angle`, `get_xy`, `video_stem`, `print_progress`). This file must be present when running any V2 stage, and must be included when uploading scripts to R2.

| Script | Input | Output | Key logic / V2 changes |
|---|---|---|---|
| `utils.py` *(V2 only)* | — | — | Shared helpers: `CONF_MIN`, `calc_angle`, `get_xy`, `video_stem`, `print_progress` — note: `seat_height.py` only imports `video_stem` |
| `side_angle.py` | video.mp4 | selection_log.json + frame images | **Quality-weighted burst selection**: scores each burst as `len × mean_squareness × mean_size_ratio × mean_normalised_cyclist_height`; keeps all bursts scoring ≥ `QUALITY_FRACTION` (50%) of best burst with ≥ `MIN_BURST_FRAMES` (5); adds `selected_bursts` metadata to log; also saves `cyclist_box` per frame |
| `pose_estimate.py` | selection_log.json | keypoints.json | **ROI crop to cyclist box, CLAHE, unsharp mask, square pad, net_resolution=656x368; keypoints transformed back to original-frame coordinates; step montages saved to `_preprocessing_steps/`**; if a selected frame has no `cyclist_box` (YOLO detected wheels but not cyclist class), estimates crop from wheel positions + median h/w aspect ratio of frames that do have a box — preserves scale rather than skipping; estimated box drawn in yellow in montage vs green for detected |
| `knee_analysis.py` *(V2 only)* | keypoints.json | knee_analysis.json | **Per-burst direction detection** from wheel x-positions (`load_direction_map`); knee selected independently per burst so opposing-direction bursts in the same clip are handled correctly; camera-facing knee selection; **processes every contiguous run independently** — Savitzky-Golay smoothing (`SAVGOL_WINDOW=11`), adaptive-prominence peak detection with `PEAK_PROMINENCE=25.0°` floor (scipy), autocorrelation fallback per run; outputs `runs[]` array + aggregated `peaks` + `angle_series` at top level for downstream compat |
| `seat_height.py` | knee_analysis.json | assessment.json | **Reads knee_analysis.json only — no keypoints re-read; adaptive peak selection: `peak_mean` (mean of validated bottom-of-stroke peaks) when ≥`PEAK_MEAN_MIN_PEAKS` (10) peaks exist — reliable for long trainer recordings; `smooth_p80` (80th percentile of SG-smoothed angle series from knee_analysis runs) for short real-world pass-by windows with <10 peaks; `peak_angle_method` field records which was used; mean/std in summary derived from smoothed angle series** |
| `rpm.py` | knee_analysis.json | rpm.json | **Pools all inter-peak periods from every run with ≥2 peaks → single mean cadence RPM**; autocorr fallback from first run with a valid period; adds `per_run_rpms` list; forwards aggregated angle_series + peak_timestamps for annotate |
| `annotate_output.py` | video + 3 JSONs | _final.mp4 | **nvenc → libx264 fallback** on encode failure; graph layout pre-computed once before render loop |

### Model classes
`best.pt` has exactly three classes: `cyclist`, `front_wheel`, `back_wheel`.

The `front_wheel` / `back_wheel` separation serves two purposes:
1. Side-angle gating — both wheels must be visible and near-square
2. Direction of travel detection — if `front_wheel` center x > `back_wheel` center x the cyclist moves right; used by `knee_analysis.py` (V2) / `rpm.py` (P1) to select the camera-facing (non-occluded) knee: moving right → right knee; moving left → left knee. In V2, direction is computed **per burst** (`load_direction_map`) so opposing-direction bursts within a single clip are handled correctly; top-level `direction`/`knee_used` in the output reflect the longest burst

### Seat height thresholds
- `too_low`  : peak knee extension < 145° (seat too low — power loss, knee stress)
- `optimal`  : 145° ≤ peak ≤ 155°
- `too_high` : peak knee extension > 155° (over-extension risk)

Peak extension:
- **Pipeline 1** — maximum knee angle across all selected frames (`max()`)
- **Pipeline V2** — adaptive selection via `peak_angle_method`:
  - `peak_mean` — mean of validated bottom-of-stroke angles from all usable runs; used when ≥10 total validated peaks exist (typical of long trainer recordings with many complete cycles)
  - `smooth_p80` — 80th percentile of the Savitzky-Golay smoothed angle series concatenated across all runs; used for short real-world pass-by windows (<10 peaks). Clips the inflated tail caused by perspective distortion without requiring trainer reference data. Evaluated against trainer videos as reference: 79% verdict agreement vs 53% for raw_p90.

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
      "front_wheel_box": [x1, y1, x2, y2],  // P1 only
      "back_wheel_box":  [x1, y1, x2, y2],  // P1 only
      "fw_box":          [x1, y1, x2, y2],  // V2 only
      "bw_box":          [x1, y1, x2, y2],  // V2 only
      "cyclist_box":     [x1, y1, x2, y2]   // V2 only
    }
  ],
  "selected_bursts": [                       // V2 only — one entry per kept burst
    {
      "burst_id": 0,
      "quality_score": 3.142,
      "start_frame_idx": 321,
      "end_frame_idx": 393,
      "frame_count": 73
    }
  ],
  "metrics": {
    "frames_processed": 1195,
    "frames_selected": 73,
    "total_bursts": 14,
    "bursts_selected": 2,                    // V2 only
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
    "avg_joint_confidence": { "Nose": 0.91, ... },
    "preprocessing": {           // V2 only
      "roi_crop": true,
      "clahe": true,
      "unsharp_mask": true,
      "square_pad": true,
      "net_resolution": "656x368"
    }
  }
}
```

**`<video>_knee_analysis.json`** — output of Stage 3 (V2 only)
```json
{
  "video": "alexb60.MOV",
  "direction": "right",
  "knee_used": "right",
  "runs": [
    {
      "run_id": 0,
      "frame_idx_start": 160,
      "frame_idx_end": 215,
      "frame_count": 56,
      "duration_sec": 0.93,
      "peaks": [
        { "frame_idx": 175, "timestamp": 2.92, "angle": 158.2 }
      ],
      "peak_method": "peak_detection",
      "autocorr_period_sec": null,
      "angle_series": [[2.67, 142.3], [2.68, 145.6]]
    }
  ],
  "best_run": {                              // longest run — kept for backward compat
    "frame_idx_start": 160,
    "frame_idx_end": 215,
    "frame_count": 56,
    "duration_sec": 0.93,
    "total_runs": 2
  },
  "peaks": [...],                            // aggregated across all runs (for annotate)
  "peak_method": "peak_detection",           // method of longest run
  "autocorr_period_sec": null,
  "angle_series": [...],                     // concatenated across all runs (for annotate)
  "metrics": {
    "frames_with_angle": 73,
    "total_runs": 2,
    "usable_runs": 1,                        // runs with >= 2 peaks
    "peaks_found": 2,
    "time_span_sec": 1.21
  }
}
```

`runs[].peak_method` is `"peak_detection"` when ≥2 peaks found in that run; `"autocorrelation"` when the autocorrelation fallback succeeded for that run. `peaks[].angle` is the raw (unsmoothed) knee angle at each detected bottom-of-stroke frame. `rpm.py` and `seat_height.py` both consume `runs[]` directly (seat_height takes knee_analysis.json as its sole input); `annotate_output.py` uses the top-level `peaks` and `angle_series`.

**`<video>_assessment.json`** — output of Stage 3 (P1) / Stage 4 (V2)

Pipeline 1 format includes a `frames[]` array with per-frame knee and hip angles. V2 omits `frames[]` — all angle data lives in `knee_analysis.json`.

```json
{
  "video": "alexb60.MOV",
  "frames": [                          // Pipeline 1 only
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
    "knee_angles_count": 73,           // V2: from knee_analysis metrics.frames_with_angle
    "knee_angle_mean": 111.26,         // V2: mean of smoothed angle series
    "knee_angle_std": 28.62,           // V2: std of smoothed angle series
    "knee_angle_peak": 162.7,
    "peak_angle_method": "peak_mean",  // V2: "peak_mean" | "smooth_p80"
    "optimal_range": [145.0, 155.0],
    "verdict": "too_high",
    "verdict_detail": "Peak knee extension 162.7° exceeds 155.0°. ..."
  }
}
```

**`<video>_rpm.json`** — output of Stage 4 (P1) / Stage 5 (V2)
```json
{
  "video": "alexb60.MOV",
  "direction": "right",
  "knee_used": "right",
  "cadence_rpm": 85.0,
  "cycle_count": 2,
  "cycle_timestamps": [5.63, 6.34],
  "cycle_periods_sec": [0.71],
  "std_dev_rpm": 4.2,
  "rpm_method": "peak_detection",
  "per_run_rpms": [                          // V2 only — RPM per usable run
    { "run_id": 0, "rpm": 85.0, "peak_count": 2 }
  ],
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
    "total_runs": 1,
    "usable_runs": 1,                        // V2 only — runs with >= 2 peaks
    "peaks_found": 2,
    "time_span_sec": 1.21
  }
}
```

`rpm_method` is `"peak_detection"` when any run has ≥2 peaks; `"autocorrelation"` when no run has ≥2 peaks but at least one run produced a valid autocorrelation period. When `"autocorrelation"` is used, `cycle_timestamps` is `[]` and `cycle_periods_sec` contains one estimated period. In V2, `cadence_rpm` is the mean over all pooled inter-peak periods across all usable runs (not just the longest run).

### Evaluation metrics
Each script prints these on exit — quote directly in the report:

| Stage | Script | Metrics printed |
|---|---|---|
| 1 | `side_angle_select.py` (P1) | frames processed, bursts found, frames selected (longest burst), selection rate, avg confidence, elapsed time |
| 1 | `side_angle.py` (V2) | frames processed, total bursts found, bursts selected (with quality threshold), per-burst frame range + score, total frames selected, selection rate, avg confidence, elapsed time |
| 2 | `pose_estimate.py` | frames processed, avg inference time/frame, per-joint avg confidence |
| 3 (V2) | `knee_analysis.py` | frames with angle, runs total / usable, per-run frame range + duration + peak count + method, total peaks |
| 3 (P1) / 4 (V2) | `seat_height.py` | knee angle count, mean, std dev, peak, peak_angle_method, verdict + detail string |
| 4 (P1) / 5 (V2) | `rpm.py` | frames with angle, runs total / usable, per-run RPM, mean cadence RPM ± std dev, direction of travel, knee used, RPM method |

## Experiment Scripts

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

## Evaluation Pipeline

Ground truth for RPM is measured manually in **Kinovea** from the `a` (trainer/controlled) videos.
`<name>a<group>` videos are controlled (trainer-mounted); `<name>b<group>` are real-world pass-by footage.
`<group>` is `30` (target 60 RPM) or `60` (target 90 RPM) — a cadence condition label, not frame rate.

### Files
- `evaluation/ground_truth.csv` — one row per video: `video,true_rpm`. Fill in Kinovea RPM values here.
- `evaluation/full_eval.py` — **comprehensive V2 evaluation report** (8 sections): dataset overview, RPM accuracy by condition/group/method, P1 vs V2 RPM comparison, trainer internal consistency (a30 vs a60), a/b seat-height verdict agreement by filming group, peak-angle method distribution, frame selection statistics, and failure cases. Output tee'd to `evaluation/full_eval.txt`. Use this for the report; `evaluate.py` is an older narrower summary.
- `evaluation/evaluate.py` — runs both evaluations and prints a report table. Automatically detects which pipeline output directories exist and shows a side-by-side comparison when both are present.
- `evaluation/compare_angles.py` — cross-video angle consistency comparison across both pipelines.
- `evaluation/angle_ceiling_eval.py` — trainer-independent peak selection strategy evaluation. For each subject with paired a/b videos in `output_v2/`, compares `current_v2`, `smooth_max`, `smooth_p80`–`smooth_p95`, and `raw_p80`–`raw_p95` against the a-condition verdict as reference. Raw angles computed directly from `keypoints.json` (not from `assessment.json`). Shows per-subject peak angles, verdict agreement counts, and correction of inflated `too_high` readings. Run with `python3 evaluation/angle_ceiling_eval.py [--v2 output_v2/]`.

### Sensitivity Sweeps
Scripts in `evaluation/sensitivity/` vary individual hyperparameters and report RPM MAE + seat-height agreement; results saved to `evaluation/sensitivity/results/`. All reconstruct raw angles from `_keypoints.json` — no OpenPose re-run needed.

| Script | Parameter swept | Values |
|---|---|---|
| `sweep_savgol.py` | Savitzky-Golay window (`SAVGOL_WINDOW`) | 7, 9, 11, 13, 15 |
| `sweep_prominence.py` | Minimum adaptive peak prominence (`PEAK_PROMINENCE`) | 10, 15, 20, 25, 30° |
| `sweep_quality_fraction.py` | Burst keep threshold (`QUALITY_FRACTION`) | 0.5–1.0 (raising only; lowering requires re-running YOLO) |
| `sweep_square_tol.py` | Wheel squareness gate (`SQUARE_TOL`) | 0.08, 0.10, 0.12, 0.15 (simulated from logs); 0.18 requires YOLO re-run |
| `sweep_percentile.py` | Percentile used in `smooth_pN` strategy | varies |
| `sweep_peak_threshold.py` | Peak detection threshold | varies |

```bash
python3 evaluation/sensitivity/sweep_savgol.py        [--v2 output_v2/] [--gt evaluation/ground_truth.csv]
python3 evaluation/sensitivity/sweep_prominence.py    [--v2 output_v2/] [--gt evaluation/ground_truth.csv]
python3 evaluation/sensitivity/sweep_quality_fraction.py [--v2 output_v2/] [--gt evaluation/ground_truth.csv]
python3 evaluation/sensitivity/sweep_square_tol.py    [--v2 output_v2/] [--gt evaluation/ground_truth.csv]
```

`evaluation/sensitivity/preprocessing_ablation_visual.py` — qualitative only (no numerical ablation). Selects representative preprocessing montage frames from `_preprocessing_steps/` and copies them into `evaluation/sensitivity/results/preprocessing_examples/` for inclusion in the report. Run with `python3 evaluation/sensitivity/preprocessing_ablation_visual.py [--v2 output_v2/]`.

### Usage
```bash
# Comprehensive evaluation (preferred for report — 8-section output):
python3 evaluation/full_eval.py

# Optional overrides:
python3 evaluation/full_eval.py --v2 output_v2/ --v1 output/ --gt evaluation/ground_truth.csv

# Narrower side-by-side P1/V2 comparison:
python3 evaluation/evaluate.py

# Angle consistency comparison across all videos and both pipelines:
python3 evaluation/compare_angles.py
python3 evaluation/compare_angles.py --p1 output/ --p2 output_v2/
```

The script searches the output directories recursively for `<stem>_rpm.json` and `<stem>_assessment.json` files.

**Behaviour by directory availability:**
- Both `output/` and `output_v2/` present → side-by-side comparison (P1 vs P2 columns + per-pipeline aggregate stats + V2 improvement summary)
- Only `output/` present → Pipeline 1 results only
- Only `output_v2/` present → Pipeline V2 results only

### RPM evaluation output
**Single-pipeline mode:** per-video table with true RPM, predicted RPM, absolute error, % error, method.
**Comparison mode:** per-video table with True / P1 Pred / P1 Err / P1 % / P2 Pred / P2 Err / P2 % columns, then separate aggregate blocks for each pipeline, then a V2 vs V1 improvement count and mean |error| reduction.

Aggregate stats (MAE, RMSE, mean % error) broken down by:
- Condition `a` (trainer) vs `b` (real-world)
- Cadence group: `30` (target 60 RPM) vs `60` (target 90 RPM)
- RPM method: `peak_detection` vs `autocorrelation`

### Seat height evaluation output
Automatically pairs `<name>a<group>` ↔ `<name>b<group>` for each subject.
Reports per-pair verdict agreement (`too_low` / `optimal` / `too_high`) and peak angle delta between conditions.
When both pipelines are present, two labelled blocks are printed (Pipeline 1, then Pipeline V2).

### Angle consistency comparison output (`compare_angles.py`)
Three sections, missing files skipped silently:
1. **Per-video table** — peak°, mean°, std, frame count, verdict for every `*_assessment.json` found, grouped by subject
2. **Within-subject consistency** — min–max peak angle range per subject per pipeline; should be small since the same bike is used across all recordings for each subject
3. **Pipeline diff** — for stems present in both `output/` and `output_v2/`, shows peak angle delta and verdict change side by side (populated once V2 assessments exist)

## Infrastructure & Deployment

### Docker Image
- `infra/Dockerfile` — build config; `infra/DOCKER_HUB.md` — push/pull instructions
- CUDA 11.8 + cuDNN 8 on Ubuntu 22.04
- OpenPose compiled from source with Python bindings (`BUILD_PYTHON=ON`), cuDNN + CUDA enabled
- GPU architectures: 60, 61, 62, 70, 72, 75, 80, 86, 89, 90
- On startup: run `infra/startup.sh` — installs Python deps (torch for **cu124** — RTX 4090 pods have CUDA 12.4 driver, cu121/cu13 builds fail CUDA init), then fetches `download_scripts.py` from R2 via `download.py`, then pulls all pipeline scripts to `/app`
- OpenPose models in `/openpose/models/` — baked into the image at build time via `COPY models/`; NOT in git (too large)
- Scripts are NOT bundled in the image — pulled from R2 at startup so updates don't require a rebuild

### Key Constraints
- **Model weights** are excluded from git (`.gitignore` ignores `/models`)
- **Raw footage** (`videos/`) is gitignored — too large for git; transfer via R2
- **runpodctl** is blocked on the college network — cannot use it for file transfer
- **HTTP transfers from laptop** time out on large files — naive HTTP upload/download direct to laptop is not viable; use the login server instead (see File Transfer section)
- Target workflow: upload input video to pod → run pipeline → retrieve output files

### File Transfer
Large files are transferred via **Cloudflare R2** (S3-compatible, HTTPS/port 443):
- `infra/upload.py <src1> [src2 ...] <r2_folder>` — uploads files or directories to R2 under the given folder prefix; directories are walked recursively; last argument is always the R2 folder
- `infra/download.py <key> [dest]` — downloads a single file from R2
- `infra/upload_scripts.py` — uploads all pipeline scripts to R2 under `scripts/` prefix; run locally after any script change
- `infra/download_scripts.py [dest_dir]` — downloads all pipeline scripts from R2; run on pod to get latest versions
- Credentials via `.env`: `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
- `boto3.upload_file` handles multipart automatically — suitable for large videos

**Uploading output folders from the pod:** tar the directory first to avoid per-file HTTP overhead (hundreds of JPEGs = hundreds of round-trips):
```bash
tar -czf /tmp/<stem>.tar.gz -C /app/data/output_v2 <stem>
python3 /app/infra/upload.py /tmp/<stem>.tar.gz output_v2
rm /tmp/<stem>.tar.gz
```
Extract locally with `tar -xzf <stem>.tar.gz`.

**Preferred method for retrieving large output folders from the pod to the college drive:**
Do NOT download via the CIFS mount on the laptop — write speed is ~2.6 MB/s and concurrent writes will freeze the system. Instead, use the college login server (`macneill.scss.tcd.ie`) which has direct fast access to the college drive:

1. On the pod, tar the output directory (skip `-z` compression — JPEGs don't compress and it's much faster without):
```bash
tar -cf /app/data/output_v2.tar -C /app output_v2
```
2. Expose the pod's HTTP server (port 8000) via RunPod's proxy, then on the login server:
```bash
curl -o ~/fyp/final-year-project/output_v2.tar https://<pod-proxy-url>/data/output_v2.tar
```
3. Extract directly into the output directory:
```bash
tar -xf ~/fyp/final-year-project/output_v2.tar --strip-components=1 -C ~/fyp/final-year-project/output_v2/
```

**Downloading from R2 to the college drive:** Use `wget` on the login server with the R2 public URL — no Python/boto3 needed, and it runs at ~30 MB/s on the college network:
```bash
wget https://pub-<id>.r2.dev/<key>.tar.gz -O ~/fyp/final-year-project/<key>.tar.gz
```

**Pod storage layout:** pipeline outputs live in `/app/data/output_v2/` (the persistent network volume), not in `/app/output_v2/`. The container root (`/`) is a separate 20 GB overlay — keep it clear of large files.

### Pod Setup & Operation

**Setup workflow:**
1. On first pod start, `infra/download.py` and `.env` must already be in `/app` (paste via web terminal or bake into image)
2. `infra/startup.sh` uses `download.py` to fetch `download_scripts.py` from R2, then runs it to pull all other scripts
3. After that, re-running `infra/startup.sh` always gets the latest script versions from R2

**Updating scripts:**
```bash
# Local — after editing any pipeline script:
python3 infra/upload_scripts.py

# Pod — to pull latest without restarting:
python3 /app/download_scripts.py
```

**Operation tips:**
- **Use Ctrl+C to interrupt a run, not Ctrl+Z.** Ctrl+Z suspends the process but leaves it alive in memory, holding the GPU CUDA/cuDNN context. Any subsequent OpenPose run will fail with `CUDNN_STATUS_NOT_INITIALIZED` because the GPU is already claimed by the suspended process.
- **If `CUDNN_STATUS_NOT_INITIALIZED` appears:** there is a stale Python process holding the GPU. Fix: `pkill -9 -f python3`, then retry.
- **Multiple SSH sessions are safe** — each SSH connection is independent. You can monitor GPU usage (`nvidia-smi -l 1`) in one terminal while the pipeline runs in another. Do not run two OpenPose stages simultaneously as they will conflict on the GPU.

### Local Lab Machine Setup (msc-linux-sls-016)

The pipeline runs directly on the lab machine without Docker. It has an NVIDIA RTX A4000 (16GB, sm_86) with CUDA 12.6.

**Environment:**
- OpenPose built from source at `~/openpose/build/`
- Python venv at `~/fyp/linux/` — activate with `source ~/fyp/linux/bin/activate`
- cuDNN installed via pip (`nvidia-cudnn-cu12`) at `~/fyp/linux/lib/python3.12/site-packages/nvidia/cudnn/`
- Models at `~/openpose/models/` (copied from `~/fyp/final-year-project/models/`)
- `~/.bashrc` sets PYTHONPATH, LD_LIBRARY_PATH, and activates the venv automatically

**Required env vars (set in ~/.bashrc):**
```bash
export PYTHONPATH=/users/ugrad/oneilk10/openpose/build/python/openpose:$PYTHONPATH
export LD_LIBRARY_PATH=/users/ugrad/oneilk10/openpose/build/src/openpose:/users/ugrad/oneilk10/fyp/linux/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export OPENPOSE_MODELS=/users/ugrad/oneilk10/openpose/models/
```

`OPENPOSE_MODELS` is critical — `pose_estimate.py` defaults to `/openpose/models/` (the Docker path) and silently returns empty keypoints if it isn't set.

**Build notes:**
- pybind11 submodule updated to v2.11.1 (bundled v2.3 incompatible with Python 3.12)
- Line in `CMakeLists.txt` that resets pybind11 submodule was commented out to prevent it reverting
- `3rdparty/caffe/src/caffe/util/io.cpp` patched: `SetTotalBytesLimit` call reduced to 1 argument (protobuf 3.21+ compatibility)
- Body25 model must be a valid download — a corrupted `pose_iter_584000.caffemodel` causes silent no-detection

### Mounting College Drive on Laptop

The college network drive can be mounted via CIFS. Use `uid`/`gid` options so files appear owned by your local user — no sudo needed for reads/writes.

```bash
sudo mount -t cifs //taughtstore.scss.tcd.ie/oneilk10 -o "user=oneilk10,domain=itserv,vers=3.0,uid=$(id -u),gid=$(id -g)" ~/lab-computer/
```

- This does **not** change ownership on the server — files on the lab machine are unaffected
- If the mount is busy on unmount, use `sudo umount -l ~/lab-computer` (lazy unmount)
- The lab machine venv (`~/fyp/linux/`) won't work on the laptop — its Python binary is a symlink to the college machine's Python install. Use a local laptop venv instead (e.g. `fyp-venv`)

## Documentation

All pipeline scripts (`pipeline/` and `pipeline_v2/`) follow standard Python best practices:
- Every script has a module-level docstring
- Every function has a docstring
- All execution code is wrapped in `main()` with an `if __name__ == "__main__": main()` guard — this makes scripts importable as modules without executing

### Generating API docs with pydoc

Run from the project root with the venv active:

```bash
# Terminal output
python3 -m pydoc pipeline.rpm
python3 -m pydoc pipeline_v2.knee_analysis

# Generate HTML files (one per module)
python3 -m pydoc -w pipeline.rpm          # → pipeline.rpm.html
python3 -m pydoc -w pipeline_v2.rpm       # → pipeline_v2.rpm.html

# Convert HTML to PDF/Markdown/Word via pandoc
pandoc pipeline.rpm.html -o pipeline.rpm.pdf
python3 -m pydoc pipeline.rpm | pandoc -f plain -o pipeline.rpm.pdf
```

## TODOs
- [ ] Choose best wheel-detection approach from research scripts and document rationale in report
- [x] Sinusoidal extrapolation for missing bottom-of-stroke — considered, rejected. Unreliable on the short/noisy windows where it would be needed; acknowledged as a known limitation in the report instead.
- [x] Fill in evaluation/ground_truth.csv with Kinovea RPM measurements
- [x] Preprocessing experiments in Pipeline V2 (`pipeline_v2/pose_estimate.py`) — keep Pipeline 1 as baseline for comparison:
  - [x] ROI crop to YOLO cyclist bounding box — `side_angle.py` now saves `cyclist_box`; `pose_estimate.py` crops to it
  - [x] Motion deblur / unsharp mask to counteract lateral motion blur from pass-by footage
  - [x] CLAHE on LAB L-channel for variable outdoor lighting
  - [x] Pad crop to square before resizing to net_resolution (avoid aspect ratio distortion)
  - [x] Tune net_resolution — using `656x368`
- [x] Improve real-world seat height peak selection — `smooth_p80` (80th percentile of SG-smoothed series) replaces `max_fallback` for short pass-by windows (<10 peaks); raises a/b verdict agreement from 32% → 79% on the 19-subject dataset without requiring trainer reference data. `peak_mean` retained for long recordings (≥10 validated peaks). See `evaluation/angle_ceiling_eval.py`.

## Evaluation Findings & Implementation Notes

This section records what was learned from systematic evaluation of the final V2 implementation across the full 10-subject dataset. All figures are from the `output_v2/` outputs with the final `smooth_p80` seat height implementation.

### Hyperparameter tuning decisions

**`PEAK_PROMINENCE` (knee_analysis.py): changed 20.0 → 25.0°**
Prominence sweep showed monotonic RPM MAE improvement from 10° (MAE 9.06) to 30° (MAE 7.49) with seat-height verdict agreement constant at 79% across the entire range and no change in coverage (n=15 throughout). 25° chosen over 30° because mean peaks/video drops to 2.00 at 30° — only one inter-peak period per video on average, giving zero noise averaging and making any single misdetected peak catastrophic. At 25°: MAE 8.52, mean peaks 2.11, same n=15 and 79% SH agreement as production.

**`SQUARE_TOL` (side_angle.py): kept at 0.15**
Tightening from 0.15 to 0.08/0.10/0.12 was evaluated post-hoc from stored per-frame squareness values in selection logs (0.18 cannot be simulated without re-running YOLO). Results: tol=0.08 drops RPM coverage from 15 to 7 videos; tol=0.10 gives 11 videos; tol=0.12 reaches 14 videos with SH agreement improving to 16/19 but RPM MAE worsening from 8.99 to 9.72. No tighter threshold improves RPM MAE by ≥0.5 RPM while maintaining ≥14 coverage — 0.15 is the best operating point. Perspective-distortion mismatches in seat height are not caused by off-axis frames that a stricter gate would reject.

### RPM evaluation results (V2 final)

**Overall:** MAE = 4.2 RPM, RMSE = 8.5 RPM, mean |%err| = 5.8% (n=34)

**By condition:**
- Trainer `a`: MAE = 0.4 RPM, RMSE = 0.7 RPM, mean |%err| = 0.6% (n=19) — near-perfect
- Real-world `b`: MAE = 9.0 RPM, RMSE = 12.7 RPM, mean |%err| = 12.3% (n=15)

**V2 vs P1:** improved on 26/33 shared videos; mean |error| reduction = 7.0 RPM

**Best individual real-world results:** jackb60 (0.4 RPM, 0.6%), liamb60 (0.6 RPM, 0.6%), paddyb30 (0.9 RPM, 1.4%), jennyb30 (1.3 RPM, 1.8%), kateb30 (1.5 RPM, 2.1%)

**No output:** dervlab30, kateb60, romanb30, romanb60 — all have very short or low-quality selection windows. romanb60 has true RPM = 39, extremely low; the pipeline is not validated below ~50 RPM.

**Autocorrelation fallback:** In V2, the 5 videos using autocorrelation have MAE = 12.4 RPM vs 2.8 RPM for peak_detection. The fallback fires on the hard cases (short windows, few peaks) — the higher error reflects difficulty, not method weakness.

### Seat height evaluation results (V2 final with smooth_p80)

**Verdict agreement (b vs a reference): 15/19 (79%)** — up from 6/19 (32%) with the original max_fallback, and from 10/19 (53%) with raw_p90.

**4 remaining mismatches:**
- `dervla30`: trainer = too_high (159.4°), real-world = optimal (147.6°) — smooth_p80 slightly clips dervla's genuinely high extension
- `jack60`: trainer = too_low (136.3°), real-world = too_high (157.5°) — severe pervasive perspective distortion, smooth_p80 cannot correct it
- `jane30`: trainer = optimal (153.1°), real-world = too_high (165.5°) — jane's real-world window has uniformly high angles
- `jenny60`: trainer = optimal (147.0°), real-world = too_low (144.6°) — 0.4° below the threshold, purely borderline

**Trainer consistency (grp30 vs grp60, same subject, same bike):**
- 8/10 subjects give identical verdicts across cadence conditions
- Mean peak delta = 3.91°, median = 2.48°
- Outliers: dervla (11.9°, different verdicts — inflated peaks in grp30 recording), roman (6.3°, same verdict — genuine biomechanical effect: lower cadence → slightly more extension at bottom of stroke), hannah (8.6°, same verdict — outlier peaks at start of grp30 recording)
- Best consistency: jenny (0.3°), jane (0.7°), jack (1.0°)

### Peak angle method — why smooth_p80 works

The key insight from systematic comparison of strategies on the b-condition angle series:

- `smooth_max` is worse than baseline (16% agreement) — smoothing alone doesn't fix the inflated peak; sustained perspective distortion is not single-frame noise
- `raw_p90` reaches 53% — clips the top 10% of raw frames but raw noise limits precision
- `smooth_p80` reaches 79% — the SG smoothing first removes frame-to-frame jitter, then p80 clips the remaining inflated tail from perspective distortion
- `smooth_p80` is not suitable for long trainer recordings (1500+ frames, 50+ full cycles): p80 of the full oscillating distribution lands at mid-stroke (~126°), not at bottom-of-stroke (~138°)
- **Hybrid solution implemented:** `peak_mean` when ≥ 10 validated peaks exist (trainer/long recordings), `smooth_p80` otherwise (real-world pass-by). The 10-peak threshold cleanly separates all trainer videos (48–156 peaks) from all real-world videos (0–8 peaks) in this dataset.

### Filming condition analysis (three groups)

Subjects were filmed in three distinct real-world conditions that have a measurable impact on pipeline performance:

**Group 1 — jenny, kate, roman (close-up, good lighting, single pass)**
- 1 burst per video, 1 run, all going in one direction (left)
- Very short windows: 18–90 frames (0.3–1.5 seconds)
- Consequence: 3/6 videos produce no RPM (roman all three, kateb60) — window too short to observe a complete cycle at 60–90 RPM
- When RPM works, it is accurate: jennyb30 1.8%, kateb30 2.1%
- Seat height is the most reliable of the three groups: clean close-up keypoints, low distortion
- RPM MAE (real-world, n=3): 3.2

**Group 2 — hannah, alex (straight road, further away, two passes — out and back)**
- 2 bursts selected for 3/4 videos, one per pass, both directions observed
- Longer windows: 37–117 frames per burst, up to 226 total selected frames
- The two-pass design is the ideal use case for V2's multi-burst architecture — bursts from each direction are processed independently with the correct knee selected per direction
- 100% seat height verdict agreement (all subjects firmly too_low, so robust to noise)
- RPM is inconsistent: hannahb30 4.1% (good), alexb30 27% (bad). Alex's runs both fall back to autocorrelation despite decent frame counts — the further distance produces noisier keypoints with lower peak detectability
- RPM MAE (real-world, n=4): 11.2

**Group 3 — dervla, jack, jane, liam, paddy (cement sports pitch, many directions)**
- Most bursts found per video (up to 78 for jackb60) — cycling in loops produces many qualifying moments
- Selected burst counts range from 1 to 4; knee_analysis runs from 1 to 4
- Best individual results in the dataset come from this group: jackb60 (0.4 RPM, 0.6%), liamb60 (0.6 RPM, 0.6%), paddyb30 (0.9 RPM, 1.4%)
- Worst results also in this group: jackb30 (32 RPM, 43%) — two runs both use autocorrelation fallback; janeb30 (17.6 RPM, 24%)
- liamb30 produces the most selected frames in the dataset (397, 30.7% selection rate) with 4 bursts and 6 peaks in the main run — the pitch's repeated passes effectively extend the observation window
- Seat height agreement 6/9 (67%) — the three failures (dervla30, jack60, jane30) are all from this group; direction changes and variable camera-subject distances increase perspective distortion
- RPM MAE (real-world, n=8): 10.0

### Key limitations to address in the report

1. **Short observation windows** are the primary constraint on RPM accuracy in Groups 1 and 2. At 60–90 RPM, a sub-1-second window may contain fewer than one complete cycle. No algorithmic improvement can compensate for insufficient signal — widening the selection window is the correct mitigation.
2. **Perspective distortion** affects real-world angle measurements even after side-angle gating. `smooth_p80` mitigates it substantially but cannot fully correct cases where distortion is pervasive across the entire burst (jack60 seat height). True mitigation requires better orthogonal camera positioning or 3D pose estimation.
3. **Very low cadence** (romanb60 true RPM = 39) causes total pipeline failure — no output produced. The system is not validated below ~50 RPM.
4. **Trainer consistency outliers** (dervla 11.9°, roman 6.3°) — dervla's grp30 recording contains a burst of inflated peaks at the start that pull the mean up; roman's 6.3° gap reflects a genuine biomechanical effect (lower cadence → more extension at bottom of stroke, documented in cycling literature).
5. **Clothing — dark and/or baggy trousers worn by many subjects.** Two distinct effects:
   - *Dark clothing*: reduces contrast between the leg and background, lowering OpenPose keypoint confidence at the knee and ankle. Frames with low-confidence detections are partially filtered by `CONF_MIN` but borderline-confidence joints still contribute noisy angle estimates.
   - *Baggy trousers*: the fabric surface occludes the actual joint centre. OpenPose detects the visible surface of the clothing, not the underlying skeletal landmark, introducing a systematic offset in knee position. This biases the Hip→Knee→Ankle angle — typically under-estimating extension because the detected knee position sits proud of the actual joint. This is a likely contributor to the consistent `too_low` verdicts seen across many subjects and may partly explain why real-world peak angles tend to be lower than expected for some subjects even after `smooth_p80` correction. In a controlled biomechanics study, subjects would wear fitted, high-contrast clothing (e.g., tight lycra with reflective markers). For real-world deployment this cannot be assumed — future work should investigate confidence-weighted angle estimation or clothing-aware keypoint correction.

## Final Year Project Report
A detailed written report is required covering approach and findings. It should document:
- **Motivation & problem statement** — why camera angle matters for cycling posture analysis; the challenge of detecting a true side-on view automatically; real-world vs. trainer-mounted distinction
- **Literature / related work** — pose estimation (OpenPose vs. MediaPipe/YOLOPose); object detection (YOLO); classical CV wheel-detection methods (Hough, RANSAC, contour filtering); existing cycling analysis systems
- **System design** — Pipeline 1 (5-stage baseline) and Pipeline V2 (6-stage experimental); the dedicated knee analysis stage as single source of truth for peak detection; computational justification for YOLO gating before OpenPose; why intermediate JSON files enable modular evaluation
- **Wheel-detection experiments** — comparison of the four approaches (`wheels.py`, `circles.py`, `arcs.py`, `contours.py`): methodology, results, failure cases, rationale for chosen approach
- **Side-angle detection** — the bounding-box squareness + size-match heuristic; quantitative evaluation (selection rate, false positive/negative analysis)
- **Pose estimation & angle measurement** — joint keypoint extraction, law-of-cosines angle calculation, seat height thresholds and verdict logic
- **RPM calculation** — knee-angle cycle counting methodology; direction-of-travel detection from front/back wheel positions; contiguous-run selection to avoid inter-window noise; accuracy vs. ground truth
- **Computational evaluation** — frames selected vs. total, OpenPose calls saved, inference time/frame, end-to-end latency
- **Results & evaluation** — qualitative and quantitative assessment on real cycling video; seat height verdict accuracy; RPM accuracy
- **Conclusions & future work** — limitations, what would improve accuracy, potential real-world deployment
  - Known limitation: opportunistic side-angle windows may not capture a full pedal cycle, so bottom-of-stroke is not always directly observed; `max()` then gives a conservative lower bound rather than a true peak. Sinusoidal extrapolation (`angle(t) = C + A·sin(2πft + φ)`, f fixed from autocorrelation) was considered as a mitigation but rejected — the short, noisy windows where peak detection fails are exactly the conditions where amplitude fitting is least reliable. Acknowledged as a limitation; the better mitigation is widening the selection window.
