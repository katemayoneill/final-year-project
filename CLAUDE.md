# Final Year Project — Cycling Posture Analysis

## Project Overview
A cycling posture analysis system that processes video of a cyclist, detects body joint keypoints using OpenPose, and measures joint angles (knee, hip) to evaluate posture. Designed to run on a GPU instance via RunPod.

## Pipeline
```
video.mp4
  → track_cyclist.py    → video_tracked.mp4         (YOLO + Kalman filter tracking)
  → snapshot.py         → video_6oclock.jpg          (6 o'clock pedal position snapshot)
                        → video_perpendicular.mp4
  → process.py          → video_keypoints.json       (25-joint keypoints per frame)
  → annotate.py         → video_annotated.mp4        (skeleton overlay)
  → annotated_angles.py → video_annotated_angles.mp4 (skeleton + angle labels)
```

## Stack
- **OpenPose** (`pyopenpose`) — Body25 model, 25-joint pose estimation
- **YOLO** (`ultralytics`, yolov8n.pt) — Person + bicycle detection
- **OpenCV** — Video I/O, annotation
- **NumPy** — Kalman filter, angle maths (law of cosines)
- **Docker** — Base: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **RunPod** — GPU cloud deployment target

## Docker Image
- CUDA 11.8 + cuDNN 8 on Ubuntu 22.04
- OpenPose compiled from source with Python bindings (`BUILD_PYTHON=ON`), cuDNN + CUDA enabled
- GPU architectures: 60, 61, 62, 70, 72, 75, 80, 86, 89, 90
- On startup: `sleep infinity` (keeps pod alive; RunPod handles SSH at the platform level)
- OpenPose models in `/openpose/models/` — baked into the image at build time via `COPY models/`; NOT in git (too large)
- Scripts are transferred to `/app` on the pod at runtime as needed (not bundled in the image)
- Python deps: `numpy`, `opencv-python-headless`, `boto3`, `python-dotenv`, `ultralytics`

## Key Constraints
- **Model weights** are excluded from git (`.gitignore` ignores `/models`)
- **runpodctl** is blocked on the college network — cannot use it for file transfer
- **HTTP transfers** time out on large files — naive HTTP upload/download is not viable
- Target workflow: upload input video to pod → run pipeline → retrieve output files

## File Transfer
Large files are transferred via **Cloudflare R2** (S3-compatible, HTTPS/port 443):
- `upload.py <file>` — uploads a file to R2 from laptop or pod
- `download.py <key> [dest]` — downloads a file from R2
- Credentials via `.env`: `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
- `boto3.upload_file` handles multipart automatically — suitable for large videos

## Scripts — Usage
All scripts take a single positional argument: the input video path.
```bash
python3 /app/track_cyclist.py   <video.mp4>
python3 /app/snapshot.py        <video_tracked.mp4>
python3 /app/process.py         <video_tracked.mp4>
python3 /app/annotate.py        <video_tracked.mp4>
python3 /app/annotated_angles.py <video_tracked.mp4>
```

## TODOs
- [ ] Finalise pipeline scripts (current versions are for testing)
