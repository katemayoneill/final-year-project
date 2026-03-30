# OpenPose GPU — Cycling Posture Analysis

A ready-to-run OpenPose environment built on CUDA 11.8 + cuDNN 8. Designed for pose estimation on video, originally built for cycling posture analysis but usable for any Body25 keypoint extraction workload.

## What's inside
- OpenPose compiled from source with Python bindings (`pyopenpose`, Body25 model)
- CUDA 11.8 / cuDNN 8 on Ubuntu 22.04
- GPU architectures: 60, 61, 62, 70, 72, 75, 80, 86, 89, 90
- `upload.py` / `download.py` — S3-compatible file transfer scripts (bring your own bucket)

## Requirements
- NVIDIA GPU with CUDA support
- Docker with `--gpus` flag (nvidia-container-toolkit)

## Quickstart

```bash
docker run --gpus all -it --rm yourimage
```

## File Transfer

The image includes transfer scripts that work with any S3-compatible storage (AWS S3, Cloudflare R2, etc.). Create a `.env` file with your credentials:

```env
R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_key
R2_SECRET_ACCESS_KEY=your_secret
R2_BUCKET=your_bucket
```

Then inside the container:

```bash
pip3 install boto3 python-dotenv

python3 /app/upload.py myfile.mp4
python3 /app/download.py myfile.mp4 /output/myfile.mp4
```

## Notes
- The container runs `sleep infinity` by default — intended for SSH-based access (e.g. RunPod)
- OpenPose models are baked into the image; no separate model download needed
- Python pipeline scripts are not included — transfer your own to `/app/` at runtime
