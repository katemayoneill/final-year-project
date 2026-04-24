#!/usr/bin/env python3
"""Download all pipeline scripts from R2, recreating the project directory structure.

Run on the pod during setup, or locally to pull the latest versions:
  python3 download_scripts.py [dest_dir]

dest_dir defaults to /app on a pod, or the project root locally.
Subdirectories (pipeline/, pipeline_v2/, infra/) are created as needed.
"""
import os
import sys
import boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
)
BUCKET = os.environ["R2_BUCKET"]

DEFAULT_DEST = "/app" if os.path.isdir("/app") else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dest_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEST

# R2 keys mirror the project structure; local paths are relative to dest_dir
SCRIPTS = [
    # Pipeline 1
    "pipeline/side_angle_select.py",
    "pipeline/pose_estimate.py",
    "pipeline/seat_height.py",
    "pipeline/rpm.py",
    "pipeline/annotate_output.py",
    "run_pipeline.py",
    # Pipeline V2
    "pipeline_v2/utils.py",
    "pipeline_v2/side_angle_select.py",
    "pipeline_v2/pose_estimate.py",
    "pipeline_v2/knee_analysis.py",
    "pipeline_v2/seat_height.py",
    "pipeline_v2/rpm.py",
    "pipeline_v2/annotate_output.py",
    "run_pipeline_v2.py",
    # Transfer utilities
    "infra/upload.py",
    "infra/download.py",
]


def download_script(key):
    dest = os.path.join(dest_dir, key)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    try:
        size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    except Exception:
        print(f"  [SKIP] {key} — not found in R2")
        return False
    print(f"  {key} ({size} bytes) → {dest}", end="", flush=True)
    s3.download_file(BUCKET, key, dest)
    os.chmod(dest, 0o755)
    print("  ✓")
    return True


print(f"Downloading {len(SCRIPTS)} scripts from R2 → {dest_dir}\n")
ok = failed = 0
for key in SCRIPTS:
    if download_script(key):
        ok += 1
    else:
        failed += 1

print(f"\nDone. {ok} downloaded, {failed} not found.")
print("Run infra/upload_scripts.py locally first if any are missing.")
