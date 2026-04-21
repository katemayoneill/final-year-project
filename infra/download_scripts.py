#!/usr/bin/env python3
"""
Download all pipeline scripts from R2 to this machine.
Run on the pod during setup, or locally to pull the latest versions.

Usage: python3 download_scripts.py [dest_dir]
  dest_dir defaults to /app on the pod, or the script's own directory locally.
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

# Default destination: /app on a RunPod pod, current dir elsewhere
DEFAULT_DEST = "/app" if os.path.isdir("/app") else os.path.dirname(os.path.abspath(__file__))
dest_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEST

SCRIPTS = [
    # Main pipeline
    "side_angle_select.py",
    "pose_estimate.py",
    "seat_height.py",
    "rpm.py",
    "annotate_output.py",
    "run_pipeline.py",
    # Transfer utilities
    "upload.py",
    "download.py",
]

os.makedirs(dest_dir, exist_ok=True)

def download_script(filename):
    key  = f"scripts/{filename}"
    dest = os.path.join(dest_dir, filename)
    try:
        size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    except s3.exceptions.ClientError:
        print(f"  [SKIP] {key} — not found in R2")
        return False
    print(f"  {key} ({size} bytes) → {dest}", end="", flush=True)
    s3.download_file(BUCKET, key, dest)
    os.chmod(dest, 0o755)
    print("  ✓")
    return True

print(f"Downloading {len(SCRIPTS)} scripts from R2 → {dest_dir}\n")
ok = failed = 0
for name in SCRIPTS:
    if download_script(name):
        ok += 1
    else:
        failed += 1

print(f"\nDone. {ok} downloaded, {failed} not found in R2.")
print("Run upload_scripts.py on your local machine first if any are missing.")
