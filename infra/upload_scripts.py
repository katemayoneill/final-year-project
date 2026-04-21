#!/usr/bin/env python3
"""
Upload all pipeline scripts to R2.
Run this locally whenever scripts are updated.

Usage: python3 infra/upload_scripts.py
"""
import os
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (local path relative to ROOT, R2 key) — pod downloads everything flat to /app
SCRIPTS = [
    ("pipeline/side_angle_select.py", "scripts/side_angle_select.py"),
    ("pipeline/pose_estimate.py",     "scripts/pose_estimate.py"),
    ("pipeline/seat_height.py",       "scripts/seat_height.py"),
    ("pipeline/rpm.py",               "scripts/rpm.py"),
    ("pipeline/annotate_output.py",   "scripts/annotate_output.py"),
    ("run_pipeline.py",               "scripts/run_pipeline.py"),
    ("infra/upload.py",               "scripts/upload.py"),
    ("infra/download.py",             "scripts/download.py"),
]


def upload_script(local_rel, key):
    path = os.path.join(ROOT, local_rel)
    if not os.path.exists(path):
        print(f"  [SKIP] {local_rel} — file not found")
        return False
    size = os.path.getsize(path)
    print(f"  {local_rel} ({size} bytes) → {key}", end="", flush=True)
    s3.upload_file(path, BUCKET, key)
    print("  ✓")
    return True


print(f"Uploading {len(SCRIPTS)} scripts to R2...\n")
ok = failed = 0
for local_rel, key in SCRIPTS:
    if upload_script(local_rel, key):
        ok += 1
    else:
        failed += 1

print(f"\nDone. {ok} uploaded, {failed} skipped.")
