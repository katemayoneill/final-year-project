#!/usr/bin/env python3
"""
Upload all pipeline scripts to R2.
Run this locally whenever scripts are updated.

Usage: python3 upload_scripts.py
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

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    # Main pipeline
    "side_angle_select.py",
    "pose_estimate.py",
    "seat_height.py",
    "rpm.py",
    "annotate_output.py",
    "run_pipeline.py",
    # Transfer utilities (needed on pod)
    "upload.py",
    "download.py",
]

def upload_script(filename):
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] {filename} — file not found")
        return False
    key  = f"scripts/{filename}"
    size = os.path.getsize(path)
    print(f"  {filename} ({size} bytes) → {key}", end="", flush=True)
    s3.upload_file(path, BUCKET, key)
    print("  ✓")
    return True

print(f"Uploading {len(SCRIPTS)} scripts to R2 (scripts/ prefix)...\n")
ok = failed = 0
for name in SCRIPTS:
    if upload_script(name):
        ok += 1
    else:
        failed += 1

print(f"\nDone. {ok} uploaded, {failed} skipped.")
