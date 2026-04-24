#!/usr/bin/env python3
"""Upload all pipeline scripts to R2, mirroring the project directory structure.

Run locally whenever scripts are updated:
  python3 infra/upload_scripts.py

R2 keys match the local paths relative to the project root, e.g.:
  pipeline/rpm.py, pipeline_v2/knee_analysis.py, run_pipeline_v2.py
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

# (local path relative to ROOT, R2 key) — R2 key mirrors project structure
SCRIPTS = [
    # Pipeline 1
    ("pipeline/side_angle_select.py",    "pipeline/side_angle_select.py"),
    ("pipeline/pose_estimate.py",        "pipeline/pose_estimate.py"),
    ("pipeline/seat_height.py",          "pipeline/seat_height.py"),
    ("pipeline/rpm.py",                  "pipeline/rpm.py"),
    ("pipeline/annotate_output.py",      "pipeline/annotate_output.py"),
    ("run_pipeline.py",                  "run_pipeline.py"),
    # Pipeline V2
    ("pipeline_v2/utils.py",             "pipeline_v2/utils.py"),
    ("pipeline_v2/side_angle_select.py", "pipeline_v2/side_angle_select.py"),
    ("pipeline_v2/pose_estimate.py",     "pipeline_v2/pose_estimate.py"),
    ("pipeline_v2/knee_analysis.py",     "pipeline_v2/knee_analysis.py"),
    ("pipeline_v2/seat_height.py",       "pipeline_v2/seat_height.py"),
    ("pipeline_v2/rpm.py",               "pipeline_v2/rpm.py"),
    ("pipeline_v2/annotate_output.py",   "pipeline_v2/annotate_output.py"),
    ("run_pipeline_v2.py",               "run_pipeline_v2.py"),
    # Transfer utilities (available on pod for video upload/download)
    ("infra/upload.py",                  "infra/upload.py"),
    ("infra/download.py",                "infra/download.py"),
]


def upload_script(local_rel, key):
    path = os.path.join(ROOT, local_rel)
    if not os.path.exists(path):
        print(f"  [SKIP] {local_rel} — not found")
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
