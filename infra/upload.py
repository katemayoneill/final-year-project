#!/usr/bin/env python3
"""Upload files or directories to R2.

The last argument is always the R2 folder prefix. All preceding arguments are
local files or directories (directories are walked recursively).

Usage:
  python3 upload.py <src1> [src2 ...] <r2_folder>

Examples:
  python3 upload.py videos/jennya30.mp4 videos/jennya60.mp4 videos
  python3 upload.py videos/. videos
  python3 upload.py pipeline/rpm.py pipeline
"""
import os, sys, boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
    config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
)
BUCKET = os.environ["R2_BUCKET"]


def collect(sources):
    """Return list of (local_path, rel_key) pairs from files/dirs."""
    items = []
    for src in sources:
        src = os.path.normpath(src)
        if os.path.isdir(src):
            for root, _, files in os.walk(src):
                for f in sorted(files):
                    path = os.path.join(root, f)
                    items.append((path, os.path.relpath(path, src)))
        else:
            items.append((src, os.path.basename(src)))
    return items


def upload_file(local_path, key):
    size = os.path.getsize(local_path)
    print(f"  {local_path} → {key} ({size / 1e6:.1f} MB)", end="", flush=True)
    s3.upload_file(
        local_path, BUCKET, key,
        Callback=lambda n: print(f"\r  {local_path} → {key}  {n / size * 100:.0f}%  ", end=""),
    )
    print("  ✓")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 upload.py <src1> [src2 ...] <r2_folder>")
        sys.exit(1)

    *sources, r2_folder = sys.argv[1:]
    r2_folder = r2_folder.rstrip("/")

    items = collect(sources)
    if not items:
        print("No files found.")
        sys.exit(1)

    print(f"Uploading {len(items)} file(s) to '{r2_folder}/'...\n")
    ok = failed = 0
    for local_path, rel in items:
        try:
            upload_file(local_path, f"{r2_folder}/{rel}")
            ok += 1
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            failed += 1

    print(f"\nDone. {ok} uploaded, {failed} failed.")
