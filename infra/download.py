#!/usr/bin/env python3
"""Download files from R2.

Single file:
  python3 download.py <key> [dest]

Entire bucket (key = "."):
  python3 download.py . [dest_dir]        # dest_dir defaults to current dir

Prefix / folder (key ends with "/"):
  python3 download.py infra/ [dest_dir]   # downloads all keys under infra/

In prefix/bucket mode, files are saved as dest_dir/<key>, preserving structure.
"""
import os, sys, boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
)
BUCKET = os.environ["R2_BUCKET"]


def download_file(key, dest):
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    print(f"  {key} ({size / 1e6:.1f} MB) → {dest}", end="", flush=True)
    s3.download_file(
        BUCKET, key, dest,
        Callback=lambda n: print(f"\r  {key}  {n / size * 100:.0f}%  ", end=""),
    )
    print("  ✓")


def download_prefix(prefix, dest_dir):
    """List all keys under prefix and download them, preserving structure."""
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": BUCKET}
    if prefix:
        kwargs["Prefix"] = prefix

    keys = []
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):  # skip folder marker objects
                keys.append(obj["Key"])

    if not keys:
        print("No objects found.")
        return

    print(f"Found {len(keys)} object(s) under '{prefix or '(all)'}' → {dest_dir}\n")
    ok = failed = 0
    for key in keys:
        dest = os.path.join(dest_dir, key)
        try:
            download_file(key, dest)
            ok += 1
        except Exception as e:
            print(f"\n  [ERROR] {key}: {e}")
            failed += 1

    print(f"\nDone. {ok} downloaded, {failed} failed.")


def download_single(key, dest=None):
    if dest is None:
        dest = key
    elif os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(key))
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    download_file(key, dest)
    print(f"\nDone. Saved to {dest}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    key = sys.argv[1]
    dest = sys.argv[2] if len(sys.argv) > 2 else None

    if key == "." or key.endswith("/"):
        prefix = "" if key == "." else key
        download_prefix(prefix, dest or ".")
    else:
        download_single(key, dest)
