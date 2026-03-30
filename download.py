#!/usr/bin/env python3
"""Download a file from R2. Usage: python3 download.py <key> [destination]"""
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


def download(key, dest=None):
    dest = dest or key
    size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    print(f"Downloading {key} ({size / 1e6:.1f} MB) → {dest}")

    s3.download_file(
        BUCKET, key, dest,
        Callback=lambda n: print(f"  {n / size * 100:.1f}%", end="\r"),
    )
    print(f"\nDone. Saved to {dest}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 download.py <key> [destination]")
        sys.exit(1)
    download(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
