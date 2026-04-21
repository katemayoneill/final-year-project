#!/usr/bin/env python3
"""Upload a file to R2. Usage: python3 upload.py <file>"""
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


def upload(path):
    key = os.path.basename(path)
    size = os.path.getsize(path)
    print(f"Uploading {key} ({size / 1e6:.1f} MB)...")

    s3.upload_file(
        path, BUCKET, key,
        Callback=lambda n: print(f"  {n / size * 100:.1f}%", end="\r"),
    )
    print(f"\nDone. Key: {key}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 upload.py <file>")
        sys.exit(1)
    upload(sys.argv[1])
