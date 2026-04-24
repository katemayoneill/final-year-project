#!/bin/bash
# Run on pod startup: install deps then pull latest pipeline scripts from R2.
set -e

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip3 install numpy opencv-python-headless boto3 python-dotenv ultralytics

echo ""
echo "Downloading pipeline scripts from R2..."
python3 /app/download.py scripts/download_scripts.py /app/download_scripts.py
python3 /app/download_scripts.py /app
