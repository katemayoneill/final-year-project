import cv2
import json
import sys
from openpose import pyopenpose as op

# ---------------------------
# OpenPose configuration
# ---------------------------
params = dict()
params["model_folder"] = "/openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# ---------------------------
# Input / output paths
# ---------------------------
input_path = sys.argv[1]
output_json_path = input_path.replace(".mp4", "_keypoints.json")

# ---------------------------
# Processing loop
# ---------------------------
cap = cv2.VideoCapture(input_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0
output_json = []

print(f"Processing {input_path} ({frame_count} frames)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Frame {frame_num}/{frame_count}", end="\r", flush=True)

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    if datum.poseKeypoints is not None:
        output_json.append(datum.poseKeypoints.tolist())
    else:
        output_json.append(None)

cap.release()

# ---------------------------
# Save keypoints to JSON
# ---------------------------
with open(output_json_path, "w") as f:
    json.dump(output_json, f)

print(f"\nDone! Saved keypoints to {output_json_path}")