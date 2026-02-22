import cv2
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
output_path = input_path.replace(".mp4", "_annotated.mp4")

# ---------------------------
# Processing loop
# ---------------------------
cap = cv2.VideoCapture(input_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None
frame_num = 0

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

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    out.write(datum.cvOutputData)

cap.release()
if out:
    out.release()

print(f"\nDone! Saved annotated video to {output_path}")