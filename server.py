import cv2
import json
import sys
from openpose import pyopenpose as op

params = dict()
params["model_folder"] = "/openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

input_path = sys.argv[1]  # pass video path as argument
output_json_path = input_path.replace(".mp4", "_keypoints.json")

cap = cv2.VideoCapture(input_path)
output_json = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    if datum.poseKeypoints is not None:
        output_json.append(datum.poseKeypoints.tolist())

cap.release()

with open(output_json_path, "w") as f:
    json.dump(output_json, f)

print(f"Done! Saved to {output_json_path}")