from fastapi import FastApi, UploadFile
import uuid
import os
import cv2
import json
from openpose import pyopenpose as op

app = FastApi()

# openpose config
params() = dict()
params["model_folder"] = "/openpose/models"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

@app.post("/process")
async def process_video(file: UploadFile):
    
    input_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)

    output_json = []
    output_video = f"/tmp/{uuid.uuid4()}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_video, fourcc, 30, (w, h))

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        out.write(datum.cvOutputData)

        if datum.poseKeyPoints is not None:
            output_json.append(datum.poseKeypoints.tolist())

    cap.release()
    if out:
        out.release()

    return {
        "json": output_json,
        "processed_video": output_video
    }
