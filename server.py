from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uuid
import os
import json
import cv2
from openpose import pyopenpose as op

app = FastAPI()

# ---------------------------
# OpenPose configuration
# ---------------------------
params = dict()
params["model_folder"] = "/openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# ---------------------------
# Output directory
# ---------------------------
OUTPUT_DIR = "/app/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/process")
async def process_video(file: UploadFile = File(...)):

    # --- Build output file names ---
    original_name, ext = os.path.splitext(file.filename)
    unique_id = str(uuid.uuid4())

    video_filename = f"{original_name}_{unique_id}.mp4"
    json_filename = f"{original_name}_{unique_id}.json"

    video_output_path = f"{OUTPUT_DIR}/{video_filename}"
    json_output_path = f"{OUTPUT_DIR}/{json_filename}"

    # --- Save uploaded video to /tmp ---
    input_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    output_json = []

    # --- Setup video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    # ---------------------------
    # Processing loop
    # ---------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(video_output_path, fourcc, 30, (w, h))

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Write processed frame to output video
        out.write(datum.cvOutputData)

        # Collect keypoints for this frame
        if datum.poseKeypoints is not None:
            output_json.append(datum.poseKeypoints.tolist())

    cap.release()
    if out:
        out.release()

    # ---------------------------
    # Save JSON to disk
    # ---------------------------
    with open(json_output_path, "w") as jf:
        json.dump(output_json, jf)

    # Print for docker logs (optional)
    print("Saved video:", video_output_path, flush=True)
    print("Saved json:", json_output_path, flush=True)

    # ---------------------------
    # Return URLs only
    # ---------------------------
    return {
        "video_url": f"/video/{video_filename}",
        "json_url": f"/json/{json_filename}"
    }


# ---------------------------
# File serving routes
# ---------------------------
@app.get("/video/{filename}")
async def get_video(filename: str):
    return FileResponse(f"{OUTPUT_DIR}/{filename}", media_type="video/mp4")


@app.get("/json/{filename}")
async def get_json(filename: str):
    return FileResponse(f"{OUTPUT_DIR}/{filename}", media_type="application/json")