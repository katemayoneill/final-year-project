import cv2
import math
import os
import subprocess
import sys
from openpose import pyopenpose as op

# ---------------------------
# Body25 joint indices (OpenPose)
# ---------------------------
JOINTS = {
    "Nose": 0, "Neck": 1,
    "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7,
    "MidHip": 8,
    "RHip": 9, "RKnee": 10, "RAnkle": 11,
    "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18,
    "LBigToe": 19, "LSmallToe": 20, "LHeel": 21,
    "RBigToe": 22, "RSmallToe": 23, "RHeel": 24
}

# ---------------------------
# Law of cosines
# Returns angle at joint B in degrees
# ---------------------------
def calc_angle(A, B, C):
    a_2 = (B[0] - C[0])**2 + (B[1] - C[1])**2
    b_2 = (A[0] - B[0])**2 + (A[1] - B[1])**2
    c_2 = (A[0] - C[0])**2 + (A[1] - C[1])**2
    try:
        theta = math.degrees(math.acos((a_2 + b_2 - c_2) / (2 * math.sqrt(a_2 * b_2))))
        return theta
    except (ZeroDivisionError, ValueError):
        return None

# ---------------------------
# Get joint position from keypoints
# ---------------------------
def get_joint(keypoints, joint_name, person=0):
    idx = JOINTS[joint_name]
    x = keypoints[person][idx][0]
    y = keypoints[person][idx][1]
    conf = keypoints[person][idx][2]
    if conf > 0.1 and x > 0 and y > 0:
        return (int(x), int(y))
    return None

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
output_path = input_path.replace(".mp4", "_annotated_angles.mp4")
temp_path = output_path.replace(".mp4", "_tmp.mp4")

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

    # run openpose
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # datum.cvOutputData already has the skeleton drawn on it
    annotated_frame = datum.cvOutputData

    if out is None:
        h, w = annotated_frame.shape[:2]
        out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    # draw angles on top of the skeleton
    if datum.poseKeypoints is not None:
        keypoints = datum.poseKeypoints

        # --- Right knee angle ---
        RHip   = get_joint(keypoints, "RHip")
        RKnee  = get_joint(keypoints, "RKnee")
        RAnkle = get_joint(keypoints, "RAnkle")

        if RHip and RKnee and RAnkle:
            angle = calc_angle(RHip, RKnee, RAnkle)
            if angle is not None:
                cv2.putText(annotated_frame, f"R Knee: {angle:.1f}",
                            (RKnee[0] + 10, RKnee[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Left knee angle ---
        LHip   = get_joint(keypoints, "LHip")
        LKnee  = get_joint(keypoints, "LKnee")
        LAnkle = get_joint(keypoints, "LAnkle")

        if LHip and LKnee and LAnkle:
            angle = calc_angle(LHip, LKnee, LAnkle)
            if angle is not None:
                cv2.putText(annotated_frame, f"L Knee: {angle:.1f}",
                            (LKnee[0] + 10, LKnee[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Right hip angle (RShoulder -> RHip -> RKnee) ---
        RShoulder = get_joint(keypoints, "RShoulder")

        if RShoulder and RHip and RKnee:
            angle = calc_angle(RShoulder, RHip, RKnee)
            if angle is not None:
                cv2.putText(annotated_frame, f"R Hip: {angle:.1f}",
                            (RHip[0] + 10, RHip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- Left hip angle (LShoulder -> LHip -> LKnee) ---
        LShoulder = get_joint(keypoints, "LShoulder")

        if LShoulder and LHip and LKnee:
            angle = calc_angle(LShoulder, LHip, LKnee)
            if angle is not None:
                cv2.putText(annotated_frame, f"L Hip: {angle:.1f}",
                            (LHip[0] + 10, LHip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    out.write(annotated_frame)

cap.release()
if out:
    out.release()

subprocess.run(
    ["ffmpeg", "-y", "-i", temp_path, "-vcodec", "h264_nvenc", "-cq", "23", "-preset", "p4", output_path],
    check=True
)
os.remove(temp_path)

print(f"\nDone! Saved to {output_path}")

