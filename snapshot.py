import cv2
import math
import sys
import os
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
# How perpendicular the hips need to be as a ratio
# of total hip-to-hip distance (accounts for scale)
# 0.2 = x difference must be less than 20% of hip distance
# increase if too few frames are found
# ---------------------------
HIP_THRESHOLD = 0.2

# ---------------------------
# Law of cosines - angle at B
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
# Check if hips are perpendicular to camera
# ---------------------------
def is_perpendicular(keypoints):
    RHip = get_joint(keypoints, "RHip")
    LHip = get_joint(keypoints, "LHip")
    if RHip is None or LHip is None:
        return False

    # total distance between hips
    hip_distance = math.sqrt((RHip[0] - LHip[0])**2 + (RHip[1] - LHip[1])**2)

    if hip_distance == 0:
        return False

    # x difference as proportion of total hip distance
    x_diff_ratio = abs(RHip[0] - LHip[0]) / hip_distance

    return x_diff_ratio < HIP_THRESHOLD

# ---------------------------
# Draw angles onto frame, returns frame and r knee angle
# ---------------------------
def draw_angles(frame, keypoints):
    RHip      = get_joint(keypoints, "RHip")
    RKnee     = get_joint(keypoints, "RKnee")
    RAnkle    = get_joint(keypoints, "RAnkle")
    LHip      = get_joint(keypoints, "LHip")
    LKnee     = get_joint(keypoints, "LKnee")
    LAnkle    = get_joint(keypoints, "LAnkle")
    RShoulder = get_joint(keypoints, "RShoulder")
    LShoulder = get_joint(keypoints, "LShoulder")

    r_knee_angle = None

    if RHip and RKnee and RAnkle:
        r_knee_angle = calc_angle(RHip, RKnee, RAnkle)
        if r_knee_angle is not None:
            cv2.putText(frame, f"R Knee: {r_knee_angle:.1f}",
                        (RKnee[0] + 10, RKnee[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if LHip and LKnee and LAnkle:
        l_knee_angle = calc_angle(LHip, LKnee, LAnkle)
        if l_knee_angle is not None:
            cv2.putText(frame, f"L Knee: {l_knee_angle:.1f}",
                        (LKnee[0] + 10, LKnee[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if RShoulder and RHip and RKnee:
        r_hip_angle = calc_angle(RShoulder, RHip, RKnee)
        if r_hip_angle is not None:
            cv2.putText(frame, f"R Hip: {r_hip_angle:.1f}",
                        (RHip[0] + 10, RHip[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if LShoulder and LHip and LKnee:
        l_hip_angle = calc_angle(LShoulder, LHip, LKnee)
        if l_hip_angle is not None:
            cv2.putText(frame, f"L Hip: {l_hip_angle:.1f}",
                        (LHip[0] + 10, LHip[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame, r_knee_angle

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
video_name = os.path.splitext(os.path.basename(input_path))[0]
output_dir = os.path.dirname(input_path)

perp_video_path = os.path.join(output_dir, f"{video_name}_perpendicular.mp4")
snapshot_path   = os.path.join(output_dir, f"{video_name}_6oclock.jpg")

# ---------------------------
# Processing loop
# ---------------------------
cap = cv2.VideoCapture(input_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps         = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
out         = None
frame_num   = 0

best_frame = None
best_angle = 0

print(f"Processing {frame_count} frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Frame {frame_num}/{frame_count}", end="\r", flush=True)

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    if datum.poseKeypoints is None:
        continue

    keypoints = datum.poseKeypoints

    if not is_perpendicular(keypoints):
        continue

    # skeleton already drawn by openpose
    annotated = datum.cvOutputData.copy()

    # draw angles
    annotated, r_knee_angle = draw_angles(annotated, keypoints)

    # frame number overlay
    cv2.putText(annotated, f"Frame: {frame_num}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # initialise video writer on first perpendicular frame
    if out is None:
        h, w = annotated.shape[:2]
        out = cv2.VideoWriter(perp_video_path, fourcc, fps, (w, h))

    out.write(annotated)

    # track frame with maximum knee extension (6 o'clock)
    if r_knee_angle is not None and r_knee_angle > best_angle:
        best_angle = r_knee_angle
        best_frame = annotated.copy()

cap.release()
if out:
    out.release()

print(f"\nDone!")

if best_frame is not None:
    cv2.imwrite(snapshot_path, best_frame)
    print(f"6 o'clock snapshot: {snapshot_path}")
    print(f"Max knee extension: {best_angle:.1f} degrees")
else:
    print("No valid frames found - try increasing HIP_THRESHOLD value")

if out:
    print(f"Perpendicular video: {perp_video_path}")
else:
    print("No perpendicular frames found - try increasing HIP_THRESHOLD value")