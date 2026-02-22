# Cycling Posture Analysis System

## Scripts

### `track_cyclist.py`
Detects the cyclist using YOLO and tracks them across frames using a Kalman filter. Outputs a cropped video with the cyclist centred and consistently scaled.

```bash
pip3 install ultralytics
python3 /app/track_cyclist.py video.mp4
# output: video_tracked.mp4
```

---

### `process.py`
Runs OpenPose on a video and saves all detected joint keypoints to a JSON file for downstream analysis. Each entry in the JSON corresponds to a frame and contains the (x, y, confidence) coordinates for all 25 body joints.

```bash
python3 /app/process.py video.mp4
# output: video_keypoints.json
```

---

### `annotate_angles.py`
Runs OpenPose on a video and outputs an annotated video with the skeleton and joint angles (knee and hip) drawn on every frame.

```bash
python3 /app/annotate_angles.py video.mp4
# output: video_annotated_angles.mp4
```

---

### `snapshot.py`
Finds frames where the cyclist is perpendicular to the camera and saves the frame with maximum knee extension (6 o'clock pedal position) as a snapshot with angles drawn on. Also outputs a video of all perpendicular frames.

```bash
python3 /app/snapshot.py video.mp4
# output: video_6oclock.jpg
#         video_perpendicular.mp4
```

---

## Recommended Workflow

```bash
# step 1: isolate and track the cyclist
python3 /app/track_cyclist.py video.mp4

# step 2: get the 6 o'clock measurement snapshot
python3 /app/snapshot.py video_tracked.mp4
```

---

## File Transfer (RunPod)

```bash
# laptop → pod
runpodctl send video.mp4        # run on laptop, gives a code
runpodctl receive <code>         # run on pod

# pod → laptop
runpodctl send result.jpg        # run on pod, gives a code
runpodctl receive <code>         # run on laptop
```