import cv2
import sys
import os
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip3 install ultralytics")
    sys.exit(1)

# ---------------------------
# Config
# ---------------------------
CONFIDENCE        = 0.5   # YOLO detection confidence
PERSON_CLASS      = 0     # YOLO class id for person
BIKE_CLASS        = 1     # YOLO class id for bicycle
CYCLIST_PROXIMITY = 0.3   # max person-bike centre distance as ratio of frame width
OUTPUT_W          = 640   # output video width
OUTPUT_H          = 480   # output video height

# cyclist always appears this many pixels tall in the output
# tune this — if cyclist gets cut off, reduce it
# if too much background, increase it
TARGET_HEIGHT = 350

# ---------------------------
# Kalman filter — tracks cx, cy, height + their velocities
# ---------------------------
def make_kalman():
    kf = cv2.KalmanFilter(6, 3)  # 6 state, 3 measurements

    # state: [cx, cy, h, vx, vy, vh]
    kf.transitionMatrix = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)

    # measure: [cx, cy, h]
    kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1
    kf.measurementMatrix[1, 1] = 1
    kf.measurementMatrix[2, 2] = 1

    kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1.0
    kf.errorCovPost        = np.eye(6, dtype=np.float32)

    return kf

# ---------------------------
# Helpers
# ---------------------------
def box_centre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def box_height(box):
    return box[3] - box[1]

def merge_boxes(box1, box2):
    return (
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3])
    )

def find_cyclist(results, frame_w):
    person_boxes = []
    bike_boxes   = []

    for result in results:
        for box in result.boxes:
            cls    = int(box.cls[0])
            coords = tuple(map(int, box.xyxy[0]))
            if cls == PERSON_CLASS:
                person_boxes.append(coords)
            elif cls == BIKE_CLASS:
                bike_boxes.append(coords)

    if not person_boxes or not bike_boxes:
        return None

    best_box  = None
    best_dist = float('inf')

    for person in person_boxes:
        for bike in bike_boxes:
            pc        = box_centre(person)
            bc        = box_centre(bike)
            dist      = ((pc[0]-bc[0])**2 + (pc[1]-bc[1])**2) ** 0.5
            norm_dist = dist / frame_w
            if norm_dist < CYCLIST_PROXIMITY and dist < best_dist:
                best_dist = dist
                best_box  = merge_boxes(person, bike)

    return best_box

# ---------------------------
# Input / output paths
# ---------------------------
input_path  = sys.argv[1]
video_name  = os.path.splitext(os.path.basename(input_path))[0]
output_dir  = os.path.dirname(input_path)
output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")

# ---------------------------
# Load YOLO
# ---------------------------
print("Loading YOLO model...")
model = YOLO("yolo26s.pt")

# ---------------------------
# Pass 1: detect + Kalman track centre and height
# ---------------------------
cap         = cv2.VideoCapture(input_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps         = cap.get(cv2.CAP_PROP_FPS) or 30
frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Pass 1: detecting and tracking cyclist in {frame_count} frames...")

kf             = make_kalman()
kf_initialised = False
tracked        = []  # list of (cx, cy, h) per frame
frame_num      = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Frame {frame_num}/{frame_count}", end="\r", flush=True)

    results  = model(frame, classes=[PERSON_CLASS, BIKE_CLASS], conf=CONFIDENCE, verbose=False)
    detected = find_cyclist(results, frame_w)

    if detected is not None:
        cx, cy = box_centre(detected)
        h      = float(box_height(detected))
        measurement = np.array([[cx], [cy], [h]], dtype=np.float32)

        if not kf_initialised:
            kf.statePre  = np.array([[cx], [cy], [h], [0], [0], [0]], dtype=np.float32)
            kf.statePost = kf.statePre.copy()
            kf_initialised = True

        kf.predict()
        state = kf.correct(measurement)
    else:
        if kf_initialised:
            state = kf.predict()
        else:
            tracked.append(None)
            continue

    tracked.append((float(state[0,0]), float(state[1,0]), float(state[2,0])))

cap.release()
print(f"\nTracked {sum(1 for t in tracked if t is not None)}/{frame_count} frames")

# ---------------------------
# Pass 2: crop window sized so cyclist always appears TARGET_HEIGHT tall
# ---------------------------
cap     = cv2.VideoCapture(input_path)
fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
out     = cv2.VideoWriter(output_path, fourcc, fps, (OUTPUT_W, OUTPUT_H))
frame_num = 0

print(f"Pass 2: writing tracked video ({OUTPUT_W}x{OUTPUT_H})...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t         = tracked[frame_num] if frame_num < len(tracked) else None
    frame_num += 1
    print(f"Frame {frame_num}/{frame_count}", end="\r", flush=True)

    if t is None:
        out.write(np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8))
        continue

    cx, cy, h = t

    if h <= 0:
        out.write(np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8))
        continue

    # scale: how much bigger is the cyclist than target?
    # if h=700 and target=350, scale=2 → take 2x output size crop → cyclist fills target height
    # if h=175 and target=350, scale=0.5 → take 0.5x output size crop → zoom in → cyclist still fills target
    scale  = h / TARGET_HEIGHT
    crop_h = int(OUTPUT_H * scale)
    crop_w = int(OUTPUT_W * scale)

    x1 = int(cx) - crop_w // 2
    y1 = int(cy) - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # clamp to frame bounds
    if x1 < 0:       x2 -= x1;             x1 = 0
    if y1 < 0:       y2 -= y1;             y1 = 0
    if x2 > frame_w: x1 -= (x2 - frame_w); x2 = frame_w
    if y2 > frame_h: y1 -= (y2 - frame_h); y2 = frame_h
    x1, y1 = max(0, x1), max(0, y1)

    crop = frame[y1:y2, x1:x2]

    # resize crop to output dimensions maintaining aspect ratio
    if crop.shape[0] > 0 and crop.shape[1] > 0:
        h_crop, w_crop = crop.shape[:2]
        scale    = min(OUTPUT_W / w_crop, OUTPUT_H / h_crop)
        new_w    = int(w_crop * scale)
        new_h    = int(h_crop * scale)
        resized  = cv2.resize(crop, (new_w, new_h))
        canvas   = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
        y_offset = (OUTPUT_H - new_h) // 2
        x_offset = (OUTPUT_W - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        resized  = canvas
    else:
        resized = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)

    out.write(resized)

cap.release()
out.release()

print(f"\nDone! Saved to {output_path}")