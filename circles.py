"""
This code is adapted and modified from the OpenCV documentation:
'Hough Circle Transform' tutorial.

Source:
https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html

Original author: Ana Huamán
Accessed: March 2026
"""

import sys
import os
import subprocess
import cv2 as cv
import numpy as np
from ultralytics import YOLO

# ---------------------------
# YOLO config (mirrors track_cyclist.py)
# ---------------------------
CONFIDENCE        = 0.5
PERSON_CLASS      = 0
BIKE_CLASS        = 1
CYCLIST_PROXIMITY = 0.3   # max person-bike centre distance as ratio of frame width

# Downscale factor applied to the crop before Hough detection.
# Smaller crops reduce noise and compression artefacts.
# Results are scaled back to full-frame coords before drawing.
DOWNSCALE = 0.5

# Spatial zone thresholds (crop-local x as fraction of crop width).
# Wheels must be in the outer portions — never the middle.
LEFT_ZONE_MAX  = 0.42
RIGHT_ZONE_MIN = 0.58


def box_centre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

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


prev_radii = {'left': None, 'right': None}

def process_frame(src, model, frame_w):
    results = model(src, classes=[PERSON_CLASS, BIKE_CLASS],
                    conf=CONFIDENCE, verbose=False)
    bbox    = find_cyclist(results, frame_w)

    if bbox is None:
        return src

    x1, y1, x2, y2 = bbox
    h_frame, w_frame = src.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_frame, x2); y2 = min(h_frame, y2)

    box_h     = y2 - y1
    box_w     = x2 - x1
    y1_bottom = y1 + box_h // 2

    crop = src[y1_bottom:y2, x1:x2]
    if crop.size == 0:
        return src

    # Downscale crop before Hough — reduces noise and compression artefacts
    small = cv.resize(crop, None, fx=DOWNSCALE, fy=DOWNSCALE,
                      interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]

    min_r = max(5, rows * 15 // 100)
    max_r = rows * 30 // 100

    raw = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows // 2,
                          param1=100, param2=40,
                          minRadius=min_r, maxRadius=max_r)

    if raw is not None:
        raw = np.uint16(np.around(raw))[0]   # (N, 3)

        left_candidates  = []
        right_candidates = []

        for c in raw:
            # scale detections back to full crop resolution
            cx_crop = int(c[0] / DOWNSCALE)
            cy_crop = int(c[1] / DOWNSCALE)
            r       = int(c[2] / DOWNSCALE)
            frac    = cx_crop / box_w

            fx = cx_crop + x1
            fy = cy_crop + y1_bottom

            if frac < LEFT_ZONE_MAX:
                left_candidates.append((fx, fy, r))
            elif frac > RIGHT_ZONE_MIN:
                right_candidates.append((fx, fy, r))
            # middle: discard

        # Pick the largest circle from each side, but cap at the previous radius
        # so detections can only shrink, never suddenly grow larger.
        chosen = []
        for side, candidates in (('left', left_candidates), ('right', right_candidates)):
            if not candidates:
                continue
            cx, cy, r = max(candidates, key=lambda c: c[2])
            if prev_radii[side] is not None:
                r = max(r, prev_radii[side])
            prev_radii[side] = r
            chosen.append((cx, cy, r))

        for (cx, cy, r) in chosen:
            cv.circle(src, (cx, cy), 3, (0, 100, 100), -1)
            cv.circle(src, (cx, cy), r, (255, 0, 255), 2)

    cv.rectangle(src, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return src


def main(argv):
    if len(argv) < 1:
        print('Usage: circles.py <video.mp4>')
        return -1

    input_path  = argv[0]
    base, ext   = os.path.splitext(input_path)
    output_path = base + '_circles' + ext
    temp_path   = base + '_circles_tmp' + ext

    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'Error opening video: {input_path}')
        return -1

    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out    = cv.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print('Loading YOLO model...')
    model = YOLO("yolo26s.pt")
    print(f'Processing {frame_count} frames → {output_path}')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(process_frame(frame, model, width))
        frame_idx += 1
        pct = frame_idx / frame_count
        bar = ('█' * int(pct * 40)).ljust(40)
        print(f'\r  [{bar}] {frame_idx}/{frame_count} ({pct:.0%})', end='', flush=True)
    print()

    cap.release()
    out.release()

    subprocess.run(
        ["ffmpeg", "-y", "-i", temp_path, "-vcodec", "h264_nvenc", "-cq", "23", "-preset", "p4", output_path],
        check=True
    )
    os.remove(temp_path)

    print('Done.')
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
