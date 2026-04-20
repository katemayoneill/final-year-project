"""
Bicycle wheel detection using RANSAC-based partial arc fitting.

Unlike Hough circle detection, RANSAC only needs 3 edge points to propose a
full circle, so it works even when pedals/feet occlude most of the wheel rim.

Key improvement over the naive approach: the bounding box is split into a left
and right wheel-search zone, each vertically constrained to where the wheel
centre must physically be (near the bottom of the box).  RANSAC is also given
an explicit centre-position constraint so it cannot accept circles whose centre
sits in the middle of the frame where there are no wheels.
"""

import sys
import os
import math
import random
import subprocess
import cv2 as cv
import numpy as np
from ultralytics import YOLO

# ---------------------------
# YOLO config (mirrors circles.py)
# ---------------------------
CONFIDENCE        = 0.5
PERSON_CLASS      = 0
BIKE_CLASS        = 1
CYCLIST_PROXIMITY = 0.3

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
            dist      = math.hypot(pc[0] - bc[0], pc[1] - bc[1])
            norm_dist = dist / frame_w
            if norm_dist < CYCLIST_PROXIMITY and dist < best_dist:
                best_dist = dist
                best_box  = merge_boxes(person, bike)

    return best_box


# ---------------------------
# RANSAC circle fitting
# ---------------------------

def fit_circle_3pts(p1, p2, p3):
    """Fit the unique circle through 3 points. Returns (cx, cy, r) or None."""
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    cx, cy = float(p3[0]), float(p3[1])

    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-6:
        return None   # collinear

    ux = ((ax**2 + ay**2) * (by - cy) +
          (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) +
          (bx**2 + by**2) * (ax - cx) +
          (cx**2 + cy**2) * (bx - ax)) / D

    r = math.hypot(ax - ux, ay - uy)
    return (ux, uy, r)


def ransac_circle(pts, expected_r,
                  cx_range=None, cy_range=None,
                  r_tol=0.10, n_iters=600, inlier_dist=4):
    """
    RANSAC circle fit constrained by radius prior and optional centre bounds.

    pts        : list of (x, y) edge points (crop-local coords)
    expected_r : prior on wheel radius (pixels)
    cx_range   : (min_cx, max_cx) crop-local — reject fits outside this range
    cy_range   : (min_cy, max_cy) crop-local — reject fits outside this range
    r_tol      : fractional radius tolerance
    n_iters    : RANSAC iterations
    inlier_dist: max pixel distance from circle edge to count as inlier

    Returns (cx, cy, r, inlier_indices) or None.
    """
    if len(pts) < 3:
        return None

    arr  = np.array(pts, dtype=np.float32)
    r_lo = expected_r * (1.0 - r_tol)
    r_hi = expected_r * (1.0 + r_tol)

    best_n      = 0
    best_result = None

    for _ in range(n_iters):
        i, j, k = random.sample(range(len(pts)), 3)
        fit = fit_circle_3pts(pts[i], pts[j], pts[k])
        if fit is None:
            continue
        cx, cy, r = fit

        # --- reject on radius prior ---
        if not (r_lo <= r <= r_hi):
            continue

        # --- reject on centre position ---
        if cx_range is not None and not (cx_range[0] <= cx <= cx_range[1]):
            continue
        if cy_range is not None and not (cy_range[0] <= cy <= cy_range[1]):
            continue

        dists = np.abs(np.hypot(arr[:, 0] - cx, arr[:, 1] - cy) - r)
        mask  = dists < inlier_dist
        n     = int(mask.sum())

        if n > best_n:
            best_n      = n
            best_result = (cx, cy, r, np.where(mask)[0])

    if best_result is None:
        return None

    # Refine radius: median distance of inliers to the fitted centre.
    # The 3-point RANSAC proposal can be pulled outward by noise; the median
    # over all inliers is far more accurate.
    cx, cy, _, inlier_idx = best_result
    inlier_pts = arr[inlier_idx]
    refined_r  = float(np.median(np.hypot(inlier_pts[:, 0] - cx,
                                          inlier_pts[:, 1] - cy)))
    return (cx, cy, refined_r, inlier_idx)


def detect_wheel_in_zone(zone, expected_r, cy_range, min_inliers=12):
    """
    Run RANSAC to find one wheel circle inside a pre-cropped zone.

    zone      : BGR image crop (already the wheel search area)
    expected_r: expected wheel radius in pixels
    cy_range  : (min_cy, max_cy) in zone-local coords — centre must be here
    min_inliers: minimum inlier count to accept a detection

    Returns (cx, cy, r) in zone-local coords, or None.
    """
    gray  = cv.cvtColor(zone, cv.COLOR_BGR2GRAY)
    gray  = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gray, 40, 120)

    ys, xs = np.where(edges > 0)
    pts = list(zip(xs.tolist(), ys.tolist()))

    if len(pts) < 3:
        return None

    if len(pts) > 1500:
        pts = random.sample(pts, 1500)

    result = ransac_circle(pts, expected_r, cy_range=cy_range)
    if result is None:
        return None

    cx, cy, r, inliers = result
    if inliers.size < min_inliers:
        return None

    return (cx, cy, r)


# ---------------------------
# Per-frame processing
# ---------------------------

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

    box_h = y2 - y1
    box_w = x2 - x1

    # Wheel radius ≈ 25 % of full bounding-box height
    expected_r = box_h * 0.25

    # Vertical search zone: tall enough to include the top of the wheel arc.
    # The wheel centre sits ~expected_r above the bottom of the box, so the
    # top of the wheel is ~2*expected_r above the bottom.  Add 20 % margin.
    zone_top = max(y1, int(y2 - expected_r * 2.4))
    zone_h   = y2 - zone_top    # height of the search zone in pixels

    # The wheel centre must be in the lower portion of the zone.
    # In zone-local y coords: from (zone_h - expected_r*1.5) to zone_h.
    cy_lo = max(0.0, zone_h - expected_r * 1.5)
    cy_hi = float(zone_h)
    cy_range = (cy_lo, cy_hi)

    # Split into left and right halves — one wheel per side.
    x_mid = x1 + box_w // 2

    left_zone  = src[zone_top:y2, x1:x_mid]
    right_zone = src[zone_top:y2, x_mid:x2]

    detections = []

    left_hit = detect_wheel_in_zone(left_zone, expected_r, cy_range)
    if left_hit is not None:
        cx, cy, r = left_hit
        detections.append((int(cx) + x1, int(cy) + zone_top, int(r)))

    right_hit = detect_wheel_in_zone(right_zone, expected_r, cy_range)
    if right_hit is not None:
        cx, cy, r = right_hit
        detections.append((int(cx) + x_mid, int(cy) + zone_top, int(r)))

    for (fx, fy, r) in detections:
        cv.circle(src, (fx, fy), 3, (0, 100, 100), -1)
        cv.circle(src, (fx, fy), r, (255, 0, 255), 2)

    cv.rectangle(src, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # DEBUG: draw the wheel search zones
    # cv.rectangle(src, (x1, zone_top), (x_mid, y2), (0, 200, 255), 1)
    # cv.rectangle(src, (x_mid, zone_top), (x2, y2), (0, 200, 255), 1)

    return src


# ---------------------------
# Main
# ---------------------------

def main(argv):
    if len(argv) < 1:
        print('Usage: arcs.py <video.mp4>')
        return -1

    input_path  = argv[0]
    base, ext   = os.path.splitext(input_path)
    output_path = base + '_arcs' + ext
    temp_path   = base + '_arcs_tmp' + ext

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
        ["ffmpeg", "-y", "-i", temp_path,
         "-vcodec", "h264_nvenc", "-cq", "23", "-preset", "p4", output_path],
        check=True
    )
    os.remove(temp_path)

    print('Done.')
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
