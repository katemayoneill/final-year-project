"""
Contour-based wheel detection using circularity filtering.

Approach: Canny edges → findContours → filter by circularity and area
→ pick pair with most similar radius → draw results.

Circularity metric from Dawson-Howe, 'A Practical Introduction to
Computer Vision with OpenCV', Section 8.3.2.7:
    circularity = 4π · area / perimeter²
A perfect circle = 1.0.
"""

import sys
import os
import subprocess
import cv2 as cv
import numpy as np

FILL_THRESH = 0.4  # min ratio of contourArea to enclosing circle area


def process_frame(src):
    h, w = src.shape[:2]

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 2)

    edges = cv.Canny(blurred, 50, 150)

    # Morphological close on the edge image joins spoke gaps
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # RETR_LIST to include inner contours (e.g. inner rim)
    contours, _ = cv.findContours(closed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    min_r = h * 0.08  # wheel radius must be at least 8% of frame height
    max_r = h * 0.50

    candidates = []
    for cnt in contours:
        (cx, cy), radius = cv.minEnclosingCircle(cnt)
        if radius < min_r or radius > max_r:
            continue
        area = cv.contourArea(cnt)
        circle_area = np.pi * radius ** 2
        if circle_area == 0:
            continue
        # Fill ratio: how much of the enclosing circle does the contour polygon cover?
        # A circular contour scores ~1.0; irregular blobs score much lower.
        if area / circle_area < FILL_THRESH:
            continue
        candidates.append((int(cx), int(cy), int(radius)))

    if not candidates:
        return src

    if len(candidates) == 1:
        best_pair = candidates
    else:
        best_pair = None
        best_diff = float('inf')
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                diff = abs(candidates[i][2] - candidates[j][2])
                if diff < best_diff:
                    best_diff = diff
                    best_pair = [candidates[i], candidates[j]]

    for (cx, cy, r) in best_pair:
        cv.circle(src, (cx, cy), 1, (0, 100, 100), 3)
        cv.circle(src, (cx, cy), r, (255, 0, 255), 3)

    return src


def main(argv):
    if len(argv) < 1:
        print('Usage: contours.py <video.mp4>')
        return -1

    input_path = argv[0]
    base, ext = os.path.splitext(input_path)
    output_path = base + '_contours' + ext
    temp_path = base + '_contours_tmp' + ext

    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'Error opening video: {input_path}')
        return -1

    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'Processing {frame_count} frames → {output_path}')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(process_frame(frame))
        frame_idx += 1
        pct = frame_idx / frame_count
        bar = ('█' * int(pct * 40)).ljust(40)
        print(f'\r  [{bar}] {frame_idx}/{frame_count} ({pct:.0%})', end='', flush=True)
    print()

    cap.release()
    out.release()

    subprocess.run(
        ['ffmpeg', '-y', '-i', temp_path, '-vcodec', 'h264_nvenc', '-cq', '23', '-preset', 'p4', output_path],
        check=True
    )
    os.remove(temp_path)

    print('Done.')
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
