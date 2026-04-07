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
import cv2 as cv
import numpy as np

def process_frame(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]

    min_r = rows // 8   # wheels are large — ignore anything tiny
    max_r = rows // 2

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 4,
                               param1=100, param2=50,
                               minRadius=min_r, maxRadius=max_r)

    if circles is not None:
        circles = np.uint16(np.around(circles))[0]  # shape: (N, 3)

        # Find the pair of circles whose radii are most similar in size.
        # With only one circle detected we just draw that one.
        best_pair = None
        best_diff = float('inf')

        if len(circles) == 1:
            best_pair = circles
        else:
            for i in range(len(circles)):
                for j in range(i + 1, len(circles)):
                    diff = abs(int(circles[i][2]) - int(circles[j][2]))
                    if diff < best_diff:
                        best_diff = diff
                        best_pair = circles[[i, j]]

        for c in best_pair:
            cv.circle(src, (c[0], c[1]), 1, (0, 100, 100), 3)
            cv.circle(src, (c[0], c[1]), c[2], (255, 0, 255), 3)

    return src

def main(argv):
    if len(argv) < 1:
        print('Usage: circles.py <video.mp4>')
        return -1

    input_path = argv[0]
    base, ext = os.path.splitext(input_path)
    output_path = base + '_circles' + ext

    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'Error opening video: {input_path}')
        return -1

    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

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
    print('Done.')
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])