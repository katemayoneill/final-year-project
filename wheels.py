import sys
import os
import subprocess
import cv2 as cv
import numpy as np
from ultralytics import YOLO

CONFIDENCE = 0.3

# Will be resolved at runtime from the model's class names.
# Override here if auto-detection picks the wrong one.
WHEEL_CLASS_NAME = 'wheel'


def find_wheel_class(model):
    """Return the class ID for 'front_wheel' or any class containing 'wheel'."""
    for cls_id, name in model.names.items():
        if name in ('front_wheel', 'back_wheel'):
            return cls_id
    for cls_id, name in model.names.items():
        if 'wheel' in name.lower():
            return cls_id
    return None


def process_frame(src, model, wheel_class):
    results = model(src, classes=[wheel_class], conf=CONFIDENCE, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r  = max(x2 - x1, y2 - y1) // 2
            cv.circle(src, (cx, cy), 3, (0, 100, 100), -1)
            cv.circle(src, (cx, cy), r, (255, 0, 255), 2)

    return src


def main(argv):
    if len(argv) < 1:
        print('Usage: wheels.py <video.mp4>')
        return -1

    input_path  = argv[0]
    base, ext   = os.path.splitext(input_path)
    output_path = base + '_wheels' + ext
    temp_path   = base + '_wheels_tmp' + ext

    print('Loading YOLO model...')
    model = YOLO('yolo26s.pt')

    print('Available classes:')
    for cls_id, name in model.names.items():
        print(f'  {cls_id}: {name}')

    wheel_class = find_wheel_class(model)
    if wheel_class is None:
        print(f'\nERROR: no class containing "wheel" found — set WHEEL_CLASS_NAME '
              f'to one of the names above and hardcode the ID.')
        return -1

    print(f'\nUsing class {wheel_class}: "{model.names[wheel_class]}"')

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
    print(f'Processing {frame_count} frames → {output_path}')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(process_frame(frame, model, wheel_class))
        frame_idx += 1
        pct = frame_idx / frame_count
        bar = ('█' * int(pct * 40)).ljust(40)
        print(f'\r  [{bar}] {frame_idx}/{frame_count} ({pct:.0%})', end='', flush=True)
    print()

    cap.release()
    out.release()

    subprocess.run(
        ['ffmpeg', '-y', '-i', temp_path,
         '-vcodec', 'h264_nvenc', '-cq', '23', '-preset', 'p4', output_path],
        check=True
    )
    os.remove(temp_path)

    print('Done.')
    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
