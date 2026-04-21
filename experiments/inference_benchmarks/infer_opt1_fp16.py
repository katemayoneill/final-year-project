"""
Optimisation 1: CUDA device + FP16 half-precision inference.

Change from base:
  - model runs on GPU (device=0)
  - half=True cuts tensor size in half — roughly 2x faster on NVIDIA GPUs
    with negligible effect on detection accuracy

Encode is still CPU libx264 (same as base) so inference speedup is isolated.
"""

import cv2
import sys
import os
import subprocess
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip3 install ultralytics")
    sys.exit(1)

CONFIDENCE = 0.5
MODEL_PATH = "best.pt"

if len(sys.argv) < 2:
    print("Usage: python3 infer_opt1_fp16.py <video> [model.pt]")
    sys.exit(1)

input_path = sys.argv[1]
if len(sys.argv) >= 3:
    MODEL_PATH = sys.argv[2]

base, ext   = os.path.splitext(input_path)
output_path = base + "_opt1_fp16.mp4"
temp_path   = base + "_opt1_fp16_tmp.mp4"

print(f"Loading model: {MODEL_PATH}")
# --- OPT 1: move model to GPU ---
model = YOLO(MODEL_PATH)
model.to("cuda")

cap         = cv2.VideoCapture(input_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps         = cap.get(cv2.CAP_PROP_FPS) or 30
frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(temp_path, fourcc, fps, (frame_w, frame_h))

print(f"Processing {frame_count} frames → {output_path}")
t_start   = time.perf_counter()
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- OPT 1: half=True enables FP16 inference ---
    results = model(frame, conf=CONFIDENCE, half=True, verbose=False)

    front_wheel_box = None
    back_wheel_box  = None
    cyclist_found   = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            if label == "cyclist":
                cyclist_found = True
            elif label == "front_wheel":
                front_wheel_box = (x1, y1, x2, y2)
            elif label == "back_wheel":
                back_wheel_box = (x1, y1, x2, y2)

    font_hud = cv2.FONT_HERSHEY_SIMPLEX
    if cyclist_found:
        hud_text, hud_colour = "Cyclist: detected", (0, 255, 0)
    else:
        hud_text, hud_colour = "Cyclist: not detected", (0, 0, 255)
    (hw, hh), _ = cv2.getTextSize(hud_text, font_hud, 0.8, 2)
    cv2.rectangle(frame, (8, 8), (16 + hw, 20 + hh), (0, 0, 0), -1)
    cv2.putText(frame, hud_text, (12, 12 + hh), font_hud, 0.8, hud_colour, 2, cv2.LINE_AA)

    SQUARE_TOL   = 0.15
    SIZE_TOL     = 0.20
    perfect_view = False
    if front_wheel_box and back_wheel_box:
        def _dims(b): return (b[2] - b[0], b[3] - b[1])
        fw, fh = _dims(front_wheel_box)
        bw, bh = _dims(back_wheel_box)
        f_square = abs(1.0 - (fw / fh if fh > 0 else 0)) < SQUARE_TOL
        b_square = abs(1.0 - (bw / bh if bh > 0 else 0)) < SQUARE_TOL
        f_area, b_area = fw * fh, bw * bh
        size_match = (min(f_area, b_area) / max(f_area, b_area)) > (1.0 - SIZE_TOL) if max(f_area, b_area) > 0 else False
        perfect_view = f_square and b_square and size_match

    sv_y_offset = 20 + hh + 12
    sv_text, sv_colour = ("Side view: perfect", (0, 255, 0)) if perfect_view else ("Side view: not perfect", (0, 100, 255))
    (sw, sh), _ = cv2.getTextSize(sv_text, font_hud, 0.8, 2)
    cv2.rectangle(frame, (8, sv_y_offset), (16 + sw, sv_y_offset + sh + 12), (0, 0, 0), -1)
    cv2.putText(frame, sv_text, (12, sv_y_offset + sh + 4), font_hud, 0.8, sv_colour, 2, cv2.LINE_AA)

    if perfect_view:
        msg = "Perfect side angle!"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        (mw, mh), _ = cv2.getTextSize(msg, font, scale, thick)
        mx = (frame_w - mw) // 2
        my = 60
        cv2.rectangle(frame, (mx - 8, my - mh - 8), (mx + mw + 8, my + 8), (0, 0, 0), -1)
        cv2.putText(frame, msg, (mx, my), font, scale, (0, 255, 0), thick, cv2.LINE_AA)

    out.write(frame)
    frame_idx += 1
    pct = frame_idx / frame_count
    bar = ('█' * int(pct * 40)).ljust(40)
    print(f'\r  [{bar}] {frame_idx}/{frame_count} ({pct:.0%})', end='', flush=True)

t_inference = time.perf_counter() - t_start
print(f'\nInference done in {t_inference:.1f}s  ({frame_idx / t_inference:.1f} fps)')

cap.release()
out.release()

t_enc = time.perf_counter()
subprocess.run(
    ['ffmpeg', '-y', '-i', temp_path,
     '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', output_path],
    check=True
)
print(f'Encode done in {time.perf_counter() - t_enc:.1f}s  (CPU libx264)')
os.remove(temp_path)

print(f'Total: {time.perf_counter() - t_start:.1f}s')
print('Done.')
