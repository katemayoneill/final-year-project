"""
Optimisation 4: All optimisations combined.

Changes from base:
  1. CUDA + FP16 half-precision inference (opt1)
  2. Batched inference, BATCH_SIZE=8 (opt2)
  3. Frame skipping, DETECT_EVERY=2 (opt3)
  4. GPU encode via NVENC h264_nvenc (replaces CPU libx264)

Expected to be the fastest version overall.
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

CONFIDENCE   = 0.5
MODEL_PATH   = "best.pt"
BATCH_SIZE   = 8
DETECT_EVERY = 2

if len(sys.argv) < 2:
    print("Usage: python3 infer_opt4_all.py <video> [model.pt]")
    sys.exit(1)

input_path = sys.argv[1]
if len(sys.argv) >= 3:
    MODEL_PATH = sys.argv[2]

base, ext   = os.path.splitext(input_path)
output_path = base + "_opt4_all.mp4"
temp_path   = base + "_opt4_all_tmp.mp4"

print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.to("cuda")   # OPT 1

cap         = cv2.VideoCapture(input_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps         = cap.get(cv2.CAP_PROP_FPS) or 30
frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(temp_path, fourcc, fps, (frame_w, frame_h))

print(f"Processing {frame_count} frames → {output_path}")
t_start        = time.perf_counter()
frame_idx      = 0
cached_results = None


def annotate_frame(frame, results):
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

    return frame


def flush_batch(batch, detect_flags):
    global frame_idx, cached_results
    # OPT 2+3: only run model on frames that need it; reuse cache for the rest
    detect_frames = [f for f, d in zip(batch, detect_flags) if d]
    if detect_frames:
        # OPT 1: half=True for FP16
        new_results = model(detect_frames, conf=CONFIDENCE, half=True, verbose=False)
        result_iter = iter(new_results)
    else:
        result_iter = iter([])

    for frame, do_detect in zip(batch, detect_flags):
        if do_detect:
            cached_results = [next(result_iter)]
        out.write(annotate_frame(frame, cached_results or []))
        frame_idx += 1
        pct = frame_idx / frame_count
        bar = ('█' * int(pct * 40)).ljust(40)
        print(f'\r  [{bar}] {frame_idx}/{frame_count} ({pct:.0%})', end='', flush=True)


pending       = []
detect_flags  = []
abs_frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    pending.append(frame)
    detect_flags.append(abs_frame_idx % DETECT_EVERY == 0)   # OPT 3
    abs_frame_idx += 1
    if len(pending) == BATCH_SIZE:
        flush_batch(pending, detect_flags)
        pending      = []
        detect_flags = []

if pending:
    flush_batch(pending, detect_flags)

t_inference = time.perf_counter() - t_start
print(f'\nInference done in {t_inference:.1f}s  ({frame_idx / t_inference:.1f} fps)')

cap.release()
out.release()

t_enc = time.perf_counter()
subprocess.run(
    ['ffmpeg', '-y', '-i', temp_path,
     '-vcodec', 'h264_nvenc', '-cq', '23', '-preset', 'p4', output_path],   # OPT 4
    check=True
)
print(f'Encode done in {time.perf_counter() - t_enc:.1f}s  (GPU h264_nvenc)')
os.remove(temp_path)

print(f'Total: {time.perf_counter() - t_start:.1f}s')
print('Done.')
