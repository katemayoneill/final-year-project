#!/usr/bin/env python3
"""
stage 5: annotate original video with skeleton, angle labels, seat height verdict, RPM.

usage: python3 annotate_output.py <video.mp4> <keypoints.json> <assessment.json> <rpm.json>
output: output_v2/<stem>/<stem>_final.mp4
"""
import cv2
import json
import math
import os
import subprocess
import sys

from utils import CONF_MIN, calc_angle, print_progress, video_stem

KNEE_OPTIMAL_LOW  = 145.0
KNEE_OPTIMAL_HIGH = 155.0

GRAPH_W = 320
GRAPH_H = 100
GRAPH_PAD = 12

ANGLE_AXIS_PAD = 10.0
GRAPH_OVERLAY_ALPHA = 0.65
OPTIMAL_ZONE_ALPHA = 0.18


BODY25_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21),
    (11, 22), (22, 23), (11, 24),
]

JOINTS = {
    "RShoulder": 2, "LShoulder": 5,
    "RHip": 9,  "RKnee": 10, "RAnkle": 11,
    "LHip": 12, "LKnee": 13, "LAnkle": 14,
}

VERDICT_COLOUR = {
    "optimal": (0, 220, 0),
    "too_high": (0, 100, 255),
    "too_low": (0, 100, 255),
    "insufficient_data": (180, 180, 180),
}


def get_xy(keypoints, idx):
    """retun int (x, y) for keypoint idx if confidence >= CONF_MIN and position is valid, else None."""
    if not keypoints or idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    return (int(x), int(y)) if c >= CONF_MIN and x > 0 and y > 0 else None


def draw_skeleton(frame, keypoints):
    """draw limb connections and joint dots onto frame."""
    joints = [get_xy(keypoints, i) for i in range(len(keypoints))]
    for a, b in BODY25_PAIRS:
        if a < len(joints) and b < len(joints) and joints[a] and joints[b]:
            cv2.line(frame, joints[a], joints[b], (0, 255, 128), 2, cv2.LINE_AA)
    for pt in joints:
        if pt:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, (0, 180, 90), 1, cv2.LINE_AA)


def draw_angle_annotation(frame, keypoints, name_a, name_b, name_c, colour, prefix, arc_radius=45):
    """draw arc + rays + label at joint B showing the angle between joints A,B,C."""
    A = get_xy(keypoints, JOINTS[name_a]) if name_a in JOINTS else None
    B = get_xy(keypoints, JOINTS[name_b]) if name_b in JOINTS else None
    C = get_xy(keypoints, JOINTS[name_c]) if name_c in JOINTS else None
    if not (A and B and C):
        return
    angle = calc_angle(A, B, C)
    if angle is None:
        return

    ang_a = math.degrees(math.atan2(A[1] - B[1], A[0] - B[0]))
    ang_c = math.degrees(math.atan2(C[1] - B[1], C[0] - B[0]))

    diff_cw = (ang_c - ang_a) % 360
    if diff_cw <= 180:
        start, span = ang_a, diff_cw
    else:
        start, span = ang_c, 360 - diff_cw

    cv2.ellipse(frame, B, (arc_radius, arc_radius), 0,
                start, start + span, colour, 2, cv2.LINE_AA)

    pt_a = (int(B[0] + arc_radius * math.cos(math.radians(ang_a))),
            int(B[1] + arc_radius * math.sin(math.radians(ang_a))))
    pt_c = (int(B[0] + arc_radius * math.cos(math.radians(ang_c))),
            int(B[1] + arc_radius * math.sin(math.radians(ang_c))))
    cv2.line(frame, B, pt_a, colour, 1, cv2.LINE_AA)
    cv2.line(frame, B, pt_c, colour, 1, cv2.LINE_AA)

    mid_rad = math.radians(start + span / 2)
    lx = int(B[0] + (arc_radius + 20) * math.cos(mid_rad))
    ly = int(B[1] + (arc_radius + 20) * math.sin(mid_rad))
    text = f"{prefix}: {angle:.1f}deg"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (0, 0, 0), -1)
    cv2.putText(frame, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def build_graph_params(angle_series, peak_timestamps, cadence_rpm, knee_used, frame_w, frame_h):
    """compute graph layout values. return None if series too short."""
    if len(angle_series) < 2:
        return None
    GW, GH = GRAPH_W, GRAPH_H
    gx = frame_w - GW - GRAPH_PAD
    gy = frame_h - GH - GRAPH_PAD - 20
    times = [t for t, _ in angle_series]
    angles = [a for _, a in angle_series]
    t_min, t_max = times[0], times[-1]
    a_min = max(0.0,   min(angles) - ANGLE_AXIS_PAD)
    a_max = min(180.0, max(angles) + ANGLE_AXIS_PAD)
    t_rng = max(t_max - t_min, 1e-6)
    a_rng = max(a_max - a_min, 1.0)

    def to_px(t, a):
        x = gx + int((t - t_min) / t_rng * GW)
        y = gy + GH - int((a - a_min) / a_rng * GH)
        return (max(gx, min(gx + GW, x)), max(gy, min(gy + GH, y)))

    y_hi = max(gy, min(gy + GH, gy + GH - int((KNEE_OPTIMAL_HIGH - a_min) / a_rng * GH)))
    y_lo = max(gy, min(gy + GH, gy + GH - int((KNEE_OPTIMAL_LOW  - a_min) / a_rng * GH)))
    pts = [to_px(t, a) for t, a in angle_series]
    peak_xs = [gx + int((pkt - t_min) / t_rng * GW) for pkt in peak_timestamps if t_min <= pkt <= t_max]
    rpm_str = f"  |  {cadence_rpm} RPM" if cadence_rpm is not None else ""
    return {
        "GW": GW, "GH": GH, "gx": gx, "gy": gy,
        "t_min": t_min, "t_max": t_max, "t_rng": t_rng,
        "y_hi": y_hi, "y_lo": y_lo,
        "pts": pts, "peak_xs": peak_xs,
        "to_px": to_px,
        "series": list(zip(times, angles)),
        "label": f"{knee_used.capitalize()} knee angle{rpm_str}",
    }


def draw_cadence_graph(frame, current_ts, gp):
    """draw knee angle over time, peak markers, progress mark."""
    GW, GH, gx, gy = gp["GW"], gp["GH"], gp["gx"], gp["gy"]
    t_min, t_max, t_rng = gp["t_min"], gp["t_max"], gp["t_rng"]

    overlay = frame.copy()
    cv2.rectangle(overlay, (gx - 6, gy - 24), (gx + GW + 6, gy + GH + 6), (0, 0, 0), -1)
    cv2.addWeighted(overlay, GRAPH_OVERLAY_ALPHA, frame, 1.0 - GRAPH_OVERLAY_ALPHA, 0, frame)

    y_hi, y_lo = gp["y_hi"], gp["y_lo"]
    if y_hi < y_lo:
        shd = frame.copy()
        cv2.rectangle(shd, (gx, y_hi), (gx + GW, y_lo), (0, 200, 80), -1)
        cv2.addWeighted(shd, OPTIMAL_ZONE_ALPHA, frame, 1.0 - OPTIMAL_ZONE_ALPHA, 0, frame)

    pts = gp["pts"]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (70, 205, 70), 2, cv2.LINE_AA)

    for mx in gp["peak_xs"]:
        cv2.line(frame,  (mx, gy), (mx, gy + GH), (30, 120, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, (mx, gy + 5), 4, (30, 120, 255), -1, cv2.LINE_AA)

    if t_min <= current_ts <= t_max:
        cx      = gx + int((current_ts - t_min) / t_rng * GW)
        cv2.line(frame, (cx, gy), (cx, gy + GH), (255, 255, 255), 1, cv2.LINE_AA)
        closest = min(gp["series"], key=lambda ta: abs(ta[0] - current_ts))
        dot     = gp["to_px"](closest[0], closest[1])
        cv2.circle(frame, dot, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, dot, 5, (70, 205, 70),  1,  cv2.LINE_AA)

    cv2.putText(frame, gp["label"], (gx, gy - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)


def draw_hud(frame, verdict, verdict_detail, cadence_rpm, frame_w, frame_h):
    """draw verdict, detail, and RPM label onto frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    colour = VERDICT_COLOUR.get(verdict, (180, 180, 180))
    label = verdict.replace("_", " ").upper()

    (tw, th), _ = cv2.getTextSize(label, font, 1.1, 2)
    mx = (frame_w - tw) // 2
    cv2.rectangle(frame, (mx - 10, 12), (mx + tw + 10, 12 + th + 14), (0, 0, 0), -1)
    cv2.putText(frame, label, (mx, 12 + th + 4), font, 1.1, colour, 2, cv2.LINE_AA)

    (dw, dh), _ = cv2.getTextSize(verdict_detail, font, 0.55, 1)
    dx = (frame_w - dw) // 2
    dy = 12 + th + 24
    cv2.rectangle(frame, (dx - 6, dy - 4), (dx + dw + 6, dy + dh + 6), (0, 0, 0), -1)
    cv2.putText(frame, verdict_detail, (dx, dy + dh), font, 0.55, colour, 1, cv2.LINE_AA)

    if cadence_rpm is not None:
        rpm_text = f"Cadence: {cadence_rpm} RPM"
        (rw, rh), _ = cv2.getTextSize(rpm_text, font, 0.8, 2)
        cv2.rectangle(frame, (8, frame_h - rh - 20), (16 + rw, frame_h - 8), (0, 0, 0), -1)
        cv2.putText(frame, rpm_text, (12, frame_h - 12), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    """annotate video with skeleton and verdict, write final.mp4."""
    if len(sys.argv) < 5:
        print("usage: python3 annotate_output.py <video.mp4> <keypoints.json> <assessment.json> <rpm.json>")
        sys.exit(1)

    video_path = sys.argv[1]
    kp_path = sys.argv[2]
    assessment_path = sys.argv[3]
    rpm_path = sys.argv[4]

    with open(kp_path) as f: kp_data  = json.load(f)
    with open(assessment_path) as f: assess   = json.load(f)
    with open(rpm_path) as f: rpm_data = json.load(f)

    stem = video_stem(video_path)
    out_dir = os.path.join("output_v2", stem)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, stem)
    temp_path = base + "_final_tmp.mp4"
    output_path = base + "_final.mp4"

    kp_by_frame = {f["frame_idx"]: f["keypoints"] for f in kp_data["frames"]}
    verdict = assess["summary"]["verdict"]
    verdict_detail = assess["summary"]["verdict_detail"]
    cadence_rpm = rpm_data.get("cadence_rpm")
    angle_series = rpm_data.get("angle_series", [])
    peak_timestamps = rpm_data.get("peak_timestamps", [])
    knee_used = rpm_data.get("knee_used", "right")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (frame_w, frame_h))
    gp = build_graph_params(angle_series, peak_timestamps, cadence_rpm, knee_used, frame_w, frame_h)

    frame_idx = 0
    print(f"[Pipeline V2] Annotating {frame_count} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_ts = frame_idx / fps

        if frame_idx in kp_by_frame:
            kp = kp_by_frame[frame_idx]
            draw_skeleton(frame, kp)
            draw_angle_annotation(frame, kp, "RHip", "RKnee", "RAnkle", (0, 255, 255), "R Knee")
            draw_angle_annotation(frame, kp, "LHip", "LKnee", "LAnkle", (0, 255, 255), "L Knee")
            draw_angle_annotation(frame, kp, "RShoulder", "RHip", "RKnee",  (255, 255, 0), "R Hip")
            draw_angle_annotation(frame, kp, "LShoulder", "LHip", "LKnee",  (255, 255, 0), "L Hip")
            draw_hud(frame, verdict, verdict_detail, cadence_rpm, frame_w, frame_h)

        if gp:
            draw_cadence_graph(frame, current_ts, gp)

        out.write(frame)
        frame_idx += 1
        print_progress(frame_idx, frame_count)

    print()
    cap.release()
    out.release()

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_path,
             "-vcodec", "h264_nvenc", "-cq", "23", "-preset", "p4", output_path],
            check=True
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_path,
             "-c:v", "libx264", "-crf", "23", output_path],
            check=True
        )
    finally:
        os.remove(temp_path)
    print(f"Done → {output_path}")

if __name__ == "__main__":
    main()
