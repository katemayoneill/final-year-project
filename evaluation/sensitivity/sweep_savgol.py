#!/usr/bin/env python3
"""
Tier 2 sensitivity sweep: Savitzky-Golay smoothing window length.

knee_analysis.py uses SAVGOL_WINDOW = 11 (with polyorder = 2). This sweep
varies the window over [7, 9, 11, 13, 15] and reports:
  - Mean peaks found per b-video
  - RPM MAE against ground truth
  - Seat height verdict agreement (b vs a reference)
  - Mean inter-peak period std dev (spread across runs)

NOTE: The angle_series in _knee_analysis.json stores the SMOOTHED series, not
the raw per-frame values. Raw angles are reconstructed here from _keypoints.json
using the same joint selection logic as knee_analysis.py (direction from
_selection_log.json, camera-facing knee, Body25 indices). This is cheap — no
OpenPose re-run required.

Output:
  evaluation/sensitivity/results/savgol_sweep.csv
  Prints a summary table to stdout.

Usage (from project root):
  python3 evaluation/sensitivity/sweep_savgol.py [--v2 output_v2/] [--gt evaluation/ground_truth.csv]
"""

import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np
from scipy.signal import savgol_filter, find_peaks as scipy_find_peaks, correlate

OPTIMAL_LOW         = 145.0
OPTIMAL_HIGH        = 155.0
SMOOTH_PCT          = 80
PEAK_MEAN_MIN_PEAKS = 10
SAVGOL_ORDER        = 2
WINDOWS             = [7, 9, 11, 13, 15]
PEAK_PROMINENCE     = 20.0
MAX_CADENCE_RPM     = 130
CONTIGUOUS_GAP      = 30    # frames
CONF_MIN            = 0.1

# Body25 joint indices (matches pipeline_v2/knee_analysis.py)
R_HIP, R_KNEE, R_ANKLE = 9, 10, 11
L_HIP, L_KNEE, L_ANKLE = 12, 13, 14

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── angle geometry ────────────────────────────────────────────────────────────

def get_xy(kp, idx):
    if not kp or idx >= len(kp):
        return None
    x, y, c = kp[idx]
    return (x, y) if c >= CONF_MIN and x > 0 and y > 0 else None


def calc_angle(A, B, C):
    a2 = (B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2
    b2 = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
    c2 = (A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2
    denom = 2 * math.sqrt(a2 * b2)
    if denom < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (a2 + b2 - c2) / denom))))


# ── peak detection (matches knee_analysis.py) ─────────────────────────────────

def detect_peaks(angles, timestamps, prominence):
    min_gap_sec = 60.0 / MAX_CADENCE_RPM
    arr = np.array(angles, dtype=float)
    adaptive_prom = max(prominence, float(arr.std()))
    if len(timestamps) >= 2:
        mean_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    else:
        mean_dt = 0
    min_gap_samples = max(1, int(min_gap_sec / mean_dt)) if mean_dt > 0 else 1
    peak_idxs, _ = scipy_find_peaks(arr, prominence=adaptive_prom, distance=min_gap_samples)
    return list(peak_idxs)


def autocorrelation_period(timestamps, smoothed):
    if len(smoothed) < 10:
        return None
    signal = np.array(smoothed, dtype=float) - np.mean(smoothed)
    corr = correlate(signal, signal, mode="full")
    corr = corr[len(corr) // 2:]
    if corr[0] == 0:
        return None
    corr /= corr[0]
    if len(timestamps) < 2:
        return None
    mean_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    if mean_dt <= 0:
        return None
    min_lag = max(1, int((60.0 / MAX_CADENCE_RPM) / mean_dt))
    ac_peaks, _ = scipy_find_peaks(corr[1:], height=0.1, distance=min_lag)
    if len(ac_peaks) == 0:
        return None
    period = (int(ac_peaks[0]) + 1) * mean_dt
    rpm = 60.0 / period
    if 20 <= rpm <= MAX_CADENCE_RPM:
        return period
    return None


def split_runs(series, gap=CONTIGUOUS_GAP):
    if not series:
        return []
    runs, current = [], [series[0]]
    for prev, curr in zip(series, series[1:]):
        if curr[0] - prev[0] <= gap:
            current.append(curr)
        else:
            runs.append(current)
            current = [curr]
    runs.append(current)
    return runs


# ── direction / knee selection ────────────────────────────────────────────────

def direction_from_log(log):
    frames = log.get("selected_frames", [])
    bursts = log.get("selected_bursts", [])

    def median_dir(entries):
        deltas = []
        for e in entries:
            fb, bb = e.get("fw_box"), e.get("bw_box")
            if fb and bb:
                deltas.append((fb[0] + fb[2]) / 2 - (bb[0] + bb[2]) / 2)
        if not deltas:
            return None
        deltas.sort()
        return "right" if deltas[len(deltas) // 2] > 0 else "left"

    if not bursts:
        return median_dir(frames), {}

    burst_dirs = {}
    for b in bursts:
        ents = [e for e in frames
                if b["start_frame_idx"] <= e.get("frame_idx", -1) <= b["end_frame_idx"]]
        burst_dirs[b["burst_id"]] = median_dir(ents)

    frame_dir_map = {}
    for e in frames:
        fi = e.get("frame_idx")
        if fi is None:
            continue
        for b in bursts:
            if b["start_frame_idx"] <= fi <= b["end_frame_idx"]:
                frame_dir_map[fi] = burst_dirs.get(b["burst_id"])
                break

    longest = max(bursts, key=lambda b: b.get("frame_count", 0))
    global_dir = burst_dirs.get(longest["burst_id"]) or median_dir(frames)
    return global_dir, frame_dir_map


def joints_for_dir(direction):
    if direction == "right":
        return (R_HIP, R_KNEE, R_ANKLE), (L_HIP, L_KNEE, L_ANKLE)
    return (L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)


# ── per-video raw angle reconstruction ───────────────────────────────────────

def reconstruct_raw_series(kp_data, log):
    direction, frame_dir_map = direction_from_log(log)
    global_dir = direction or "left"

    series = []
    for entry in kp_data.get("frames", []):
        kp  = entry.get("keypoints", [])
        idx = entry["frame_idx"]
        t   = entry["timestamp"]

        frame_dir = frame_dir_map.get(idx, global_dir)
        primary, fallback = joints_for_dir(frame_dir)

        hip   = get_xy(kp, primary[0])
        knee  = get_xy(kp, primary[1])
        ankle = get_xy(kp, primary[2])

        if not (hip and knee and ankle):
            hip   = get_xy(kp, fallback[0])
            knee  = get_xy(kp, fallback[1])
            ankle = get_xy(kp, fallback[2])

        angle = calc_angle(hip, knee, ankle) if (hip and knee and ankle) else None
        if angle is not None:
            series.append((idx, t, angle))

    return series


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_rpm(runs):
    periods = []
    for run in runs:
        peaks = run.get("peaks", [])
        if len(peaks) >= 2:
            ts = [p["timestamp"] for p in peaks]
            periods.extend(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
    if periods:
        return round(60.0 / (sum(periods) / len(periods)), 1), periods
    for run in runs:
        if run.get("autocorr"):
            return round(60.0 / run["autocorr"], 1), [run["autocorr"]]
    return None, []


def compute_seat(runs):
    usable_peaks = [p["angle"] for run in runs
                    if len(run.get("peaks", [])) >= 2
                    for p in run["peaks"]]
    smoothed = [ang for run in runs for ang in run.get("smoothed", [])]
    if len(usable_peaks) >= PEAK_MEAN_MIN_PEAKS:
        peak = sum(usable_peaks) / len(usable_peaks)
    elif smoothed:
        peak = float(np.percentile(smoothed, SMOOTH_PCT))
    else:
        return None, None
    v = "too_low" if peak < OPTIMAL_LOW else ("too_high" if peak > OPTIMAL_HIGH else "optimal")
    return peak, v


def analyse_run_with_window(run_series, window, prominence=PEAK_PROMINENCE):
    """Smooth and detect peaks for one run using given SavGol window."""
    if len(run_series) < 3:
        return {"peaks": [], "smoothed": [s[2] for s in run_series], "autocorr": None}

    frame_indices = [s[0] for s in run_series]
    timestamps    = [s[1] for s in run_series]
    raw_angles    = [s[2] for s in run_series]

    win = min(window, len(raw_angles))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        smoothed = raw_angles[:]
    else:
        smoothed = list(savgol_filter(raw_angles, window_length=win, polyorder=SAVGOL_ORDER))

    peak_idxs = detect_peaks(smoothed, timestamps, prominence) if len(smoothed) >= 3 else []

    autocorr = None
    if len(peak_idxs) < 2:
        autocorr = autocorrelation_period(timestamps, smoothed)

    peaks_out = [
        {"frame_idx": frame_indices[j], "timestamp": timestamps[j], "angle": raw_angles[j]}
        for j in peak_idxs
    ]
    return {"peaks": peaks_out, "smoothed": smoothed, "autocorr": autocorr}


# ── data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_gt(gt_path):
    import csv as _csv
    gt = {}
    with open(gt_path, newline="") as f:
        for row in _csv.DictReader(f):
            stem, val = row["video"].strip(), row["true_rpm"].strip()
            if stem and val:
                try:
                    gt[stem] = float(val)
                except ValueError:
                    pass
    return gt


def parse_stem(stem):
    m = re.match(r"^([a-z]+)(a|b)(\d+)a?$", stem)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def verdict_str(deg):
    if deg is None:
        return None
    return "too_low" if deg < OPTIMAL_LOW else ("too_high" if deg > OPTIMAL_HIGH else "optimal")


def collect_videos(v2_dir):
    """Load raw angle series, a-verdicts, and ground truth stems for all b-videos."""
    stems = [d for d in os.listdir(v2_dir) if os.path.isdir(os.path.join(v2_dir, d))]

    from collections import defaultdict
    by_subj_grp = defaultdict(lambda: {"a": [], "b": []})
    for stem in stems:
        parsed = parse_stem(stem)
        if parsed:
            subj, cond, grp = parsed
            by_subj_grp[(subj, grp)][cond].append(stem)

    videos = []
    for (subj, grp), conds in sorted(by_subj_grp.items()):
        a_ref_verdict = None
        for a_stem in sorted(conds.get("a", [])):
            ap = os.path.join(v2_dir, a_stem, f"{a_stem}_assessment.json")
            if not os.path.exists(ap):
                continue
            try:
                peak = load_json(ap)["summary"].get("knee_angle_peak")
                if peak is not None:
                    a_ref_verdict = verdict_str(peak)
                    break
            except (KeyError, json.JSONDecodeError):
                continue

        for b_stem in sorted(conds.get("b", [])):
            kp_path  = os.path.join(v2_dir, b_stem, f"{b_stem}_keypoints.json")
            log_path = os.path.join(v2_dir, b_stem, f"{b_stem}_selection_log.json")
            if not os.path.exists(kp_path) or not os.path.exists(log_path):
                continue
            try:
                kp_data = load_json(kp_path)
                log     = load_json(log_path)
            except json.JSONDecodeError:
                continue

            raw_series = reconstruct_raw_series(kp_data, log)
            if not raw_series:
                continue

            videos.append({
                "stem":      b_stem,
                "subject":   subj,
                "series":    raw_series,
                "a_verdict": a_ref_verdict,
            })

    return videos


# ── main sweep ────────────────────────────────────────────────────────────────

def run(v2_dir, gt_path):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    videos = collect_videos(v2_dir)

    if not videos:
        print("ERROR: no b-condition data found.", file=sys.stderr)
        sys.exit(1)

    gt = load_gt(gt_path) if gt_path and os.path.exists(gt_path) else {}

    print(f"\nSavitzky-Golay window sweep (order={SAVGOL_ORDER}, prominence={PEAK_PROMINENCE}°)")
    print(f"B-videos: {len(videos)}   GT-available: {sum(1 for v in videos if v['stem'] in gt)}")
    print()

    hdr = (f"{'window':>6}  {'mean_peaks':>10}  {'rpm_mae':>8}  "
           f"{'rpm_n':>6}  {'sh_agree':>9}  {'sh_n':>5}  {'mean_ipk_std':>13}")
    print(hdr)
    print("-" * len(hdr))

    csv_rows = []

    for win in WINDOWS:
        all_peak_counts = []
        rpm_errors      = []
        sh_agrees       = 0
        sh_total        = 0
        ipk_stds        = []

        for v in videos:
            raw_runs = split_runs(v["series"])
            proc_runs = [analyse_run_with_window(r, win) for r in raw_runs]

            # Combine runs with peak info
            combined = []
            for pr in proc_runs:
                combined.append(pr)

            total_peaks = sum(len(r["peaks"]) for r in combined)
            all_peak_counts.append(total_peaks)

            # Inter-peak period std
            run_periods = []
            for r in combined:
                peaks = r["peaks"]
                if len(peaks) >= 2:
                    ts = [p["timestamp"] for p in peaks]
                    run_periods.extend(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
            if len(run_periods) >= 2:
                ipk_stds.append(float(np.std(run_periods)))

            # RPM
            pred_rpm, _ = compute_rpm(combined)
            if v["stem"] in gt and pred_rpm is not None:
                rpm_errors.append(abs(pred_rpm - gt[v["stem"]]))

            # Seat height
            if v["a_verdict"] is not None:
                _, sv = compute_seat(combined)
                if sv is not None:
                    sh_total += 1
                    if sv == v["a_verdict"]:
                        sh_agrees += 1

        mean_peaks   = sum(all_peak_counts) / len(all_peak_counts) if all_peak_counts else 0
        rpm_mae      = sum(rpm_errors) / len(rpm_errors) if rpm_errors else None
        mean_ipk_std = sum(ipk_stds) / len(ipk_stds) if ipk_stds else None
        sh_pct       = 100 * sh_agrees / sh_total if sh_total else None

        rpm_note = f"{rpm_mae:.1f}" if rpm_mae is not None else "—"
        ipk_note = f"{mean_ipk_std:.4f}" if mean_ipk_std is not None else "—"
        sh_note  = f"{sh_agrees}/{sh_total}" if sh_total else "—"
        marker   = "  ← prod" if win == 11 else ""

        print(f"  w={win:<3}  {mean_peaks:>9.2f}  {rpm_note:>8}  "
              f"{len(rpm_errors):>6}  {sh_note:>9}  {sh_total:>5}  {ipk_note:>13}{marker}")

        csv_rows.append({
            "window":            win,
            "mean_peaks_per_video": round(mean_peaks, 2),
            "rpm_mae":           round(rpm_mae, 2) if rpm_mae is not None else "",
            "rpm_n":             len(rpm_errors),
            "seat_height_agree": sh_agrees,
            "seat_height_total": sh_total,
            "seat_height_agree_pct": round(sh_pct, 1) if sh_pct is not None else "",
            "mean_inter_peak_std": round(mean_ipk_std, 4) if mean_ipk_std is not None else "",
        })

    csv_path = os.path.join(RESULTS_DIR, "savgol_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window", "mean_peaks_per_video", "rpm_mae",
                                          "rpm_n", "seat_height_agree", "seat_height_total",
                                          "seat_height_agree_pct", "mean_inter_peak_std"])
        w.writeheader()
        w.writerows(csv_rows)

    print()
    print("── Interpretation ───────────────────────────────────────────────────────────────")
    valid = [r for r in csv_rows if r["rpm_mae"] != ""]
    if valid:
        best = min(valid, key=lambda r: r["rpm_mae"])
        prod = next(r for r in csv_rows if r["window"] == 11)
        print(f"  Best RPM MAE        : w={best['window']} ({best['rpm_mae']} RPM)")
        print(f"  Production (w=11)   : {prod['rpm_mae'] if prod['rpm_mae'] else '—'} RPM")
        if prod["rpm_mae"] and best["rpm_mae"] and \
                abs(float(prod["rpm_mae"]) - float(best["rpm_mae"])) <= 0.5:
            print(f"  Verdict             : w=11 within 0.5 RPM of optimum — CURRENT VALUE JUSTIFIED.")
        elif prod["rpm_mae"] and best["rpm_mae"]:
            print(f"  Verdict             : w={best['window']} outperforms w=11 by "
                  f"{float(prod['rpm_mae']) - float(best['rpm_mae']):.1f} RPM — CONSIDER CHANGING.")

    print(f"\n  CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", default="output_v2")
    parser.add_argument("--gt", default="evaluation/ground_truth.csv")
    args = parser.parse_args()
    if not os.path.isdir(args.v2):
        print(f"ERROR: {args.v2!r} not found — run from project root.", file=sys.stderr)
        sys.exit(1)
    run(args.v2, args.gt)


if __name__ == "__main__":
    main()
