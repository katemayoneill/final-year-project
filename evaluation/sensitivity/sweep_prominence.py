#!/usr/bin/env python3
"""
Tier 2 sensitivity sweep: peak prominence floor.

knee_analysis.py uses PEAK_PROMINENCE = 20.0 degrees as the minimum adaptive
prominence for peak detection. The actual prominence used per run is:
  max(PEAK_PROMINENCE, std(smoothed_angles_in_run))

This sweep varies PEAK_PROMINENCE over [10, 15, 20, 25, 30] while holding
SAVGOL_WINDOW = 11 (the production value) constant.

Reports per-prominence-floor:
  - Mean peaks found per b-video
  - RPM MAE against ground truth
  - Seat height verdict agreement (b vs a reference)

Raw angles are reconstructed from _keypoints.json using the same joint
selection logic as knee_analysis.py. No OpenPose re-run required.

Output:
  evaluation/sensitivity/results/prominence_sweep.csv
  Prints a summary table to stdout.

Usage (from project root):
  python3 evaluation/sensitivity/sweep_prominence.py [--v2 output_v2/] [--gt evaluation/ground_truth.csv]
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
SAVGOL_WINDOW       = 11    # production value — held constant
SAVGOL_ORDER        = 2
PROMINENCES         = [10, 15, 20, 25, 30]
MAX_CADENCE_RPM     = 130
CONTIGUOUS_GAP      = 30
CONF_MIN            = 0.1

R_HIP, R_KNEE, R_ANKLE = 9, 10, 11
L_HIP, L_KNEE, L_ANKLE = 12, 13, 14

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── geometry (same as utils.py) ───────────────────────────────────────────────

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


# ── SavGol + peak detection ───────────────────────────────────────────────────

def smooth(angles):
    win = min(SAVGOL_WINDOW, len(angles))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return angles[:]
    return list(savgol_filter(angles, window_length=win, polyorder=SAVGOL_ORDER))


def detect_peaks(angles, timestamps, prominence_floor):
    arr = np.array(angles, dtype=float)
    adaptive_prom = max(prominence_floor, float(arr.std()))
    min_gap_sec = 60.0 / MAX_CADENCE_RPM
    if len(timestamps) >= 2:
        mean_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    else:
        mean_dt = 0
    min_gap = max(1, int(min_gap_sec / mean_dt)) if mean_dt > 0 else 1
    idxs, _ = scipy_find_peaks(arr, prominence=adaptive_prom, distance=min_gap)
    return list(idxs)


def autocorr_period(timestamps, smoothed):
    if len(smoothed) < 10:
        return None
    sig = np.array(smoothed, dtype=float) - np.mean(smoothed)
    corr = correlate(sig, sig, mode="full")
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
    if not len(ac_peaks):
        return None
    period = (int(ac_peaks[0]) + 1) * mean_dt
    if 20 <= 60.0 / period <= MAX_CADENCE_RPM:
        return period
    return None


def split_runs(series):
    if not series:
        return []
    runs, cur = [], [series[0]]
    for prev, curr in zip(series, series[1:]):
        if curr[0] - prev[0] <= CONTIGUOUS_GAP:
            cur.append(curr)
        else:
            runs.append(cur)
            cur = [curr]
    runs.append(cur)
    return runs


# ── direction / knee ──────────────────────────────────────────────────────────

def direction_from_log(log):
    frames = log.get("selected_frames", [])
    bursts = log.get("selected_bursts", [])

    def median_dir(entries):
        deltas = []
        for e in entries:
            fb, bb = e.get("front_wheel_box"), e.get("back_wheel_box")
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
                if b["frame_idx_start"] <= e.get("frame_idx", -1) <= b["frame_idx_end"]]
        burst_dirs[b["burst_id"]] = median_dir(ents)

    frame_dir_map = {}
    for e in frames:
        fi = e.get("frame_idx")
        if fi is None:
            continue
        for b in bursts:
            if b["frame_idx_start"] <= fi <= b["frame_idx_end"]:
                frame_dir_map[fi] = burst_dirs.get(b["burst_id"])
                break

    longest = max(bursts, key=lambda b: b.get("frame_count", 0))
    return burst_dirs.get(longest["burst_id"]) or median_dir(frames), frame_dir_map


def joints_for_dir(direction):
    if direction == "right":
        return (R_HIP, R_KNEE, R_ANKLE), (L_HIP, L_KNEE, L_ANKLE)
    return (L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)


def reconstruct_raw(kp_data, log):
    direction, frame_dir_map = direction_from_log(log)
    global_dir = direction or "left"
    series = []
    for entry in kp_data.get("frames", []):
        kp = entry.get("keypoints", [])
        idx, t = entry["frame_idx"], entry["timestamp"]
        fd = frame_dir_map.get(idx, global_dir)
        primary, fallback = joints_for_dir(fd)
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


# ── per-run analysis ──────────────────────────────────────────────────────────

def analyse_run(run_series, prominence_floor):
    if len(run_series) < 3:
        return {"peaks": [], "smoothed": [s[2] for s in run_series], "autocorr": None}
    frame_idxs = [s[0] for s in run_series]
    timestamps = [s[1] for s in run_series]
    raw        = [s[2] for s in run_series]
    sm         = smooth(raw)
    pk_idxs    = detect_peaks(sm, timestamps, prominence_floor)
    ac         = None
    if len(pk_idxs) < 2:
        ac = autocorr_period(timestamps, sm)
    peaks_out = [{"frame_idx": frame_idxs[j], "timestamp": timestamps[j], "angle": raw[j]}
                 for j in pk_idxs]
    return {"peaks": peaks_out, "smoothed": sm, "autocorr": ac}


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_rpm(runs):
    periods = []
    for r in runs:
        pks = r["peaks"]
        if len(pks) >= 2:
            ts = [p["timestamp"] for p in pks]
            periods.extend(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
    if periods:
        return round(60.0 / (sum(periods) / len(periods)), 1)
    for r in runs:
        if r.get("autocorr"):
            return round(60.0 / r["autocorr"], 1)
    return None


def compute_seat(runs):
    peaks_all = [p["angle"] for r in runs
                 if len(r["peaks"]) >= 2 for p in r["peaks"]]
    smoothed  = [ang for r in runs for ang in r["smoothed"]]
    if len(peaks_all) >= PEAK_MEAN_MIN_PEAKS:
        peak = sum(peaks_all) / len(peaks_all)
    elif smoothed:
        peak = float(np.percentile(smoothed, SMOOTH_PCT))
    else:
        return None, None
    v = "too_low" if peak < OPTIMAL_LOW else ("too_high" if peak > OPTIMAL_HIGH else "optimal")
    return peak, v


# ── loading helpers ───────────────────────────────────────────────────────────

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
        a_ref = None
        for a_stem in sorted(conds.get("a", [])):
            ap = os.path.join(v2_dir, a_stem, f"{a_stem}_assessment.json")
            if not os.path.exists(ap):
                continue
            try:
                peak = load_json(ap)["summary"].get("knee_angle_peak")
                if peak is not None:
                    a_ref = verdict_str(peak)
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
            series = reconstruct_raw(kp_data, log)
            if not series:
                continue
            videos.append({"stem": b_stem, "subject": subj, "series": series, "a_verdict": a_ref})

    return videos


# ── main ──────────────────────────────────────────────────────────────────────

def run(v2_dir, gt_path):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    videos = collect_videos(v2_dir)
    if not videos:
        print("ERROR: no b-condition data found.", file=sys.stderr)
        sys.exit(1)

    gt = load_gt(gt_path) if gt_path and os.path.exists(gt_path) else {}

    print(f"\nPeak prominence floor sweep (savgol_window={SAVGOL_WINDOW} held constant)")
    print(f"B-videos: {len(videos)}   GT-available: {sum(1 for v in videos if v['stem'] in gt)}")
    print()

    hdr = (f"{'prom':>5}  {'mean_peaks':>10}  {'rpm_mae':>8}  "
           f"{'rpm_n':>6}  {'sh_agree':>9}  {'sh_n':>5}")
    print(hdr)
    print("-" * len(hdr))

    csv_rows = []
    for prom in PROMINENCES:
        peak_counts = []
        rpm_errors  = []
        sh_agrees   = 0
        sh_total    = 0

        for v in videos:
            raw_runs  = split_runs(v["series"])
            proc_runs = [analyse_run(r, prom) for r in raw_runs]
            peak_counts.append(sum(len(r["peaks"]) for r in proc_runs))

            pred_rpm = compute_rpm(proc_runs)
            if v["stem"] in gt and pred_rpm is not None:
                rpm_errors.append(abs(pred_rpm - gt[v["stem"]]))

            if v["a_verdict"] is not None:
                _, sv = compute_seat(proc_runs)
                if sv is not None:
                    sh_total += 1
                    if sv == v["a_verdict"]:
                        sh_agrees += 1

        mean_peaks = sum(peak_counts) / len(peak_counts) if peak_counts else 0
        rpm_mae    = sum(rpm_errors) / len(rpm_errors) if rpm_errors else None
        sh_pct     = 100 * sh_agrees / sh_total if sh_total else None
        marker     = "  ← prod" if prom == 20 else ""

        rpm_note = f"{rpm_mae:.1f}" if rpm_mae is not None else "—"
        sh_note  = f"{sh_agrees}/{sh_total}" if sh_total else "—"
        sh_pct_str = f"{sh_pct:.0f}%" if sh_pct is not None else "—"
        print(f"  p={prom:<4}  {mean_peaks:>9.2f}  {rpm_note:>8}  "
              f"{len(rpm_errors):>6}  {sh_note:>9}  {sh_total:>5}  {sh_pct_str}{marker}")

        csv_rows.append({
            "prominence_floor":     prom,
            "mean_peaks_per_video": round(mean_peaks, 2),
            "rpm_mae":              round(rpm_mae, 2) if rpm_mae is not None else "",
            "rpm_n":                len(rpm_errors),
            "seat_height_agree":    sh_agrees,
            "seat_height_total":    sh_total,
            "seat_height_agree_pct": round(sh_pct, 1) if sh_pct is not None else "",
        })

    csv_path = os.path.join(RESULTS_DIR, "prominence_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prominence_floor", "mean_peaks_per_video",
                                          "rpm_mae", "rpm_n", "seat_height_agree",
                                          "seat_height_total", "seat_height_agree_pct"])
        w.writeheader()
        w.writerows(csv_rows)

    print()
    print("── Interpretation ───────────────────────────────────────────────────────────────")
    valid = [r for r in csv_rows if r["rpm_mae"] != ""]
    if valid:
        best = min(valid, key=lambda r: r["rpm_mae"])
        prod = next(r for r in csv_rows if r["prominence_floor"] == 20)
        print(f"  Best RPM MAE        : prom={best['prominence_floor']}° ({best['rpm_mae']} RPM)")
        print(f"  Production (p=20°)  : {prod['rpm_mae'] if prod['rpm_mae'] else '—'} RPM")
        if prod["rpm_mae"] and best["rpm_mae"] and \
                abs(float(prod["rpm_mae"]) - float(best["rpm_mae"])) <= 0.5:
            print(f"  Verdict             : p=20° within 0.5 RPM of optimum — CURRENT VALUE JUSTIFIED.")
        elif prod["rpm_mae"] and best["rpm_mae"]:
            print(f"  Verdict             : prom={best['prominence_floor']}° outperforms p=20° — CONSIDER CHANGING.")

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
