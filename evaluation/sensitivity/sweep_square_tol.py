#!/usr/bin/env python3
"""
Tier 1 sensitivity sweep: SQUARE_TOL — wheel bounding-box squareness gate.

side_angle.py accepts frames only when both wheel bounding boxes have an
aspect-ratio deviation from square < SQUARE_TOL (default 0.15).  The stored
fw_squareness / bw_squareness fields in selection_log.json encode each frame's
squareness as (1 - |1 - aspect_ratio|), where 1.0 is a perfect square.

A frame passes the gate when:
    fw_squareness > 1 - SQUARE_TOL  AND  bw_squareness > 1 - SQUARE_TOL

This lets us simulate TIGHTER thresholds (0.08, 0.10, 0.12) from existing
selection logs without re-running YOLO.

LIMITATION: Looser thresholds (> 0.15) cannot be simulated — frames that
failed the 0.15 gate are absent from the log.  The 0.18 entry is marked N/A
in the CSV.

For each simulatable threshold:
  1. Filter selected_frames by the new squareness criterion
  2. Regroup filtered frames into contiguous bursts (gap <= 1 frame)
  3. Apply quality scoring (fw/bw squareness * size_ratio * cyclist height)
     with QUALITY_FRACTION=0.5 and MIN_BURST_FRAMES=5
  4. Find knee_analysis runs that overlap any retained burst
  5. Filter each run's angle_series and peaks to the valid frame set
  6. Recompute RPM (pooled inter-peak periods + autocorr fallback),
     seat height (smooth_p80 / peak_mean), and angle range (max - min)

RPM MAE reported against evaluation/ground_truth.csv (b-condition only).
Seat height agreement: b-condition verdict vs a-condition reference.

Output:
  evaluation/sensitivity/results/square_tol_sweep.csv
  Prints a summary table to stdout.

Usage (from project root):
  python3 evaluation/sensitivity/sweep_square_tol.py [--v2 output_v2/] [--gt evaluation/ground_truth.csv]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# ── pipeline constants (match pipeline_v2 defaults) ───────────────────────────
OPTIMAL_LOW         = 145.0
OPTIMAL_HIGH        = 155.0
SMOOTH_PCT          = 80
PEAK_MEAN_MIN_PEAKS = 10
QUALITY_FRACTION    = 0.5
MIN_BURST_FRAMES    = 5
APPROX_HEIGHT_FRAC  = 0.5   # fallback when no cyclist_box available

SWEEP_TOLS   = [0.08, 0.10, 0.12, 0.15]  # simulatable from existing logs
SKIPPED_TOLS = [0.18]                      # requires re-running YOLO

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── helpers ───────────────────────────────────────────────────────────────────

def verdict(deg):
    if deg is None:
        return None
    return "too_low" if deg < OPTIMAL_LOW else ("too_high" if deg > OPTIMAL_HIGH else "optimal")


def parse_stem(stem):
    m = re.match(r"^([a-z]+)(a|b)(\d+)a?$", stem)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_gt(gt_path):
    gt = {}
    with open(gt_path, newline="") as f:
        for row in csv.DictReader(f):
            stem = row["video"].strip()
            val  = row["true_rpm"].strip()
            if stem and val:
                try:
                    gt[stem] = float(val)
                except ValueError:
                    pass
    return gt


def quality_score_frames(frames, frame_h):
    """Quality score for a list of frame dicts, matching side_angle.py logic."""
    n = len(frames)
    if n == 0:
        return 0.0
    sq   = sum((f["fw_squareness"] + f["bw_squareness"]) / 2 for f in frames) / n
    size = sum(f.get("wheel_size_ratio", 1.0) for f in frames) / n
    heights = [
        (f["cyclist_box"][3] - f["cyclist_box"][1]) / frame_h
        for f in frames if f.get("cyclist_box")
    ]
    h = sum(heights) / len(heights) if heights else APPROX_HEIGHT_FRAC
    return n * sq * size * h


def filter_and_burst(selected_frames, tol, frame_h):
    """
    Filter frames by tol, regroup into contiguous bursts, apply quality
    scoring + QUALITY_FRACTION + MIN_BURST_FRAMES.

    Returns:
        kept_bursts    — list of {start_frame_idx, end_frame_idx, frame_count}
        valid_frame_idx — set of frame_idx values in retained bursts
        valid_timestamps — set of rounded timestamps in retained bursts
    """
    sq_floor = 1.0 - tol
    passing  = [
        f for f in selected_frames
        if f["fw_squareness"] > sq_floor and f["bw_squareness"] > sq_floor
    ]

    if not passing:
        return [], set(), set()

    # Group into contiguous bursts (consecutive frame_idx)
    bursts  = []
    current = [passing[0]]
    for prev, curr in zip(passing, passing[1:]):
        if curr["frame_idx"] - prev["frame_idx"] == 1:
            current.append(curr)
        else:
            bursts.append(current)
            current = [curr]
    bursts.append(current)

    # Quality score every burst then apply threshold
    scores     = [quality_score_frames(b, frame_h) for b in bursts]
    best_score = max(scores)
    q_thresh   = QUALITY_FRACTION * best_score

    good_pairs = [
        (s, b) for s, b in zip(scores, bursts)
        if s >= q_thresh and len(b) >= MIN_BURST_FRAMES
    ]
    if not good_pairs:
        # Always keep at least the best burst (matches side_angle.py fallback)
        best_i = scores.index(best_score)
        good_pairs = [(best_score, bursts[best_i])]

    good_pairs.sort(key=lambda x: x[1][0]["frame_idx"])

    kept_bursts = [
        {
            "start_frame_idx": b[0]["frame_idx"],
            "end_frame_idx":   b[-1]["frame_idx"],
            "frame_count":     len(b),
        }
        for _, b in good_pairs
    ]
    valid_frame_idx  = {f["frame_idx"] for _, b in good_pairs for f in b}
    valid_timestamps = {round(f["timestamp"], 4) for _, b in good_pairs for f in b}

    return kept_bursts, valid_frame_idx, valid_timestamps


def run_overlaps_burst(run, kept_bursts):
    """True if the run's frame range overlaps any kept burst."""
    rs, re_ = run["frame_idx_start"], run["frame_idx_end"]
    return any(rs <= b["end_frame_idx"] and re_ >= b["start_frame_idx"]
               for b in kept_bursts)


def filter_run(run, valid_frame_idx, valid_timestamps):
    """Return run with angle_series and peaks restricted to valid frames."""
    peaks = [p for p in run.get("peaks", [])
             if p["frame_idx"] in valid_frame_idx]
    angle_series = [
        [ts, ang] for ts, ang in run.get("angle_series", [])
        if round(ts, 4) in valid_timestamps
    ]
    return {**run, "peaks": peaks, "angle_series": angle_series}


def compute_rpm(filtered_runs):
    """Pool inter-peak periods; autocorr fallback if no run has >= 2 peaks."""
    periods = []
    for run in filtered_runs:
        peaks = run.get("peaks", [])
        if len(peaks) >= 2:
            ts = [p["timestamp"] for p in peaks]
            periods.extend(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
    if periods:
        return round(60.0 / (sum(periods) / len(periods)), 1)
    for run in filtered_runs:
        if run.get("peak_method") == "autocorrelation" and run.get("autocorr_period_sec"):
            return round(60.0 / run["autocorr_period_sec"], 1)
    return None


def compute_seat_verdict(filtered_runs):
    """Adaptive peak selection: peak_mean if >= 10 validated peaks, else smooth_p80."""
    usable_peaks = [
        p["angle"]
        for run in filtered_runs
        if len(run.get("peaks", [])) >= 2
        for p in run["peaks"]
    ]
    smoothed = [ang for run in filtered_runs for _, ang in run.get("angle_series", [])]

    if len(usable_peaks) >= PEAK_MEAN_MIN_PEAKS:
        peak = sum(usable_peaks) / len(usable_peaks)
    elif smoothed:
        peak = float(np.percentile(smoothed, SMOOTH_PCT))
    else:
        return None, None
    return peak, verdict(peak)


def compute_angle_range(filtered_runs):
    """Max - min of all angle_series values across retained runs."""
    angles = [ang for run in filtered_runs for _, ang in run.get("angle_series", [])]
    return (max(angles) - min(angles)) if len(angles) >= 2 else None


# ── data loading ──────────────────────────────────────────────────────────────

def collect_video_data(v2_dir):
    """
    Load per-b-video data needed for the sweep.
    Returns list of dicts: stem, selected_frames, ka_runs, a_verdict, frame_h.
    """
    stems = [d for d in os.listdir(v2_dir) if os.path.isdir(os.path.join(v2_dir, d))]

    by_subj_grp = defaultdict(lambda: {"a": [], "b": []})
    for stem in stems:
        parsed = parse_stem(stem)
        if parsed:
            subj, cond, grp = parsed
            by_subj_grp[(subj, grp)][cond].append(stem)

    videos = []
    for (subj, grp), conds in sorted(by_subj_grp.items()):
        # Derive a-condition reference verdict
        a_ref_verdict = None
        for a_stem in sorted(conds.get("a", [])):
            ap = os.path.join(v2_dir, a_stem, f"{a_stem}_assessment.json")
            if not os.path.exists(ap):
                continue
            try:
                peak = load_json(ap)["summary"].get("knee_angle_peak")
                if peak is not None:
                    a_ref_verdict = verdict(peak)
                    break
            except (KeyError, json.JSONDecodeError):
                continue

        for b_stem in sorted(conds.get("b", [])):
            log_path = os.path.join(v2_dir, b_stem, f"{b_stem}_selection_log.json")
            ka_path  = os.path.join(v2_dir, b_stem, f"{b_stem}_knee_analysis.json")
            if not os.path.exists(log_path) or not os.path.exists(ka_path):
                continue
            try:
                log = load_json(log_path)
                ka  = load_json(ka_path)
            except json.JSONDecodeError:
                continue

            frames = log.get("selected_frames", [])
            if not frames or "fw_squareness" not in frames[0]:
                continue

            # Estimate frame_h from cyclist box y2 values; fallback to 1080
            ys     = [f["cyclist_box"][3] for f in frames if f.get("cyclist_box")]
            frame_h = max(ys) * 1.05 if ys else 1080.0

            videos.append({
                "stem":            b_stem,
                "selected_frames": frames,
                "ka_runs":         ka.get("runs", []),
                "a_verdict":       a_ref_verdict,
                "frame_h":         frame_h,
            })

    return videos


# ── main sweep ────────────────────────────────────────────────────────────────

def run_sweep(v2_dir, gt_path):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    videos = collect_video_data(v2_dir)

    if not videos:
        print("ERROR: no b-condition data with selection logs found.", file=sys.stderr)
        sys.exit(1)

    gt = load_gt(gt_path) if gt_path and os.path.exists(gt_path) else {}

    print("\nSQUARE_TOL sensitivity sweep — wheel bounding-box squareness gate")
    print(f"  Production default: SQUARE_TOL = 0.15")
    print(f"  LIMITATION: Thresholds > 0.15 cannot be simulated from existing logs.")
    print(f"              Frames failing the 0.15 gate are absent from selection_log.json.")
    print(f"\nB-videos loaded: {len(videos)}")
    print(f"B-videos with ground-truth RPM: {sum(1 for v in videos if v['stem'] in gt)}")

    # Show per-video squareness distribution at production threshold
    print()
    print("Per-video frame counts at each simulated threshold:")
    hdr2 = f"  {'stem':<22}  {'tol=0.08':>8}  {'tol=0.10':>8}  {'tol=0.12':>8}  {'tol=0.15':>8}"
    print(hdr2)
    for v in videos:
        counts = []
        for tol in SWEEP_TOLS:
            sq_floor = 1.0 - tol
            n = sum(1 for f in v["selected_frames"]
                    if f["fw_squareness"] > sq_floor and f["bw_squareness"] > sq_floor)
            counts.append(n)
        print(f"  {v['stem']:<22}  {counts[0]:>8}  {counts[1]:>8}  {counts[2]:>8}  {counts[3]:>8}")

    print()
    hdr = (f"  {'tol':>5}  {'mean_frames':>11}  {'rpm_mae':>8}  {'rpm_n':>6}  "
           f"{'sh_agree':>9}  {'sh_pct':>7}  {'mean_ang_range':>14}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    csv_rows = []
    prod_row = None

    all_tols = SWEEP_TOLS + SKIPPED_TOLS
    for tol in all_tols:
        if tol in SKIPPED_TOLS:
            print(f"  tol={tol:.2f}  SKIPPED — requires re-running YOLO (looser than production 0.15)")
            csv_rows.append({
                "square_tol":            tol,
                "mean_frames_per_video": "",
                "rpm_mae":               "",
                "rpm_n":                 "",
                "seat_height_agree":     "",
                "seat_height_total":     "",
                "seat_height_agree_pct": "",
                "mean_angle_range":      "",
                "note":                  "requires_yolo_rerun",
            })
            continue

        total_frames = 0
        rpm_errors   = []
        sh_agrees    = 0
        sh_total     = 0
        angle_ranges = []

        for v in videos:
            kept_bursts, valid_idx, valid_ts = filter_and_burst(
                v["selected_frames"], tol, v["frame_h"]
            )
            total_frames += sum(b["frame_count"] for b in kept_bursts)

            overlapping   = [r for r in v["ka_runs"] if run_overlaps_burst(r, kept_bursts)]
            filtered_runs = [filter_run(r, valid_idx, valid_ts) for r in overlapping]

            ar = compute_angle_range(filtered_runs)
            if ar is not None:
                angle_ranges.append(ar)

            if v["stem"] in gt:
                pred_rpm = compute_rpm(filtered_runs)
                if pred_rpm is not None:
                    rpm_errors.append(abs(pred_rpm - gt[v["stem"]]))

            if v["a_verdict"] is not None:
                _, v_pred = compute_seat_verdict(filtered_runs)
                if v_pred is not None:
                    sh_total += 1
                    if v_pred == v["a_verdict"]:
                        sh_agrees += 1

        n       = len(videos)
        rpm_mae = sum(rpm_errors) / len(rpm_errors) if rpm_errors else None
        sh_pct  = 100 * sh_agrees / sh_total if sh_total else None
        mean_ar = sum(angle_ranges) / len(angle_ranges) if angle_ranges else None

        rpm_str    = f"{rpm_mae:.1f}" if rpm_mae is not None else "—"
        sh_str     = f"{sh_agrees}/{sh_total}" if sh_total else "—"
        sh_pct_str = f"{sh_pct:.0f}%" if sh_pct is not None else "—"
        ar_str     = f"{mean_ar:.1f}°" if mean_ar is not None else "—"

        print(f"  tol={tol:.2f}  {total_frames / n:>10.1f}  {rpm_str:>8}  "
              f"{len(rpm_errors):>6}  {sh_str:>9}  {sh_pct_str:>7}  {ar_str:>13}")

        row = {
            "square_tol":            tol,
            "mean_frames_per_video": round(total_frames / n, 1),
            "rpm_mae":               round(rpm_mae, 2) if rpm_mae is not None else "",
            "rpm_n":                 len(rpm_errors),
            "seat_height_agree":     sh_agrees,
            "seat_height_total":     sh_total,
            "seat_height_agree_pct": round(sh_pct, 1) if sh_pct is not None else "",
            "mean_angle_range":      round(mean_ar, 2) if mean_ar is not None else "",
            "note":                  "",
        }
        csv_rows.append(row)
        if tol == 0.15:
            prod_row = row

    # ── write CSV ─────────────────────────────────────────────────────────────
    csv_path  = os.path.join(RESULTS_DIR, "square_tol_sweep.csv")
    fieldnames = [
        "square_tol", "mean_frames_per_video", "rpm_mae", "rpm_n",
        "seat_height_agree", "seat_height_total", "seat_height_agree_pct",
        "mean_angle_range", "note",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    # ── interpretation + upgrade decision ────────────────────────────────────
    print()
    print("── Interpretation ───────────────────────────────────────────────────────────────")
    if prod_row:
        pr_mae = prod_row["rpm_mae"]
        print(f"  Production (tol=0.15): mean {prod_row['mean_frames_per_video']} frames/video, "
              f"RPM MAE={pr_mae} (n={prod_row['rpm_n']}), "
              f"SH agree={prod_row['seat_height_agree']}/{prod_row['seat_height_total']}")

    print()
    print("  Upgrade criteria:")
    print("    – RPM MAE improves by >= 0.5 RPM vs production")
    print("    – Seat height agreement stays the same or improves")
    print("    – >= 14 videos still produce RPM output (lose at most 1)")
    print()

    prod_mae = float(prod_row["rpm_mae"]) if prod_row and prod_row["rpm_mae"] != "" else None
    prod_sh  = int(prod_row["seat_height_agree"]) if prod_row else 0
    prod_n   = int(prod_row["rpm_n"]) if prod_row else 0

    meeting = []
    for row in csv_rows:
        if row.get("note") == "requires_yolo_rerun":
            continue
        t     = float(row["square_tol"])
        if t == 0.15:
            continue
        mae   = float(row["rpm_mae"]) if row["rpm_mae"] != "" else None
        sh_a  = int(row["seat_height_agree"]) if row["seat_height_agree"] != "" else 0
        n_rpm = int(row["rpm_n"]) if row["rpm_n"] != "" else 0

        if mae is None or prod_mae is None:
            continue
        mae_ok = mae <= prod_mae - 0.5
        sh_ok  = sh_a >= prod_sh
        n_ok   = n_rpm >= 14
        status = "MEETS" if (mae_ok and sh_ok and n_ok) else "fails"
        reasons = []
        if not mae_ok:
            reasons.append(f"MAE not improved by 0.5 ({mae:.2f} vs {prod_mae:.2f})")
        if not sh_ok:
            reasons.append(f"SH agree dropped ({sh_a} < {prod_sh})")
        if not n_ok:
            reasons.append(f"too few RPM outputs ({n_rpm} < 14)")
        reason_str = "; ".join(reasons) if reasons else "all criteria met"
        print(f"  tol={t:.2f}: {status} — {reason_str}")
        if mae_ok and sh_ok and n_ok:
            meeting.append((t, mae))

    print()
    if meeting:
        best_tol, best_mae = min(meeting, key=lambda x: x[1])
        print(f"  RECOMMENDATION: Change SQUARE_TOL to {best_tol:.2f} (RPM MAE={best_mae})")
    else:
        print("  RECOMMENDATION: Keep SQUARE_TOL = 0.15 — no tighter threshold meets criteria.")

    print(f"\n  CSV saved: {csv_path}")

    return meeting[0][0] if meeting else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2",  default="output_v2",
                        help="Path to output_v2/ directory (default: output_v2)")
    parser.add_argument("--gt",  default="evaluation/ground_truth.csv",
                        help="Path to ground_truth.csv (default: evaluation/ground_truth.csv)")
    args = parser.parse_args()

    if not os.path.isdir(args.v2):
        print(f"ERROR: {args.v2!r} not found — run from project root.", file=sys.stderr)
        sys.exit(1)

    run_sweep(args.v2, args.gt)


if __name__ == "__main__":
    main()
