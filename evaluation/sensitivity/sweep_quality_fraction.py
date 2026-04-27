#!/usr/bin/env python3
"""
Tier 1 sensitivity sweep: burst quality fraction threshold (LIMITED SWEEP).

side_angle_select.py keeps bursts scoring >= QUALITY_FRACTION (default 0.5) of
the best burst's quality_score. The selection_log.json records ONLY the already-kept
bursts — rejected bursts are not logged.

LIMITATION: Because unkept bursts are absent from the log, this sweep can only
simulate RAISING the threshold from 0.5 (i.e., dropping more bursts from the kept
set). It cannot simulate lowering the threshold below 0.5 without re-running YOLO.
This limitation is documented in the CSV header and the printed output.

For each threshold t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
  - A burst is retained if its quality_score >= t * best_score (within that video).
  - Knee-analysis runs that fall within retained burst frame ranges are kept.
  - RPM is recomputed from the filtered runs' peaks (pooled inter-peak periods).
  - Seat height is recomputed using smooth_p80 on the filtered smoothed series
    (peak_mean only if the retained run count >= PEAK_MEAN_MIN_PEAKS = 10).

RPM MAE is reported against evaluation/ground_truth.csv (b-condition only).
Seat height agreement is reported as b-condition vs a-condition reference.

Output:
  evaluation/sensitivity/results/quality_fraction_sweep.csv
  Prints a summary table to stdout.

Usage (from project root):
  python3 evaluation/sensitivity/sweep_quality_fraction.py [--v2 output_v2/] [--gt evaluation/ground_truth.csv]
"""

import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np

OPTIMAL_LOW         = 145.0
OPTIMAL_HIGH        = 155.0
SMOOTH_PCT          = 80
PEAK_MEAN_MIN_PEAKS = 10
THRESHOLDS          = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


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
    import csv as _csv
    gt = {}
    with open(gt_path, newline="") as f:
        for row in _csv.DictReader(f):
            stem = row["video"].strip()
            val  = row["true_rpm"].strip()
            if stem and val:
                try:
                    gt[stem] = float(val)
                except ValueError:
                    pass
    return gt


def run_in_bursts(run, kept_bursts):
    """True if run's frame range falls entirely within any kept burst."""
    for b in kept_bursts:
        if (run["frame_idx_start"] >= b["start_frame_idx"] and
                run["frame_idx_end"] <= b["end_frame_idx"]):
            return True
    return False


def compute_rpm(filtered_runs):
    """Pool inter-peak periods from runs with >= 2 peaks; autocorr fallback."""
    all_periods = []
    for run in filtered_runs:
        peaks = run.get("peaks", [])
        if len(peaks) >= 2:
            ts = [p["timestamp"] for p in peaks]
            all_periods.extend(ts[i + 1] - ts[i] for i in range(len(ts) - 1))

    if all_periods:
        avg = sum(all_periods) / len(all_periods)
        return round(60.0 / avg, 1)

    # Autocorr fallback
    for run in filtered_runs:
        if run.get("peak_method") == "autocorrelation" and run.get("autocorr_period_sec"):
            return round(60.0 / run["autocorr_period_sec"], 1)
    return None


def compute_seat_verdict(filtered_runs):
    """Recompute seat height verdict from filtered runs."""
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


def collect_video_data(v2_dir):
    """
    Returns list of dicts per b-video with all data needed for the sweep.
    Keys: stem, subject, bursts (list), ka_runs (list), a_verdict
    """
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
        # a-condition reference verdict (for seat height comparison)
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
                log    = load_json(log_path)
                ka     = load_json(ka_path)
            except json.JSONDecodeError:
                continue

            bursts = log.get("selected_bursts", [])
            if not bursts:
                continue

            videos.append({
                "stem":      b_stem,
                "subject":   subj,
                "bursts":    bursts,
                "ka_runs":   ka.get("runs", []),
                "a_verdict": a_ref_verdict,
            })

    return videos


def run(v2_dir, gt_path):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    videos = collect_video_data(v2_dir)

    if not videos:
        print("ERROR: no b-condition data with selection logs found.", file=sys.stderr)
        sys.exit(1)

    gt = load_gt(gt_path) if gt_path and os.path.exists(gt_path) else {}
    b_stems_in_gt = {v["stem"] for v in videos if v["stem"] in gt}

    print(f"\nQuality fraction sweep — burst retention threshold (LIMITED: upward from 0.5 only)")
    print(f"  LIMITATION: Selection log records only kept bursts. Thresholds below 0.5")
    print(f"  cannot be simulated without re-running YOLO side-angle detection.")
    print(f"\nB-videos available: {len(videos)}")
    print(f"B-videos with ground-truth RPM: {len(b_stems_in_gt)}")

    # Show burst distribution
    print()
    print("Per-video burst summary (at production threshold t=0.5):")
    for v in videos:
        best_score = max(b["quality_score"] for b in v["bursts"])
        scores_str = ", ".join(f"{b['quality_score']:.1f}" for b in v["bursts"])
        print(f"  {v['stem']:<22}  bursts_kept={len(v['bursts'])}  "
              f"scores=[{scores_str}]  best={best_score:.1f}")

    print()
    hdr = (f"{'thresh':>6}  {'mean_bursts':>11}  {'mean_frames':>11}  "
           f"{'rpm_mae':>8}  {'rpm_n':>6}  {'sh_agree':>9}  {'sh_n':>5}")
    print(hdr)
    print("-" * len(hdr))

    csv_rows = []

    for t in THRESHOLDS:
        total_bursts = 0
        total_frames = 0
        rpm_errors   = []
        sh_agrees    = 0
        sh_total     = 0

        for v in videos:
            best_score    = max(b["quality_score"] for b in v["bursts"])
            kept_bursts   = [b for b in v["bursts"] if b["quality_score"] >= t * best_score]
            filtered_runs = [r for r in v["ka_runs"] if run_in_bursts(r, kept_bursts)]

            total_bursts += len(kept_bursts)
            total_frames += sum(b["frame_count"] for b in kept_bursts)

            # RPM
            if v["stem"] in gt:
                pred_rpm = compute_rpm(filtered_runs)
                if pred_rpm is not None:
                    rpm_errors.append(abs(pred_rpm - gt[v["stem"]]))

            # Seat height
            if v["a_verdict"] is not None:
                _, v_pred = compute_seat_verdict(filtered_runs)
                if v_pred is not None:
                    sh_total += 1
                    if v_pred == v["a_verdict"]:
                        sh_agrees += 1

        n = len(videos)
        rpm_mae  = sum(rpm_errors) / len(rpm_errors) if rpm_errors else None
        sh_pct   = 100 * sh_agrees / sh_total if sh_total else None
        rpm_note = f"{rpm_mae:.1f}" if rpm_mae is not None else "—"
        sh_note  = f"{sh_agrees}/{sh_total}" if sh_total else "—"

        sh_pct_str = f"{sh_pct:.0f}%" if sh_pct is not None else "—"
        print(f"  t={t:.1f}  {total_bursts/n:>9.2f}  {total_frames/n:>10.1f}  "
              f"{rpm_note:>8}  {len(rpm_errors):>6}  {sh_note:>9}  {sh_pct_str}")

        csv_rows.append({
            "fraction":             t,
            "mean_bursts_per_video": round(total_bursts / n, 2),
            "mean_frames_per_video": round(total_frames / n, 1),
            "rpm_mae":              round(rpm_mae, 2) if rpm_mae is not None else "",
            "rpm_n":                len(rpm_errors),
            "seat_height_agree":    sh_agrees,
            "seat_height_total":    sh_total,
            "seat_height_agree_pct": round(sh_pct, 1) if sh_pct is not None else "",
        })

    csv_path = os.path.join(RESULTS_DIR, "quality_fraction_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fraction", "mean_bursts_per_video",
                                          "mean_frames_per_video", "rpm_mae", "rpm_n",
                                          "seat_height_agree", "seat_height_total",
                                          "seat_height_agree_pct"])
        w.writeheader()
        w.writerows(csv_rows)

    prod_row = next(r for r in csv_rows if r["fraction"] == 0.5)
    print()
    print("── Interpretation ───────────────────────────────────────────────────────────────")
    print(f"  NOTE: This is a LIMITED upward sweep only. Lower thresholds (< 0.5) cannot")
    print(f"  be evaluated without re-running the YOLO side-angle detection stage.")
    print()
    print(f"  At t=0.5 (production): mean {prod_row['mean_bursts_per_video']:.2f} bursts/video, "
          f"mean {prod_row['mean_frames_per_video']:.1f} frames/video")
    if prod_row["rpm_mae"]:
        print(f"  RPM MAE at t=0.5: {prod_row['rpm_mae']} RPM (n={prod_row['rpm_n']})")

    # Find where quality starts dropping
    prev_sh = None
    for r in csv_rows:
        sh = r["seat_height_agree_pct"]
        if prev_sh is not None and sh != "" and prev_sh != "" and float(sh) < float(prev_sh):
            print(f"  Seat height agreement drops at t={r['fraction']:.1f}: "
                  f"{r['seat_height_agree']}/{r['seat_height_total']} "
                  f"({r['seat_height_agree_pct']}%) vs previous {prev_sh}%")
        prev_sh = sh

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
