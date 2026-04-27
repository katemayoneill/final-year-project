#!/usr/bin/env python3
"""
Tier 1 sensitivity sweep: PEAK_MEAN_MIN_PEAKS threshold.

Currently seat_height.py uses PEAK_MEAN_MIN_PEAKS = 10 to decide whether to
compute the peak angle as:
  peak_mean  — mean of all validated peaks from runs with >= 2 peaks
  smooth_p80 — 80th percentile of the SG-smoothed angle series

This sweep varies that threshold over [3, 5, 8, 10, 12, 15, 20] and reports:
  - How many b-videos and a-videos fall on each side of the threshold
  - Verdict agreement (b-condition vs a-condition reference) at each threshold
  - Whether the threshold cleanly separates trainer vs real-world conditions

The a-condition verdict is used as a within-subject reference only — not ground truth.

Output:
  evaluation/sensitivity/results/peak_threshold_sweep.csv
  Prints a summary table to stdout.

Usage (from project root):
  python3 evaluation/sensitivity/sweep_peak_threshold.py [--v2 output_v2/]
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

OPTIMAL_LOW  = 145.0
OPTIMAL_HIGH = 155.0
SMOOTH_PCT   = 80       # matches production seat_height.py
THRESHOLDS   = [3, 5, 8, 10, 12, 15, 20]

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


def count_usable_peaks(ka):
    """Count validated peaks from runs with >= 2 peaks (matches seat_height.py)."""
    return sum(
        len(run["peaks"])
        for run in ka.get("runs", [])
        if len(run.get("peaks", [])) >= 2
    )


def get_peak_mean(ka):
    """Mean angle of validated peaks from runs with >= 2 peaks."""
    peaks = [
        p["angle"]
        for run in ka.get("runs", [])
        if len(run.get("peaks", [])) >= 2
        for p in run["peaks"]
    ]
    return sum(peaks) / len(peaks) if peaks else None


def get_smooth_p80(ka):
    """80th percentile of SG-smoothed series from all runs."""
    angles = [ang for run in ka.get("runs", []) for _, ang in run.get("angle_series", [])]
    return float(np.percentile(angles, SMOOTH_PCT)) if angles else None


def find_data(v2_dir):
    """
    Returns:
      b_records — list of dicts per b-video:
        {stem, subject, peak_count, peak_mean, smooth_p80, a_verdict}
      a_records — list of dicts per a-video:
        {stem, subject, peak_count}
    """
    stems = [d for d in os.listdir(v2_dir) if os.path.isdir(os.path.join(v2_dir, d))]

    from collections import defaultdict
    by_subj_grp = defaultdict(lambda: {"a": [], "b": []})
    for stem in stems:
        parsed = parse_stem(stem)
        if parsed:
            subj, cond, grp = parsed
            by_subj_grp[(subj, grp)][cond].append(stem)

    b_records = []
    a_records = []

    for (subj, grp), conds in sorted(by_subj_grp.items()):
        # a-condition records
        for a_stem in sorted(conds.get("a", [])):
            ka_path = os.path.join(v2_dir, a_stem, f"{a_stem}_knee_analysis.json")
            if not os.path.exists(ka_path):
                continue
            try:
                ka = load_json(ka_path)
                a_records.append({"stem": a_stem, "subject": subj, "peak_count": count_usable_peaks(ka)})
            except (json.JSONDecodeError, KeyError):
                continue

        if not conds.get("a") or not conds.get("b"):
            continue

        # a-condition reference verdict
        a_ref_verdict = None
        for a_stem in sorted(conds["a"]):
            ap = os.path.join(v2_dir, a_stem, f"{a_stem}_assessment.json")
            if not os.path.exists(ap):
                continue
            try:
                a_assess = load_json(ap)
                peak = a_assess["summary"].get("knee_angle_peak")
                if peak is not None:
                    a_ref_verdict = verdict(peak)
                    break
            except (KeyError, json.JSONDecodeError):
                continue

        if a_ref_verdict is None:
            continue

        # b-condition records
        for b_stem in sorted(conds.get("b", [])):
            ka_path = os.path.join(v2_dir, b_stem, f"{b_stem}_knee_analysis.json")
            if not os.path.exists(ka_path):
                continue
            try:
                ka = load_json(ka_path)
            except json.JSONDecodeError:
                continue

            pc  = count_usable_peaks(ka)
            pm  = get_peak_mean(ka)
            sp  = get_smooth_p80(ka)

            b_records.append({
                "stem":       b_stem,
                "subject":    subj,
                "peak_count": pc,
                "peak_mean":  pm,
                "smooth_p80": sp,
                "a_verdict":  a_ref_verdict,
            })

    return b_records, a_records


def run(v2_dir):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    b_records, a_records = find_data(v2_dir)

    if not b_records:
        print("ERROR: no b-condition data found.", file=sys.stderr)
        sys.exit(1)

    total_b = len(b_records)
    total_a = len(a_records)

    # Print per-video peak counts for reference
    print(f"\nPeak-mean threshold sweep — PEAK_MEAN_MIN_PEAKS sensitivity")
    print(f"B-videos: {total_b}   A-videos: {total_a}")
    print()
    print("Per-video validated peak counts (b-condition):")
    for r in sorted(b_records, key=lambda x: -x["peak_count"]):
        method = "peak_mean" if r["peak_count"] >= 10 else "smooth_p80"
        print(f"  {r['stem']:<20}  peaks={r['peak_count']:>3}  [{method} at t=10]")

    print()
    print("Per-video validated peak counts (a-condition):")
    for r in sorted(a_records, key=lambda x: -x["peak_count"]):
        print(f"  {r['stem']:<20}  peaks={r['peak_count']:>3}")

    print()
    hdr = (f"{'thresh':>6}  {'b:peak_mean':>11}  {'b:smooth_p80':>12}  "
           f"{'a:peak_mean':>11}  {'a:smooth_p80':>12}  {'agree':>8}  {'agree%':>7}")
    print(hdr)
    print("-" * len(hdr))

    csv_rows = []
    for t in THRESHOLDS:
        b_pm   = sum(1 for r in b_records if r["peak_count"] >= t)
        b_sp80 = total_b - b_pm
        a_pm   = sum(1 for r in a_records if r["peak_count"] >= t)
        a_sp80 = total_a - a_pm

        agrees = 0
        for r in b_records:
            if r["peak_count"] >= t:
                peak = r["peak_mean"]
            else:
                peak = r["smooth_p80"]
            if verdict(peak) == r["a_verdict"]:
                agrees += 1

        agree_pct = 100 * agrees / total_b if total_b else 0
        print(f"  t={t:<3}  {b_pm:>6}/{total_b:<2}  {b_sp80:>7}/{total_b:<2}  "
              f"{a_pm:>6}/{total_a:<2}  {a_sp80:>7}/{total_a:<2}  "
              f"{agrees:>4}/{total_b:<2}  {agree_pct:>5.0f}%")

        csv_rows.append({
            "threshold":        t,
            "b_using_peak_mean": b_pm,
            "b_using_smooth_p80": b_sp80,
            "a_using_peak_mean": a_pm,
            "a_using_smooth_p80": a_sp80,
            "total_agreement":  agrees,
            "agreement_pct":    round(agree_pct, 1),
        })

    csv_path = os.path.join(RESULTS_DIR, "peak_threshold_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "b_using_peak_mean",
                                          "b_using_smooth_p80", "a_using_peak_mean",
                                          "a_using_smooth_p80", "total_agreement",
                                          "agreement_pct"])
        w.writeheader()
        w.writerows(csv_rows)

    print()
    print("── Interpretation ───────────────────────────────────────────────────────────────")
    best = max(csv_rows, key=lambda r: r["total_agreement"])
    prod = next(r for r in csv_rows if r["threshold"] == 10)

    # Separation check: does any threshold cleanly separate a from b?
    print("  Threshold separation (does it put all a-videos on peak_mean side, "
          "all b-videos on smooth_p80 side?):")
    for r in csv_rows:
        t = r["threshold"]
        a_all_pm  = r["a_using_peak_mean"]  == total_a
        b_all_sp  = r["b_using_smooth_p80"] == total_b
        clean = "CLEAN SEPARATION" if (a_all_pm and b_all_sp) else ""
        note  = f"(a: {r['a_using_peak_mean']}/{total_a} on peak_mean, b: {r['b_using_smooth_p80']}/{total_b} on smooth_p80)"
        print(f"    t={t:<3}  {note}  {clean}")

    print()
    print(f"  Best agreement      : t={best['threshold']}  ({best['total_agreement']}/{total_b}, "
          f"{best['agreement_pct']:.0f}%)")
    print(f"  Production value    : t=10  ({prod['total_agreement']}/{total_b}, "
          f"{prod['agreement_pct']:.0f}%)")
    if prod["total_agreement"] == best["total_agreement"]:
        print(f"  Verdict             : t=10 achieves maximum agreement — CURRENT VALUE JUSTIFIED.")
    elif best["total_agreement"] - prod["total_agreement"] <= 1:
        print(f"  Verdict             : t=10 within 1 of optimum — CURRENT VALUE JUSTIFIED.")
    else:
        print(f"  Verdict             : t={best['threshold']} outperforms t=10 — CONSIDER CHANGING.")
    print(f"\n  CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", default="output_v2", help="Path to output_v2 directory")
    args = parser.parse_args()
    if not os.path.isdir(args.v2):
        print(f"ERROR: {args.v2!r} not found — run from project root.", file=sys.stderr)
        sys.exit(1)
    run(args.v2)


if __name__ == "__main__":
    main()
