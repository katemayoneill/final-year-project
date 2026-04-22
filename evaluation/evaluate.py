#!/usr/bin/env python3
"""
Evaluation script for the cycling posture analysis pipeline.

RPM evaluation:
  Compares pipeline cadence_rpm (from *_rpm.json) against Kinovea ground truth
  in ground_truth.csv. Reports MAE, RMSE, mean % error broken down by:
    - condition (a = trainer/controlled, b = real-world pass-by)
    - framerate (30 vs 60 fps)
    - RPM method (peak_detection vs autocorrelation)

Seat height evaluation:
  For each subject with both an 'a' and 'b' video at the same fps, reports
  whether the pipeline verdict (too_low / optimal / too_high) agrees between
  the controlled and real-world conditions.

Usage:
  python3 evaluation/evaluate.py [--gt ground_truth.csv] [--videos output/]
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path


def find_json(stem, suffix, videos_root):
    """Search videos_root recursively for <stem><suffix>.json."""
    for path in Path(videos_root).rglob(f"{stem}{suffix}.json"):
        return path
    return None


def parse_condition(stem):
    """
    Return (subject, condition, cadence_group) from a video stem like 'alexa30' or 'janeb60'.
    condition: 'a' or 'b'
    cadence_group: '30' (target 60 RPM) or '60' (target 90 RPM)
    """
    m = re.match(r"^([a-zA-Z]+)(a|b)(\d+)", stem)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def load_ground_truth(csv_path):
    """Returns {stem: true_rpm} for rows where true_rpm is filled in."""
    gt = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            stem = row["video"].strip()
            val  = row["true_rpm"].strip()
            if stem and val:
                try:
                    gt[stem] = float(val)
                except ValueError:
                    print(f"Warning: non-numeric true_rpm for {stem!r}: {val!r}", file=sys.stderr)
    return gt


def load_rpm_json(path):
    with open(path) as f:
        return json.load(f)


def load_assessment_json(path):
    with open(path) as f:
        return json.load(f)


# ── helpers ──────────────────────────────────────────────────────────────────

def mae(errors):
    return sum(abs(e) for e in errors) / len(errors)

def rmse(errors):
    return math.sqrt(sum(e ** 2 for e in errors) / len(errors))

def mean_pct_error(pct_errors):
    return sum(abs(e) for e in pct_errors) / len(pct_errors)

def fmt(v, decimals=1):
    return f"{v:.{decimals}f}" if v is not None else "—"


# ── RPM evaluation ────────────────────────────────────────────────────────────

def evaluate_rpm(gt, videos_root):
    rows = []

    for stem, true_rpm in sorted(gt.items()):
        json_path = find_json(stem, "_rpm", videos_root)
        if json_path is None:
            print(f"  [skip] {stem}_rpm.json not found", file=sys.stderr)
            continue

        data = load_rpm_json(json_path)
        pred_rpm = data.get("cadence_rpm")
        method   = data.get("rpm_method", "unknown")

        subject, condition, cadence_group = parse_condition(stem)
        if subject is None:
            subject, condition, cadence_group = stem, "?", "?"

        if pred_rpm is None:
            rows.append({
                "stem": stem, "subject": subject, "condition": condition,
                "cadence_group": cadence_group, "true_rpm": true_rpm, "pred_rpm": None,
                "error": None, "pct_error": None, "method": method,
            })
            continue

        error     = pred_rpm - true_rpm
        pct_error = (error / true_rpm) * 100

        rows.append({
            "stem": stem, "subject": subject, "condition": condition,
            "cadence_group": cadence_group, "true_rpm": true_rpm, "pred_rpm": pred_rpm,
            "error": error, "pct_error": pct_error, "method": method,
        })

    return rows


def print_rpm_table(rows):
    valid = [r for r in rows if r["error"] is not None]
    failed = [r for r in rows if r["pred_rpm"] is None]

    col_w = [10, 5, 5, 8, 8, 7, 7, 18]
    headers = ["Video", "Cond", "Grp", "True RPM", "Pred RPM", "Error", "% Err", "Method"]

    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))

    print("\n── RPM Evaluation ──────────────────────────────────────────────────────")
    print(hdr)
    print(sep)

    for r in rows:
        cols = [
            r["stem"].ljust(col_w[0]),
            r["condition"].ljust(col_w[1]),
            str(r["cadence_group"]).ljust(col_w[2]),
            fmt(r["true_rpm"]).rjust(col_w[3]),
            fmt(r["pred_rpm"]).rjust(col_w[4]),
            (fmt(r["error"], 1) if r["error"] is not None else "—").rjust(col_w[5]),
            (fmt(r["pct_error"], 1) + "%" if r["pct_error"] is not None else "—").rjust(col_w[6]),
            str(r["method"]).ljust(col_w[7]),
        ]
        print("  ".join(cols))

    print()

    def group_stats(label, subset):
        errs     = [r["error"]     for r in subset if r["error"] is not None]
        pct_errs = [r["pct_error"] for r in subset if r["pct_error"] is not None]
        n = len(errs)
        if n == 0:
            print(f"  {label:30s}  n=0  (no data)")
            return
        print(f"  {label:30s}  n={n:<3d}  MAE={fmt(mae(errs))} RPM  "
              f"RMSE={fmt(rmse(errs))} RPM  mean |%err|={fmt(mean_pct_error(pct_errs))}%")

    print("── Aggregate statistics ────────────────────────────────────────────────")
    group_stats("All",                   valid)
    group_stats("Condition a (trainer)", [r for r in valid if r["condition"] == "a"])
    group_stats("Condition b (real-world)", [r for r in valid if r["condition"] == "b"])
    group_stats("Group 30 (target: 60 RPM)", [r for r in valid if r["cadence_group"] == "30"])
    group_stats("Group 60 (target: 90 RPM)", [r for r in valid if r["cadence_group"] == "60"])
    group_stats("Method: peak_detection",[r for r in valid if r["method"] == "peak_detection"])
    group_stats("Method: autocorrelation",[r for r in valid if r["method"] == "autocorrelation"])

    if failed:
        print(f"\n  No RPM output for: {', '.join(r['stem'] for r in failed)}")


# ── Seat height evaluation ────────────────────────────────────────────────────

def evaluate_seat_height(videos_root):
    """
    For each subject+fps pair that has both an 'a' and 'b' assessment JSON,
    compare verdicts and peak knee angles.
    """
    assessment_paths = list(Path(videos_root).rglob("*_assessment.json"))

    # Index by stem
    by_stem = {}
    for p in assessment_paths:
        stem = p.stem.replace("_assessment", "")
        by_stem[stem] = p

    # Find pairs: same subject + fps, one 'a' one 'b'
    pairs = []
    for stem_a, path_a in sorted(by_stem.items()):
        subj, cond, fps = parse_condition(stem_a)
        if cond != "a":
            continue
        stem_b = f"{subj}b{fps}"
        if stem_b in by_stem:
            pairs.append((stem_a, path_a, stem_b, by_stem[stem_b]))

    if not pairs:
        print("\n── Seat Height Evaluation ──────────────────────────────────────────────")
        print("  No paired a/b assessment JSONs found.")
        return

    print("\n── Seat Height Evaluation ──────────────────────────────────────────────")
    headers = ["Subject", "Grp", "Verdict (a)", "Peak° (a)", "Verdict (b)", "Peak° (b)", "Match"]
    col_w   = [10, 4, 12, 10, 12, 10, 5]
    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(hdr)
    print(sep)

    matches = 0
    for stem_a, path_a, stem_b, path_b in pairs:
        subj, _, cadence_group = parse_condition(stem_a)
        da = load_assessment_json(path_a)
        db = load_assessment_json(path_b)

        verdict_a = da.get("summary", {}).get("verdict", "—")
        verdict_b = db.get("summary", {}).get("verdict", "—")
        peak_a    = da.get("summary", {}).get("knee_angle_peak")
        peak_b    = db.get("summary", {}).get("knee_angle_peak")
        match     = "Y" if verdict_a == verdict_b else "N"
        if verdict_a == verdict_b:
            matches += 1

        cols = [
            subj.ljust(col_w[0]),
            str(cadence_group).ljust(col_w[1]),
            verdict_a.ljust(col_w[2]),
            fmt(peak_a, 1).rjust(col_w[3]),
            verdict_b.ljust(col_w[4]),
            fmt(peak_b, 1).rjust(col_w[5]),
            match.ljust(col_w[6]),
        ]
        print("  ".join(cols))

    pct = 100 * matches / len(pairs)
    print(f"\n  Verdict agreement: {matches}/{len(pairs)} ({pct:.0f}%)")

    # Peak angle delta
    deltas = []
    for _, path_a, _, path_b in pairs:
        da = load_assessment_json(path_a)
        db = load_assessment_json(path_b)
        pa = da.get("summary", {}).get("knee_angle_peak")
        pb = db.get("summary", {}).get("knee_angle_peak")
        if pa is not None and pb is not None:
            deltas.append(abs(pa - pb))
    if deltas:
        print(f"  Peak angle delta (a vs b): mean={fmt(sum(deltas)/len(deltas))}°  "
              f"max={fmt(max(deltas))}°")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline outputs against ground truth")
    parser.add_argument("--gt",     default="evaluation/ground_truth.csv",
                        help="Path to ground_truth.csv (default: evaluation/ground_truth.csv)")
    parser.add_argument("--videos", default="output/",
                        help="Root directory containing pipeline output folders (default: output/)")
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        sys.exit(f"Ground truth CSV not found: {args.gt}")
    if not os.path.isdir(args.videos):
        sys.exit(f"Videos directory not found: {args.videos}")

    gt = load_ground_truth(args.gt)
    if not gt:
        sys.exit("No ground truth rows with RPM values found. Fill in ground_truth.csv first.")

    rpm_rows = evaluate_rpm(gt, args.videos)
    print_rpm_table(rpm_rows)
    evaluate_seat_height(args.videos)


if __name__ == "__main__":
    main()
