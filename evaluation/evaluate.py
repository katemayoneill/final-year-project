#!/usr/bin/env python3
"""
Evaluation script for the cycling posture analysis pipeline.

RPM evaluation:
  Compares pipeline cadence_rpm (from *_rpm.json) against Kinovea ground truth
  in ground_truth.csv. Reports MAE, RMSE, mean % error broken down by:
    - condition (a = trainer/controlled, b = real-world pass-by)
    - cadence group (30 vs 60)
    - RPM method (peak_detection vs autocorrelation)

  When both output/ and output_v2/ are present, prints a side-by-side
  comparison table with aggregate stats for each pipeline.

Seat height evaluation:
  For each subject with both an 'a' and 'b' video at the same group, reports
  whether the pipeline verdict (too_low / optimal / too_high) agrees between
  the controlled and real-world conditions.

  When both pipelines are present, each is shown in a separate labelled block.

Usage:
  python3 evaluation/evaluate.py [--gt ground_truth.csv] [--videos output/] [--videos-v2 output_v2/]
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
    """Returns list of per-video result dicts for the given output directory."""
    rows = []
    for stem, true_rpm in sorted(gt.items()):
        json_path = find_json(stem, "_rpm", videos_root)
        if json_path is None:
            print(f"  [skip] {stem}_rpm.json not found in {videos_root}", file=sys.stderr)
            continue

        data     = load_rpm_json(json_path)
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


def _aggregate_block(rows, label_width=30):
    """Return formatted aggregate stats lines for a set of RPM rows."""
    valid = [r for r in rows if r["error"] is not None]

    def group_line(label, subset):
        errs     = [r["error"]     for r in subset if r["error"] is not None]
        pct_errs = [r["pct_error"] for r in subset if r["pct_error"] is not None]
        n = len(errs)
        if n == 0:
            return f"  {label:{label_width}s}  n=0  (no data)"
        return (f"  {label:{label_width}s}  n={n:<3d}  MAE={fmt(mae(errs))} RPM  "
                f"RMSE={fmt(rmse(errs))} RPM  mean |%err|={fmt(mean_pct_error(pct_errs))}%")

    lines = [
        group_line("All",                      valid),
        group_line("Condition a (trainer)",     [r for r in valid if r["condition"] == "a"]),
        group_line("Condition b (real-world)",  [r for r in valid if r["condition"] == "b"]),
        group_line("Group 30 (target: 60 RPM)", [r for r in valid if r["cadence_group"] == "30"]),
        group_line("Group 60 (target: 90 RPM)", [r for r in valid if r["cadence_group"] == "60"]),
        group_line("Method: peak_detection",    [r for r in valid if r["method"] == "peak_detection"]),
        group_line("Method: autocorrelation",   [r for r in valid if r["method"] == "autocorrelation"]),
    ]
    return lines


def print_rpm_table(rows, label="Pipeline"):
    """Print per-video RPM results and aggregate stats for a single pipeline."""
    failed = [r for r in rows if r["pred_rpm"] is None]

    col_w   = [10, 5, 5, 8, 8, 7, 7, 18]
    headers = ["Video", "Cond", "Grp", "True RPM", "Pred RPM", "Error", "% Err", "Method"]

    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))

    print(f"\n── RPM Evaluation — {label} {'─' * max(0, 54 - len(label))}")
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
    print("── Aggregate statistics ────────────────────────────────────────────────")
    for line in _aggregate_block(rows):
        print(line)

    if failed:
        print(f"\n  No RPM output for: {', '.join(r['stem'] for r in failed)}")


def print_rpm_comparison(rows_v1, rows_v2):
    """Print side-by-side RPM results for Pipeline 1 vs Pipeline V2."""
    # Index by stem
    by_stem_v1 = {r["stem"]: r for r in rows_v1}
    by_stem_v2 = {r["stem"]: r for r in rows_v2}
    all_stems  = sorted(set(by_stem_v1) | set(by_stem_v2))

    col_w   = [10, 5, 5, 8,  8, 6, 6,  8, 6, 6]
    headers = ["Video", "Cond", "Grp",
               "True", "P1 Pred", "P1 Err", "P1 %",
               "P2 Pred", "P2 Err", "P2 %"]

    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))

    print("\n── RPM Evaluation — Pipeline 1 vs Pipeline V2 ──────────────────────────")
    print(hdr)
    print(sep)

    for stem in all_stems:
        r1 = by_stem_v1.get(stem)
        r2 = by_stem_v2.get(stem)
        base = r1 or r2
        cols = [
            stem.ljust(col_w[0]),
            base["condition"].ljust(col_w[1]),
            str(base["cadence_group"]).ljust(col_w[2]),
            fmt(base["true_rpm"]).rjust(col_w[3]),
            # Pipeline 1
            fmt(r1["pred_rpm"] if r1 else None).rjust(col_w[4]),
            (fmt(r1["error"], 1) if r1 and r1["error"] is not None else "—").rjust(col_w[5]),
            (fmt(r1["pct_error"], 1) + "%" if r1 and r1["pct_error"] is not None else "—").rjust(col_w[6]),
            # Pipeline V2
            fmt(r2["pred_rpm"] if r2 else None).rjust(col_w[7]),
            (fmt(r2["error"], 1) if r2 and r2["error"] is not None else "—").rjust(col_w[8]),
            (fmt(r2["pct_error"], 1) + "%" if r2 and r2["pct_error"] is not None else "—").rjust(col_w[9]),
        ]
        print("  ".join(cols))

    print()
    print("── Aggregate — Pipeline 1 ──────────────────────────────────────────────")
    for line in _aggregate_block(rows_v1):
        print(line)
    print()
    print("── Aggregate — Pipeline V2 ─────────────────────────────────────────────")
    for line in _aggregate_block(rows_v2):
        print(line)

    # Delta: V2 improvement over V1
    shared = [s for s in all_stems if s in by_stem_v1 and s in by_stem_v2
              and by_stem_v1[s]["error"] is not None and by_stem_v2[s]["error"] is not None]
    if shared:
        deltas = [abs(by_stem_v1[s]["error"]) - abs(by_stem_v2[s]["error"]) for s in shared]
        improved = sum(1 for d in deltas if d > 0)
        print(f"\n  V2 vs V1 on {len(shared)} shared videos: "
              f"improved={improved}, worse={sum(1 for d in deltas if d < 0)}, "
              f"same={sum(1 for d in deltas if d == 0)}")
        print(f"  Mean |error| reduction: {fmt(sum(deltas)/len(deltas))} RPM "
              f"(positive = V2 better)")

    v1_failed = [r for r in rows_v1 if r["pred_rpm"] is None]
    v2_failed = [r for r in rows_v2 if r["pred_rpm"] is None]
    if v1_failed:
        print(f"\n  P1 no output for: {', '.join(r['stem'] for r in v1_failed)}")
    if v2_failed:
        print(f"  P2 no output for: {', '.join(r['stem'] for r in v2_failed)}")


# ── Seat height evaluation ────────────────────────────────────────────────────

def _seat_height_pairs(videos_root):
    """Return list of (stem_a, path_a, stem_b, path_b) pairs from videos_root."""
    assessment_paths = list(Path(videos_root).rglob("*_assessment.json"))
    by_stem = {}
    for p in assessment_paths:
        stem = p.stem.replace("_assessment", "")
        by_stem[stem] = p

    pairs = []
    for stem_a, path_a in sorted(by_stem.items()):
        subj, cond, grp = parse_condition(stem_a)
        if cond != "a":
            continue
        stem_b = f"{subj}b{grp}"
        if stem_b in by_stem:
            pairs.append((stem_a, path_a, stem_b, by_stem[stem_b]))
    return pairs


def _print_seat_height_table(pairs, label):
    print(f"\n── Seat Height Evaluation — {label} {'─' * max(0, 44 - len(label))}")

    if not pairs:
        print("  No paired a/b assessment JSONs found.")
        return

    headers = ["Subject", "Grp", "Verdict (a)", "Peak° (a)", "Verdict (b)", "Peak° (b)", "Match"]
    col_w   = [10, 4, 12, 10, 12, 10, 5]
    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(hdr)
    print(sep)

    matches = 0
    for stem_a, path_a, _, path_b in pairs:
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


def evaluate_seat_height(videos_root, label="Pipeline"):
    pairs = _seat_height_pairs(videos_root)
    _print_seat_height_table(pairs, label)


def evaluate_seat_height_both(v1_root, v2_root):
    pairs_v1 = _seat_height_pairs(v1_root)
    pairs_v2 = _seat_height_pairs(v2_root)
    _print_seat_height_table(pairs_v1, "Pipeline 1")
    _print_seat_height_table(pairs_v2, "Pipeline V2")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline outputs against ground truth")
    parser.add_argument("--gt",        default="evaluation/ground_truth.csv",
                        help="Path to ground_truth.csv (default: evaluation/ground_truth.csv)")
    parser.add_argument("--videos",    default="output/",
                        help="Pipeline 1 output directory (default: output/)")
    parser.add_argument("--videos-v2", default="output_v2/",
                        help="Pipeline V2 output directory (default: output_v2/)")
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        sys.exit(f"Ground truth CSV not found: {args.gt}")

    has_v1 = os.path.isdir(args.videos)
    has_v2 = os.path.isdir(args.videos_v2)

    if not has_v1 and not has_v2:
        sys.exit(f"Neither output directory found: {args.videos!r}, {args.videos_v2!r}")

    gt = load_ground_truth(args.gt)
    if not gt:
        sys.exit("No ground truth rows with RPM values found. Fill in ground_truth.csv first.")

    if has_v1 and has_v2:
        rows_v1 = evaluate_rpm(gt, args.videos)
        rows_v2 = evaluate_rpm(gt, args.videos_v2)
        print_rpm_comparison(rows_v1, rows_v2)
        evaluate_seat_height_both(args.videos, args.videos_v2)
    elif has_v1:
        rows = evaluate_rpm(gt, args.videos)
        print_rpm_table(rows, label="Pipeline 1")
        evaluate_seat_height(args.videos, label="Pipeline 1")
    else:
        rows = evaluate_rpm(gt, args.videos_v2)
        print_rpm_table(rows, label="Pipeline V2")
        evaluate_seat_height(args.videos_v2, label="Pipeline V2")


if __name__ == "__main__":
    main()
