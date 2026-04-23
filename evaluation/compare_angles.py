#!/usr/bin/env python3
"""
Cross-video angle consistency comparison.

Loads *_assessment.json files from both pipeline output directories and
prints per-video angle stats grouped by subject.  Missing files are skipped
silently so the script works even when only one pipeline has been run.

Usage:
  python3 evaluation/compare_angles.py
  python3 evaluation/compare_angles.py --p1 output/ --p2 output_v2/
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_stem(stem):
    """Return (subject, condition, cadence_group) or (stem, '?', '?') on failure."""
    m = re.match(r"^([a-zA-Z]+)(a|b)(\d+)$", stem)
    if not m:
        return stem, "?", "?"
    return m.group(1), m.group(2), m.group(3)


def load_assessments(root, pipeline_label):
    """Scan root recursively for *_assessment.json; return list of row dicts."""
    rows = []
    for path in sorted(Path(root).rglob("*_assessment.json")):
        stem = path.stem.replace("_assessment", "")
        subject, cond, group = parse_stem(stem)
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warn] could not read {path}: {e}", file=sys.stderr)
            continue
        s = d.get("summary", {})
        rows.append({
            "stem":     stem,
            "pipeline": pipeline_label,
            "subject":  subject,
            "cond":     cond,
            "group":    group,
            "peak":     s.get("knee_angle_peak"),
            "mean":     s.get("knee_angle_mean"),
            "std":      s.get("knee_angle_std"),
            "n":        s.get("knee_angles_count"),
            "verdict":  s.get("verdict", "—"),
        })
    return rows


def fmt(v, dec=1):
    return f"{v:.{dec}f}" if v is not None else "—"


def print_full_table(rows):
    col_w = [10, 14, 4, 5, 4, 7, 7, 6, 5, 10]
    headers = ["Subject", "Video", "Pipe", "Cond", "Grp", "Peak°", "Mean°", "Std", "N", "Verdict"]
    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))

    print("\n── Per-video angle measurements ────────────────────────────────────────────")
    print(hdr)
    print(sep)

    subjects = sorted(set(r["subject"] for r in rows))
    for subj in subjects:
        subj_rows = sorted(
            [r for r in rows if r["subject"] == subj],
            key=lambda r: (r["cond"], r["group"], r["pipeline"]),
        )
        for r in subj_rows:
            cols = [
                r["subject"].ljust(col_w[0]),
                r["stem"].ljust(col_w[1]),
                r["pipeline"].ljust(col_w[2]),
                r["cond"].ljust(col_w[3]),
                str(r["group"]).ljust(col_w[4]),
                fmt(r["peak"]).rjust(col_w[5]),
                fmt(r["mean"]).rjust(col_w[6]),
                fmt(r["std"]).rjust(col_w[7]),
                str(r["n"] if r["n"] is not None else "—").rjust(col_w[8]),
                r["verdict"].ljust(col_w[9]),
            ]
            print("  ".join(cols))
        print()


def print_consistency_table(rows, pipelines):
    """Per-subject summary: peak angle range across all videos for each pipeline."""
    print("── Within-subject consistency (peak angle range) ───────────────────────────")

    pipe_labels = sorted(pipelines)
    col_w = [10] + [22] * len(pipe_labels)
    headers = ["Subject"] + [f"Pipeline {p}" for p in pipe_labels]
    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(hdr)
    print(sep)

    subjects = sorted(set(r["subject"] for r in rows))
    for subj in subjects:
        cols = [subj.ljust(col_w[0])]
        for pipe in pipe_labels:
            peaks = [
                r["peak"] for r in rows
                if r["subject"] == subj and r["pipeline"] == pipe and r["peak"] is not None
            ]
            if not peaks:
                cols.append("—".ljust(col_w[1]))
            else:
                lo, hi = min(peaks), max(peaks)
                span = hi - lo
                cols.append(
                    f"{fmt(lo)}–{fmt(hi)}°  (range {fmt(span)}°)".ljust(col_w[1])
                )
        print("  ".join(cols))
    print()


def print_pipeline_diff(rows, p1_label, p2_label):
    """For stems present in both pipelines, show the peak angle difference."""
    p1 = {r["stem"]: r for r in rows if r["pipeline"] == p1_label}
    p2 = {r["stem"]: r for r in rows if r["pipeline"] == p2_label}
    common = sorted(set(p1) & set(p2))
    if not common:
        print(f"── Pipeline diff ({p1_label} vs {p2_label}) ─────────────────────────────────────")
        print("  No stems found in both pipelines yet.")
        return

    col_w = [14, 8, 8, 7, 8, 8]
    headers = ["Video", f"Peak {p1_label}", f"Peak {p2_label}", "Δ peak", f"Verd {p1_label}", f"Verd {p2_label}"]
    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))

    print(f"\n── Pipeline diff ({p1_label} vs {p2_label}) ─────────────────────────────────────")
    print(hdr)
    print(sep)

    diffs = []
    for stem in common:
        r1, r2 = p1[stem], p2[stem]
        peak1, peak2 = r1["peak"], r2["peak"]
        delta = (peak2 - peak1) if (peak1 is not None and peak2 is not None) else None
        if delta is not None:
            diffs.append(abs(delta))
        cols = [
            stem.ljust(col_w[0]),
            fmt(peak1).rjust(col_w[1]),
            fmt(peak2).rjust(col_w[2]),
            (("+" if delta >= 0 else "") + fmt(delta) if delta is not None else "—").rjust(col_w[3]),
            r1["verdict"].ljust(col_w[4]),
            r2["verdict"].ljust(col_w[5]),
        ]
        print("  ".join(cols))

    if diffs:
        print(f"\n  Δ peak: mean={fmt(sum(diffs)/len(diffs))}°  max={fmt(max(diffs))}°  n={len(diffs)}")


def main():
    parser = argparse.ArgumentParser(description="Compare angle measurements across both pipelines")
    parser.add_argument("--p1", default="output",    help="Pipeline 1 output root (default: output/)")
    parser.add_argument("--p2", default="output_v2", help="Pipeline V2 output root (default: output_v2/)")
    args = parser.parse_args()

    rows = []
    for root, label in [(args.p1, "p1"), (args.p2, "p2")]:
        p = Path(root)
        if not p.is_dir():
            print(f"  [skip] directory not found: {root}", file=sys.stderr)
            continue
        found = load_assessments(root, label)
        if not found:
            print(f"  [skip] no *_assessment.json files in {root}", file=sys.stderr)
        rows.extend(found)

    if not rows:
        sys.exit("No assessment files found in either pipeline output directory.")

    pipelines = set(r["pipeline"] for r in rows)

    print_full_table(rows)
    print_consistency_table(rows, pipelines)
    if "p1" in pipelines and "p2" in pipelines:
        print_pipeline_diff(rows, "p1", "p2")
    elif len(pipelines) == 1:
        only = next(iter(pipelines))
        print(f"  (Only {only} data available — pipeline diff skipped)")


if __name__ == "__main__":
    main()
