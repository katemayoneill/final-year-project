#!/usr/bin/env python3
"""
Angle peak-selection strategy evaluation — trainer-independent approaches only.

For each subject with a real-world (b) video in output_v2/, compares multiple
peak-selection strategies against the Pipeline V2 baseline, using only data
available from the b video itself (no trainer/a data required).

Strategies evaluated:
  current_v2   — peak in assessment.json (mean of validated knee_analysis peaks,
                 or max() fallback)
  smooth_max   — max of the Savitzky-Golay smoothed angle series from
                 knee_analysis.json (removes single-frame noise spikes)
  smooth_pN    — Nth percentile of the smoothed series (N = 80, 85, 90, 92, 95)
  raw_pN       — Nth percentile of the raw per-frame angle series

Verdict agreement is computed against the a-condition verdict for the same
subject/group (used only as a reference label — not fed into any strategy).

Usage:
  python3 evaluation/angle_ceiling_eval.py [--v2 output_v2/]
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np


OPTIMAL_LOW  = 145.0
OPTIMAL_HIGH = 155.0
PERCENTILES  = [80, 85, 90, 92, 95]
CONF_MIN     = 0.1

_JOINT = {"RHip": 9, "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14}


def verdict(deg):
    if deg is None:
        return "—"
    return "optimal" if OPTIMAL_LOW <= deg <= OPTIMAL_HIGH else ("too_low" if deg < OPTIMAL_LOW else "too_high")


def sv(deg):
    """Short verdict label."""
    return {"too_low": "low", "optimal": "OPT", "too_high": "HIGH", "—": "—"}.get(verdict(deg), "—")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_stems(v2_dir):
    return [d for d in os.listdir(v2_dir) if os.path.isdir(os.path.join(v2_dir, d))]


def parse_stem(stem):
    m = re.match(r"^([a-z]+)(a|b)(\d+)a?$", stem)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def load_data(v2_dir, stem):
    """Load assessment, knee_analysis, and keypoints for a stem. Returns None if assessment or knee_analysis missing."""
    base = os.path.join(v2_dir, stem)
    ap  = os.path.join(base, f"{stem}_assessment.json")
    kap = os.path.join(base, f"{stem}_knee_analysis.json")
    kpp = os.path.join(base, f"{stem}_keypoints.json")
    if not os.path.exists(ap) or not os.path.exists(kap):
        return None
    kp_data = load_json(kpp) if os.path.exists(kpp) else None
    return load_json(ap), load_json(kap), kp_data


def _get_xy(kp, idx):
    if idx < len(kp) and kp[idx][2] >= CONF_MIN:
        return (kp[idx][0], kp[idx][1])
    return None


def _calc_angle(a, b, c):
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    mag = math.sqrt(ax**2 + ay**2) * math.sqrt(cx**2 + cy**2)
    if mag == 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (ax*cx + ay*cy) / mag))))


def get_raw_angles(kp_data, knee):
    """Per-frame raw angle series for the given knee computed from keypoints.json."""
    if kp_data is None:
        return np.array([], dtype=float)
    side   = "R" if knee == "right" else "L"
    hi, ki, ai = _JOINT[side + "Hip"], _JOINT[side + "Knee"], _JOINT[side + "Ankle"]
    angles = []
    for frame in kp_data["frames"]:
        kp = frame.get("keypoints", [])
        h, k, a = _get_xy(kp, hi), _get_xy(kp, ki), _get_xy(kp, ai)
        if h and k and a:
            ang = _calc_angle(h, k, a)
            if ang is not None:
                angles.append(ang)
    return np.array(angles, dtype=float)


def get_smoothed_angles(knee_data):
    """
    Concatenated Savitzky-Golay smoothed angle values from all runs in
    knee_analysis.json. Each run's angle_series is [[timestamp, angle], ...].
    """
    angles = []
    for run in knee_data.get("runs", []):
        for ts, ang in run.get("angle_series", []):
            angles.append(ang)
    return np.array(angles, dtype=float) if angles else np.array([], dtype=float)


def compute_strategies(assess, knee_data, kp_data=None):
    """
    Return a dict of peak values for every strategy, keyed by strategy name.
    """
    knee     = knee_data.get("knee_used", "right")
    raw      = get_raw_angles(kp_data, knee)
    smoothed = get_smoothed_angles(knee_data)
    current  = assess["summary"].get("knee_angle_peak")

    results = {"current_v2": current}

    if len(smoothed) > 0:
        results["smooth_max"] = float(np.max(smoothed))
        for p in PERCENTILES:
            results[f"smooth_p{p}"] = float(np.percentile(smoothed, p))

    if len(raw) > 0:
        for p in PERCENTILES:
            results[f"raw_p{p}"] = float(np.percentile(raw, p))

    return results


def run(v2_dir):
    stems = find_stems(v2_dir)

    groups = defaultdict(lambda: defaultdict(list))
    for stem in stems:
        parsed = parse_stem(stem)
        if parsed:
            name, cond, grp = parsed
            groups[(name, grp)][cond].append(stem)

    pairs = {k: v for k, v in groups.items() if v.get("a") and v.get("b")}

    # Collect all strategy names (in order) from first valid pair
    strategy_order = None
    rows = []

    for (name, grp), conds in sorted(pairs.items()):
        # Reference verdict from a condition (not used by any strategy)
        a_ref_verdict = "—"
        for a_stem in sorted(conds["a"]):
            data = load_data(v2_dir, a_stem)
            if data:
                peak = data[0]["summary"].get("knee_angle_peak")  # data[0] is assess
                if peak is not None:
                    a_ref_verdict = verdict(peak)
                    break

        for b_stem in sorted(conds["b"]):
            data = load_data(v2_dir, b_stem)
            if data is None:
                continue
            assess, knee_data, kp_data = data
            strategies = compute_strategies(assess, knee_data, kp_data)

            if strategy_order is None:
                strategy_order = list(strategies.keys())

            rows.append({
                "label":      name + grp,
                "a_ref":      a_ref_verdict,
                "a_verdict":  a_ref_verdict,
                "strategies": strategies,
            })

    if not rows or strategy_order is None:
        print("No paired a/b data found.")
        return

    # ── Per-subject table ────────────────────────────────────────────────────
    col_w = 7
    strat_labels = strategy_order
    header_strats = "  ".join(f"{s:>{col_w}}" for s in strat_labels)
    print(f"\n── Per-subject peak angles (°) ─────────────────────────────────────────────────")
    print(f"{'Subject':<12}  {'a_ref':<10}  {header_strats}")
    print("-" * (14 + 12 + len(strat_labels) * (col_w + 2)))

    for r in rows:
        vals = "  ".join(
            f"{r['strategies'].get(s, None):>{col_w}.1f}" if r['strategies'].get(s) is not None else f"{'—':>{col_w}}"
            for s in strat_labels
        )
        print(f"{r['label']:<12}  {r['a_ref']:<10}  {vals}")

    # ── Verdicts table ───────────────────────────────────────────────────────
    print(f"\n── Per-subject verdicts ─────────────────────────────────────────────────────────")
    header_strats_v = "  ".join(f"{s:>10}" for s in strat_labels)
    print(f"{'Subject':<12}  {'a_ref':<10}  {header_strats_v}")
    print("-" * (14 + 12 + len(strat_labels) * 12))

    for r in rows:
        verdicts = "  ".join(
            f"{sv(r['strategies'].get(s)):>10}" for s in strat_labels
        )
        print(f"{r['label']:<12}  {r['a_ref']:<10}  {verdicts}")

    # ── Agreement summary ────────────────────────────────────────────────────
    valid = [r for r in rows if r["a_verdict"] != "—"]
    print(f"\n── Verdict agreement vs a-condition reference ({len(valid)} pairs) ──────────────────")
    print(f"  {'Strategy':<20}  {'Agree':>7}  {'%':>6}  notes")
    print(f"  {'-'*60}")

    for s in strat_labels:
        eligible = [r for r in valid if r["strategies"].get(s) is not None]
        agreed   = sum(1 for r in eligible if verdict(r["strategies"][s]) == r["a_verdict"])
        pct      = 100 * agreed / len(eligible) if eligible else 0
        note = ""
        if s == "current_v2":
            note = "← baseline"
        print(f"  {s:<20}  {agreed:>4}/{len(eligible):<2}  {pct:>5.0f}%  {note}")

    # ── Focus: too_high corrections ──────────────────────────────────────────
    high_rows = [r for r in valid if verdict(r["strategies"].get("current_v2")) == "too_high"]
    print(f"\n── Inflated too_high cases (current_v2=HIGH, n={len(high_rows)}) ────────────────────")
    print(f"  Showing how many are corrected by each strategy:\n")
    print(f"  {'Strategy':<20}  {'Corrected':>10}  {'Still HIGH':>10}")
    print(f"  {'-'*45}")

    for s in strat_labels:
        if s == "current_v2":
            continue
        corrected = sum(
            1 for r in high_rows
            if r["strategies"].get(s) is not None and verdict(r["strategies"][s]) != "too_high"
        )
        still_high = len(high_rows) - corrected
        print(f"  {s:<20}  {corrected:>10}  {still_high:>10}")

    # ── Best single strategy ─────────────────────────────────────────────────
    print(f"\n── Best trainer-independent strategy ───────────────────────────────────────────")
    best_s, best_agree, best_pct = None, 0, 0
    for s in strat_labels:
        if s == "current_v2":
            continue
        eligible = [r for r in valid if r["strategies"].get(s) is not None]
        agreed   = sum(1 for r in eligible if verdict(r["strategies"][s]) == r["a_verdict"])
        pct      = 100 * agreed / len(eligible) if eligible else 0
        if pct > best_pct:
            best_s, best_agree, best_pct = s, agreed, pct

    n_eligible = sum(1 for r in valid if r["strategies"].get(best_s) is not None)
    print(f"  Best: {best_s}  →  {best_agree}/{n_eligible} ({best_pct:.0f}%)")
    print(f"  Baseline (current_v2): {sum(1 for r in valid if verdict(r['strategies'].get('current_v2')) == r['a_verdict'])}/{len(valid)} (32%)")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", default="output_v2", help="Path to output_v2 directory")
    args = parser.parse_args()
    if not os.path.isdir(args.v2):
        print(f"ERROR: {args.v2} not found", file=sys.stderr)
        sys.exit(1)
    run(args.v2)


if __name__ == "__main__":
    main()
