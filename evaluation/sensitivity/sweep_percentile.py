#!/usr/bin/env python3
"""
Tier 1 sensitivity sweep: percentile for smooth_pN seat-height selection.

For each percentile in PERCENTILES, recomputes the b-condition peak knee angle
as the Nth percentile of the Savitzky-Golay-smoothed angle series from
_knee_analysis.json (concatenated across all runs, exactly as seat_height.py does).
Applies the 145°/155° verdict thresholds and compares against the a-condition
verdict from _assessment.json (used as a within-subject reference, not ground truth).

Subject filming-group membership (from CLAUDE.md):
  Group 1 — jenny, kate, roman     (close-up, good lighting, single pass)
  Group 2 — hannah, alex           (straight road, further away, two passes)
  Group 3 — dervla, jack, jane, liam, paddy  (cement sports pitch, many directions)

Output:
  evaluation/sensitivity/results/percentile_sweep.csv
  Prints a summary table to stdout.

Usage (from project root):
  python3 evaluation/sensitivity/sweep_percentile.py [--v2 output_v2/]
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
PERCENTILES  = [70, 72, 75, 77, 80, 82, 85, 87, 90, 92, 95]

FILMING_GROUPS = {
    1: {"jenny", "kate", "roman"},
    2: {"hannah", "alex"},
    3: {"dervla", "jack", "jane", "liam", "paddy"},
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def verdict(deg):
    if deg is None:
        return None
    if deg < OPTIMAL_LOW:
        return "too_low"
    if deg > OPTIMAL_HIGH:
        return "too_high"
    return "optimal"


def parse_stem(stem):
    """Returns (subject, condition, group) or None."""
    m = re.match(r"^([a-z]+)(a|b)(\d+)a?$", stem)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def filming_group(subject):
    for grp, names in FILMING_GROUPS.items():
        if subject in names:
            return grp
    return None


def get_smoothed_angles(ka):
    """Concatenate smoothed angle values from all runs (matches seat_height.py logic)."""
    return [ang for run in ka.get("runs", []) for _, ang in run.get("angle_series", [])]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_pairs(v2_dir):
    """
    Returns list of dicts with keys:
      subject, group_num, grp_label, b_stem, a_stem,
      smoothed_angles (list of floats), a_verdict (str)
    """
    stems = [d for d in os.listdir(v2_dir) if os.path.isdir(os.path.join(v2_dir, d))]

    from collections import defaultdict
    by_subject_grp = defaultdict(lambda: {"a": [], "b": []})
    for stem in stems:
        parsed = parse_stem(stem)
        if parsed:
            subj, cond, grp = parsed
            by_subject_grp[(subj, grp)][cond].append(stem)

    pairs = []
    for (subj, grp), conds in sorted(by_subject_grp.items()):
        a_stems = conds.get("a", [])
        b_stems = conds.get("b", [])
        if not a_stems or not b_stems:
            continue

        # Get a-condition reference verdict (first a-stem with valid assessment)
        a_ref_verdict = None
        for a_stem in sorted(a_stems):
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

        # Get b-condition smoothed angles
        for b_stem in sorted(b_stems):
            ka_path = os.path.join(v2_dir, b_stem, f"{b_stem}_knee_analysis.json")
            if not os.path.exists(ka_path):
                continue
            try:
                ka = load_json(ka_path)
            except json.JSONDecodeError:
                continue
            angles = get_smoothed_angles(ka)
            if not angles:
                continue

            pairs.append({
                "subject":    subj,
                "grp_label":  grp,
                "grp_num":    filming_group(subj),
                "b_stem":     b_stem,
                "smoothed":   angles,
                "a_verdict":  a_ref_verdict,
            })

    return pairs


def run(v2_dir):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pairs = find_pairs(v2_dir)

    if not pairs:
        print("ERROR: no paired a/b data found.", file=sys.stderr)
        sys.exit(1)

    total = len(pairs)
    grp_members = {g: [p for p in pairs if p["grp_num"] == g] for g in [1, 2, 3]}

    print(f"\nPercentile sweep — smooth_pN seat-height peak selection")
    print(f"Pairs available: {total}  (group1={len(grp_members[1])}, group2={len(grp_members[2])}, group3={len(grp_members[3])})")
    print()

    header = f"{'pct':>4}  {'agree':>6}  {'total':>6}  {'pct%':>6}  {'grp1':>6}  {'grp2':>6}  {'grp3':>6}"
    print(header)
    print("-" * len(header))

    csv_rows = []

    for pct in PERCENTILES:
        agrees       = 0
        grp_agrees   = {1: 0, 2: 0, 3: 0}
        grp_totals   = {g: len(grp_members[g]) for g in [1, 2, 3]}

        for p in pairs:
            peak = float(np.percentile(p["smoothed"], pct))
            v    = verdict(peak)
            if v == p["a_verdict"]:
                agrees += 1
                g = p["grp_num"]
                if g in grp_agrees:
                    grp_agrees[g] += 1

        agree_pct = 100 * agrees / total if total > 0 else 0.0
        print(f"  p{pct:<3}  {agrees:>4}/{total:<2}  {agree_pct:>5.0f}%"
              f"  {grp_agrees[1]}/{grp_totals[1]}"
              f"  {grp_agrees[2]}/{grp_totals[2]}"
              f"  {grp_agrees[3]}/{grp_totals[3]}")

        csv_rows.append({
            "percentile":    pct,
            "agreements":    agrees,
            "total":         total,
            "agreement_pct": round(agree_pct, 1),
            "group1_agree":  grp_agrees[1],
            "group2_agree":  grp_agrees[2],
            "group3_agree":  grp_agrees[3],
        })

    # Write CSV
    csv_path = os.path.join(RESULTS_DIR, "percentile_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["percentile", "agreements", "total",
                                          "agreement_pct", "group1_agree",
                                          "group2_agree", "group3_agree"])
        w.writeheader()
        w.writerows(csv_rows)

    # Interpretation
    best_row  = max(csv_rows, key=lambda r: (r["agreements"], -r["percentile"]))
    best_pct  = best_row["percentile"]
    best_n    = best_row["agreements"]
    near_best = [r for r in csv_rows if r["agreements"] >= best_n - 1]
    lo_band   = min(r["percentile"] for r in near_best)
    hi_band   = max(r["percentile"] for r in near_best)

    print()
    print("── Interpretation ───────────────────────────────────────────────────────────────")
    print(f"  Optimal percentile   : p{best_pct}  ({best_n}/{total} agreements, "
          f"{best_row['agreement_pct']:.0f}%)")
    if lo_band == hi_band:
        print(f"  Result is knife-edge : only p{best_pct} achieves this score.")
    else:
        print(f"  Result is flat       : p{lo_band}–p{hi_band} all within 1 of optimum "
              f"({best_n - 1}–{best_n}/{total}).")

    prod_row = next((r for r in csv_rows if r["percentile"] == 80), None)
    if prod_row:
        print(f"  Production value     : p80  ({prod_row['agreements']}/{total} agreements, "
              f"{prod_row['agreement_pct']:.0f}%)")
        if prod_row["agreements"] == best_n:
            print(f"  Verdict              : p80 is tied for the optimum — "
                  f"CURRENT VALUE JUSTIFIED.")
        elif best_n - prod_row["agreements"] <= 1:
            print(f"  Verdict              : p80 is within 1 of optimum — "
                  f"CURRENT VALUE JUSTIFIED (margin is negligible).")
        else:
            print(f"  Verdict              : p{best_pct} outperforms p80 by "
                  f"{best_n - prod_row['agreements']} pairs — CONSIDER CHANGING TO p{best_pct}.")

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
