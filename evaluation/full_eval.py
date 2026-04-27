#!/usr/bin/env python3
"""
Comprehensive evaluation of Pipeline V2 results.

Covers:
  1. Dataset overview
  2. RPM accuracy — overall, by condition, by cadence group, by filming group, by method
  3. RPM — P1 vs V2 comparison (if output/ present)
  4. Seat height — trainer internal consistency (a30 vs a60)
  5. Seat height — a/b verdict agreement overall and by filming group
  6. Seat height — peak angle method distribution
  7. Frame selection statistics by filming group
  8. Failure / no-output cases

Output is printed to stdout and saved to evaluation/full_eval.txt.

Usage:
  python3 evaluation/full_eval.py [--v2 output_v2/] [--v1 output/] [--gt evaluation/ground_truth.csv]
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

# ── Filming groups ──────────────────────────────────────────────────────────
FILMING_GROUPS = {
    'jenny': 1, 'kate': 1, 'roman': 1,
    'hannah': 2, 'alex': 2,
    'dervla': 3, 'jack': 3, 'jane': 3, 'liam': 3, 'paddy': 3,
}
GROUP_DESC = {
    1: "close-up, good lighting, single pass",
    2: "straight road, further away, two passes (out + back)",
    3: "sports pitch, cycling in multiple directions",
}

OPTIMAL_LOW  = 145.0
OPTIMAL_HIGH = 155.0


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def verdict(deg):
    if deg is None: return "—"
    return "optimal" if OPTIMAL_LOW <= deg <= OPTIMAL_HIGH else ("too_low" if deg < OPTIMAL_LOW else "too_high")

def sv(deg):
    return {"too_low": "low", "optimal": "OPT", "too_high": "HIGH", "—": "—"}.get(verdict(deg), "—")

def parse_stem(stem):
    m = re.match(r'^([a-z]+)(a|b)(\d+)a?$', stem)
    return (m.group(1), m.group(2), m.group(3)) if m else None

def rmse(errs):
    return math.sqrt(sum(e**2 for e in errs) / len(errs)) if errs else None

def load_gt(path):
    gt = {}
    with open(path) as f:
        for line in f.read().splitlines()[1:]:
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2:
                    gt[parts[0].strip()] = float(parts[1].strip())
    return gt

def collect_rows(v2_dir, gt):
    """Load all per-video data from output_v2 into a list of dicts."""
    rows = []
    for stem in sorted(os.listdir(v2_dir)):
        if not os.path.isdir(os.path.join(v2_dir, stem)):
            continue
        parsed = parse_stem(stem)
        if not parsed:
            continue
        name, cond, grp = parsed
        base = os.path.join(v2_dir, stem)

        ap  = os.path.join(base, f"{stem}_assessment.json")
        rp  = os.path.join(base, f"{stem}_rpm.json")
        slp = os.path.join(base, f"{stem}_selection_log.json")
        kap = os.path.join(base, f"{stem}_knee_analysis.json")

        if not os.path.exists(ap):
            rows.append(dict(stem=stem, name=name, cond=cond, grp=grp,
                             film_group=FILMING_GROUPS.get(name, 0),
                             no_output=True))
            continue

        ass = load_json(ap)
        rpm_d = load_json(rp) if os.path.exists(rp) else {}
        sl_d  = load_json(slp) if os.path.exists(slp) else {}
        ka_d  = load_json(kap) if os.path.exists(kap) else {}

        true_rpm = gt.get(stem)
        pred_rpm = rpm_d.get('cadence_rpm')
        rpm_err  = abs(pred_rpm - true_rpm) if pred_rpm is not None and true_rpm else None
        pct_err  = 100 * rpm_err / true_rpm if rpm_err is not None and true_rpm else None

        sl_m = sl_d.get('metrics', {})
        bursts = sl_d.get('selected_bursts', [])
        runs   = ka_d.get('runs', [])

        rows.append(dict(
            stem=stem, name=name, cond=cond, grp=grp,
            film_group=FILMING_GROUPS.get(name, 0),
            no_output=False,
            true_rpm=true_rpm, pred_rpm=pred_rpm,
            rpm_err=rpm_err, pct_err=pct_err,
            rpm_method=rpm_d.get('rpm_method'),
            peak=ass['summary'].get('knee_angle_peak'),
            mean_angle=ass['summary'].get('knee_angle_mean'),
            std_angle=ass['summary'].get('knee_angle_std'),
            verdict=ass['summary'].get('verdict'),
            angle_method=ass['summary'].get('peak_angle_method'),
            n_angle_frames=ass['summary'].get('knee_angles_count'),
            n_selected=sl_m.get('frames_selected'),
            n_total=sl_m.get('frames_processed'),
            n_bursts_found=sl_m.get('total_bursts', sl_m.get('bursts_found')),
            n_bursts_selected=len(bursts),
            n_runs=len(runs),
            n_peaks=ka_d.get('metrics', {}).get('peaks_found', 0),
            direction=ka_d.get('direction'),
        ))
    return rows


def aggregate_rpm(rows):
    """Return MAE, RMSE, mean %err for rows that have rpm_err."""
    errs = [r['rpm_err'] for r in rows if r.get('rpm_err') is not None]
    pcts = [r['pct_err'] for r in rows if r.get('pct_err') is not None]
    if not errs:
        return None, None, None, 0
    return np.mean(errs), rmse(errs), np.mean(pcts), len(errs)


# ── Output helpers ───────────────────────────────────────────────────────────

class Tee:
    def __init__(self, path):
        self._f = open(path, 'w')

    def write(self, s):
        sys.stdout.write(s)
        self._f.write(s)

    def flush(self):
        sys.stdout.flush()
        self._f.flush()

    def close(self):
        self._f.close()


def p(s='', tee=None):
    line = s + '\n'
    if tee:
        tee.write(line)
    else:
        sys.stdout.write(line)


# ── Main evaluation ──────────────────────────────────────────────────────────

def run(v2_dir, v1_dir, gt_path, out):

    gt   = load_gt(gt_path)
    rows = collect_rows(v2_dir, gt)

    valid    = [r for r in rows if not r.get('no_output')]
    b_rows   = [r for r in valid if r['cond'] == 'b']
    a_rows   = [r for r in valid if r['cond'] == 'a']
    no_out   = [r for r in rows  if r.get('no_output')]

    # ── 1. Dataset overview ────────────────────────────────────────────────
    p('', out)
    p('╔══════════════════════════════════════════════════════════════════╗', out)
    p('║          COMPREHENSIVE PIPELINE V2 EVALUATION REPORT            ║', out)
    p('╚══════════════════════════════════════════════════════════════════╝', out)

    subjects = sorted(set(r['name'] for r in rows))
    p('', out)
    p('── 1. DATASET OVERVIEW ─────────────────────────────────────────────────────', out)
    p(f'  Subjects          : {len(subjects)}  ({", ".join(subjects)})', out)
    p(f'  Total videos      : {len(rows)}  (a={len(a_rows)}  b={len(b_rows)}  no_output={len(no_out)})', out)
    p(f'  Cadence groups    : 30 (target 60 RPM) / 60 (target 90 RPM)', out)
    p(f'  Filming groups    :', out)
    for fg in [1, 2, 3]:
        subs = sorted(k for k, v in FILMING_GROUPS.items() if v == fg)
        p(f'    Group {fg}: {", ".join(subs)} — {GROUP_DESC[fg]}', out)

    # ── 2. RPM — overall ──────────────────────────────────────────────────
    p('', out)
    p('── 2. RPM ACCURACY ─────────────────────────────────────────────────────────', out)

    # Per-video table
    p('', out)
    p('  Per-video (real-world b, sorted by |error|):', out)
    p(f"  {'Video':<14} {'Grp':<5} {'FG':<4} {'True':>5} {'Pred':>7} {'|Err|':>6} {'%err':>6}  {'Method'}", out)
    p('  ' + '-'*62, out)

    b_with_err = sorted([r for r in b_rows if r.get('rpm_err') is not None],
                        key=lambda r: r['rpm_err'])
    b_no_rpm   = [r for r in b_rows if r.get('pred_rpm') is None]

    for r in b_with_err:
        p(f"  {r['stem']:<14} {r['grp']:<5} {r['film_group']:<4} "
          f"{r['true_rpm']:>5.0f} {r['pred_rpm']:>7.1f} "
          f"{r['rpm_err']:>6.1f} {r['pct_err']:>5.1f}%  {r['rpm_method']}", out)
    for r in b_no_rpm:
        p(f"  {r['stem']:<14} {r['grp']:<5} {r['film_group']:<4} "
          f"{r['true_rpm']:>5.0f}       —      —      —   no output", out)

    # Aggregate by breakdown
    p('', out)
    p('  Aggregate breakdown:', out)
    p(f"  {'Subset':<35} {'n':>4}  {'MAE':>6}  {'RMSE':>6}  {'mean%err':>9}", out)
    p('  ' + '-'*60, out)

    breakdowns = [
        ('All b (real-world)',           b_rows),
        ('All a (trainer)',              a_rows),
        ('b — cadence group 30',         [r for r in b_rows if r['grp'] == '30']),
        ('b — cadence group 60',         [r for r in b_rows if r['grp'] == '60']),
        ('b — peak_detection',           [r for r in b_rows if r.get('rpm_method') == 'peak_detection']),
        ('b — autocorrelation',          [r for r in b_rows if r.get('rpm_method') == 'autocorrelation']),
        ('b — Group 1 (close-up)',       [r for r in b_rows if r['film_group'] == 1]),
        ('b — Group 2 (straight road)',  [r for r in b_rows if r['film_group'] == 2]),
        ('b — Group 3 (sports pitch)',   [r for r in b_rows if r['film_group'] == 3]),
    ]
    for label, subset in breakdowns:
        mae, rms, mpct, n = aggregate_rpm(subset)
        if mae is not None:
            p(f"  {label:<35} {n:>4}  {mae:>6.1f}  {rms:>6.1f}  {mpct:>8.1f}%", out)
        else:
            p(f"  {label:<35} {n:>4}  {'—':>6}  {'—':>6}  {'—':>9}", out)

    # ── 3. RPM P1 vs V2 ───────────────────────────────────────────────────
    if v1_dir and os.path.isdir(v1_dir):
        p('', out)
        p('── 3. RPM: PIPELINE 1 vs PIPELINE V2 ──────────────────────────────────────', out)
        p('', out)
        p(f"  {'Video':<14} {'True':>5}  {'P1':>7} {'P1err':>6}  {'V2':>7} {'V2err':>6}  {'Better'}", out)
        p('  ' + '-'*60, out)
        improved = worse = same = 0
        p1_errs, v2_errs = [], []
        for r in sorted(b_rows, key=lambda x: x['stem']):
            v2_err = r.get('rpm_err')
            # load P1
            p1_rp = os.path.join(v1_dir, r['stem'], f"{r['stem']}_rpm.json")
            if not os.path.exists(p1_rp):
                continue
            p1_d = load_json(p1_rp)
            p1_pred = p1_d.get('cadence_rpm')
            p1_err = abs(p1_pred - r['true_rpm']) if p1_pred and r.get('true_rpm') else None
            if p1_err is None or v2_err is None:
                continue
            p1_errs.append(p1_err)
            v2_errs.append(v2_err)
            if v2_err < p1_err - 0.5:
                flag = 'V2 ✓'; improved += 1
            elif v2_err > p1_err + 0.5:
                flag = 'P1'; worse += 1
            else:
                flag = '='; same += 1
            p(f"  {r['stem']:<14} {r['true_rpm']:>5.0f}  {p1_pred:>7.1f} {p1_err:>6.1f}  "
              f"{r['pred_rpm']:>7.1f} {v2_err:>6.1f}  {flag}", out)
        n_shared = len(p1_errs)
        if n_shared:
            p('', out)
            p(f"  Shared videos: {n_shared}  V2 better: {improved}  P1 better: {worse}  same: {same}", out)
            p(f"  P1 MAE={np.mean(p1_errs):.1f}  V2 MAE={np.mean(v2_errs):.1f}  "
              f"reduction={np.mean(p1_errs)-np.mean(v2_errs):.1f} RPM", out)
    else:
        p('', out)
        p('── 3. RPM: P1 vs V2 — skipped (output/ not found) ─────────────────────────', out)

    # ── 4. Seat height — trainer consistency ──────────────────────────────
    p('', out)
    p('── 4. SEAT HEIGHT: TRAINER CONSISTENCY (a30 vs a60) ───────────────────────', out)
    p('', out)
    p(f"  {'Subject':<10} {'peak_a30':>9} {'v_a30':<8} {'peak_a60':>9} {'v_a60':<8} {'Δ°':>5}  {'same?'}", out)
    p('  ' + '-'*58, out)

    a_by_sub = defaultdict(dict)
    for r in a_rows:
        a_by_sub[r['name']][r['grp']] = r

    deltas = []
    for name in sorted(a_by_sub):
        d = a_by_sub[name]
        if '30' not in d or '60' not in d:
            continue
        r30, r60 = d['30'], d['60']
        if r30['peak'] is None or r60['peak'] is None:
            continue
        delta = abs(r30['peak'] - r60['peak'])
        deltas.append(delta)
        same = '✓' if r30['verdict'] == r60['verdict'] else '✗'
        p(f"  {name:<10} {r30['peak']:>9.2f} {sv(r30['peak']):<8} "
          f"{r60['peak']:>9.2f} {sv(r60['peak']):<8} {delta:>5.2f}  {same}", out)
    if deltas:
        p('', out)
        p(f"  Peak delta: mean={np.mean(deltas):.2f}°  median={np.median(deltas):.2f}°  "
          f"min={np.min(deltas):.2f}°  max={np.max(deltas):.2f}°", out)
        same_count = sum(
            1 for name in a_by_sub
            if '30' in a_by_sub[name] and '60' in a_by_sub[name]
            and a_by_sub[name]['30']['verdict'] == a_by_sub[name]['60']['verdict']
        )
        total_pairs = sum(
            1 for name in a_by_sub if '30' in a_by_sub[name] and '60' in a_by_sub[name]
        )
        p(f"  Same verdict: {same_count}/{total_pairs}", out)

    # ── 5. Seat height — a/b verdict agreement ────────────────────────────
    p('', out)
    p('── 5. SEAT HEIGHT: a/b VERDICT AGREEMENT ──────────────────────────────────', out)

    all_rows_by = defaultdict(lambda: defaultdict(dict))
    for r in valid:
        all_rows_by[r['name']][r['grp']][r['cond']] = r

    p('', out)
    p(f"  {'Subject':<10} {'Grp':<5} {'FG':<4} {'a_peak':>7} {'b_peak':>7} {'Δ°':>5}  "
      f"{'a_v':<8} {'b_v':<8} {'match'}", out)
    p('  ' + '-'*65, out)

    matches = 0
    pairs_total = 0
    fg_matches = defaultdict(lambda: [0, 0])  # [matches, total]

    for name in sorted(all_rows_by):
        for grp in sorted(all_rows_by[name]):
            d = all_rows_by[name][grp]
            if 'a' not in d or 'b' not in d:
                continue
            a, b = d['a'], d['b']
            if a['peak'] is None or b['peak'] is None:
                continue
            delta = abs(a['peak'] - b['peak'])
            match = a['verdict'] == b['verdict']
            pairs_total += 1
            fg = FILMING_GROUPS.get(name, 0)
            fg_matches[fg][1] += 1
            if match:
                matches += 1
                fg_matches[fg][0] += 1
            flag = '✓' if match else '✗'
            p(f"  {name:<10} {grp:<5} {fg:<4} {a['peak']:>7.1f} {b['peak']:>7.1f} {delta:>5.1f}  "
              f"{a['verdict']:<8} {b['verdict']:<8} {flag}", out)

    p('', out)
    p(f"  Overall agreement: {matches}/{pairs_total} ({100*matches/pairs_total:.0f}%)", out)
    p('', out)
    p('  By filming group:', out)
    for fg in [1, 2, 3]:
        m, t = fg_matches[fg]
        subs = sorted(k for k, v in FILMING_GROUPS.items() if v == fg)
        p(f"    Group {fg} ({', '.join(subs)}): {m}/{t} ({100*m/t:.0f}%)", out)

    # ── 6. Peak angle method distribution ────────────────────────────────
    p('', out)
    p('── 6. PEAK ANGLE METHOD DISTRIBUTION ──────────────────────────────────────', out)
    p('', out)
    for cond_label, cond_rows in [('Trainer (a)', a_rows), ('Real-world (b)', b_rows)]:
        methods = defaultdict(int)
        for r in cond_rows:
            methods[r.get('angle_method', '—')] += 1
        p(f"  {cond_label}:", out)
        for method, count in sorted(methods.items(), key=lambda x: -x[1]):
            p(f"    {method:<20} {count:>3} videos", out)

    # ── 7. Frame selection statistics ─────────────────────────────────────
    p('', out)
    p('── 7. FRAME SELECTION STATISTICS (real-world b) ───────────────────────────', out)
    p('', out)
    p(f"  {'Video':<14} {'FG':<4} {'Selected':>9} {'Total':>7} {'Rate':>7} {'Bursts':>7} {'Runs':>6} {'Peaks':>6}", out)
    p('  ' + '-'*60, out)

    for fg in [1, 2, 3]:
        fg_b = [r for r in b_rows if r['film_group'] == fg]
        for r in sorted(fg_b, key=lambda x: x['stem']):
            sel  = r.get('n_selected') or 0
            tot  = r.get('n_total') or 0
            rate = f"{100*sel/tot:.1f}%" if tot else '—'
            p(f"  {r['stem']:<14} {fg:<4} {sel:>9} {tot:>7} {rate:>7} "
              f"{r.get('n_bursts_selected') or 0:>7} "
              f"{r.get('n_runs') or 0:>6} "
              f"{r.get('n_peaks') or 0:>6}", out)
        # group summary
        sel_vals  = [r['n_selected'] for r in fg_b if r.get('n_selected')]
        tot_vals  = [r['n_total']    for r in fg_b if r.get('n_total')]
        if sel_vals:
            p(f"  {'Group '+str(fg)+' mean':<14} {'':4} {int(np.mean(sel_vals)):>9} "
              f"{int(np.mean(tot_vals)):>7} "
              f"{100*np.mean(sel_vals)/np.mean(tot_vals):>6.1f}%", out)
        p('', out)

    # ── 8. Failure / no-output cases ──────────────────────────────────────
    p('── 8. FAILURE CASES (no RPM output) ───────────────────────────────────────', out)
    p('', out)

    # Videos with no RPM at all (no output or pred_rpm=None)
    b_failed_rpm = [r for r in b_rows if r.get('pred_rpm') is None]
    if b_failed_rpm:
        p('  Real-world videos with no RPM output:', out)
        for r in b_failed_rpm:
            sel = r.get('n_selected', 0) or 0
            peaks = r.get('n_peaks', 0) or 0
            p(f"    {r['stem']:<14}  selected={sel}  peaks={peaks}  "
              f"true_rpm={r.get('true_rpm','?')}", out)
    else:
        p('  No real-world videos with missing RPM output.', out)

    p('', out)
    p('  Note: romanb30/b60, kateb60 fail due to very short selection windows', out)
    p('  (<32 frames, <1s). romanb60 true RPM=39 — outside validated range (~50+ RPM).', out)

    p('', out)
    p('── END OF REPORT ───────────────────────────────────────────────────────────', out)
    p('', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2', default='output_v2', help='Path to output_v2/')
    parser.add_argument('--v1', default='output',    help='Path to output/ (Pipeline 1, optional)')
    parser.add_argument('--gt', default='evaluation/ground_truth.csv')
    args = parser.parse_args()

    if not os.path.isdir(args.v2):
        print(f'ERROR: {args.v2} not found', file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join('evaluation', 'full_eval.txt')
    tee = Tee(out_path)
    v1 = args.v1 if os.path.isdir(args.v1) else None
    run(args.v2, v1, args.gt, tee)
    tee.close()
    sys.stdout.write(f'\nSaved to {out_path}\n')


if __name__ == '__main__':
    main()
