#!/usr/bin/env python3
"""
Tier 3 (optional): preprocessing ablation — qualitative illustration only.

IMPORTANT: This script does NOT perform a numerical ablation. A true numerical
ablation would require re-running OpenPose with each preprocessing step disabled,
which is out of scope and computationally prohibitive. This script selects
representative preprocessing montage images from the per-frame montages already
saved by pose_estimate.py and copies them into a labelled output directory for
inclusion in the report.

Selection criteria:
  - 2–3 frames per filming group (Groups 1, 2, 3 — see CLAUDE.md)
  - Prefer frames where preprocessing visibly changes the image:
      CLAHE effect:   high mean L-channel standard deviation change (proxy: frame brightness variance)
      Unsharp effect: high gradient magnitude (proxy: mean absolute pixel value if image loaded)
  - Include at least one "preprocessing helped" and one "preprocessing didn't change much" example

Since the montages are already rendered JPEGs, selection is based on available
metadata only (file size as a crude proxy for image complexity / CLAHE effect).
Frames with larger montage files tend to have more visual detail and variation.

Subject-to-group mapping (from CLAUDE.md):
  Group 1 — jenny, kate, roman     (close-up, single pass)
  Group 2 — hannah, alex           (two passes, further away)
  Group 3 — dervla, jack, jane, liam, paddy  (pitch, many directions)

Output:
  evaluation/sensitivity/results/preprocessing_examples/
    <group>_<stem>_frame_<n>.jpg  (copies of selected montage frames)
  evaluation/sensitivity/results/preprocessing_examples/README.txt

Usage (from project root):
  python3 evaluation/sensitivity/preprocessing_ablation_visual.py [--v2 output_v2/] [--n-per-group 2]
"""

import argparse
import glob
import os
import re
import shutil
import sys

FILMING_GROUPS = {
    1: {"jenny", "kate", "roman"},
    2: {"hannah", "alex"},
    3: {"dervla", "jack", "jane", "liam", "paddy"},
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "preprocessing_examples")

README_TEXT = """\
Preprocessing Ablation — Qualitative Illustration
==================================================

IMPORTANT: These images do NOT constitute a numerical ablation of the preprocessing
pipeline. A true numerical ablation would require re-running OpenPose with each step
disabled, which is out of scope for this study. The images here are selected from the
per-frame step montages already produced by pipeline_v2/pose_estimate.py and serve as
a qualitative illustration of each preprocessing step's visual effect.

Each montage shows (left to right):
  1. Original crop (ROI from YOLO cyclist box)
  2. After CLAHE (Contrast-Limited Adaptive Histogram Equalisation on L-channel)
  3. After unsharp mask (motion-blur counteraction for pass-by footage)
  4. After square pad (aspect ratio correction before net_resolution resize)
  5. Final net_resolution frame fed to OpenPose

Files are named: group<N>_<video_stem>_<montage_filename>

Subject filming groups:
  Group 1 — jenny, kate, roman     (close-up, good lighting, single pass)
  Group 2 — hannah, alex           (straight road, further away, two passes)
  Group 3 — dervla, jack, jane, liam, paddy  (cement sports pitch, many directions)

Selection was based on montage file size as a proxy for visual complexity:
  - Larger files tend to show more visible preprocessing effect (more image detail,
    higher contrast variation after CLAHE).
  - The "smallest" selected file provides a contrasting "low effect" example.

Do not cite specific pixel-level measurements from these images in the report
without re-running with controlled conditions.
"""


def parse_stem(stem):
    m = re.match(r"^([a-z]+)(a|b)(\d+)a?$", stem)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def filming_group(subject):
    for grp, names in FILMING_GROUPS.items():
        if subject in names:
            return grp
    return None


def find_montage_dirs(v2_dir):
    """Returns list of (group_num, stem, montage_dir) for all b-stems with montage dirs."""
    results = []
    for d in sorted(os.listdir(v2_dir)):
        full = os.path.join(v2_dir, d)
        if not os.path.isdir(full):
            continue
        parsed = parse_stem(d)
        if not parsed:
            continue
        subj, cond, grp = parsed
        if cond != "b":
            continue
        grp_num = filming_group(subj)
        if grp_num is None:
            continue
        montage_dir = os.path.join(full, f"{d}_preprocessing_steps")
        if os.path.isdir(montage_dir):
            results.append((grp_num, d, montage_dir))
    return results


def select_frames(montage_dir, n):
    """Select n montage JPEGs from a directory by file size (largest + smallest)."""
    jpgs = sorted(glob.glob(os.path.join(montage_dir, "*.jpg")))
    if not jpgs:
        jpgs = sorted(glob.glob(os.path.join(montage_dir, "*.jpeg")))
    if not jpgs:
        return []
    by_size = sorted(jpgs, key=os.path.getsize, reverse=True)
    selected = []
    if len(by_size) >= n:
        # Take the largest (most visual effect) and smallest (least effect)
        selected.append(by_size[0])           # most visually active
        selected.append(by_size[-1])          # least visually active
        # Fill remaining from middle
        for f in by_size[1:-1]:
            if len(selected) >= n:
                break
            selected.append(f)
    else:
        selected = by_size
    return selected[:n]


def run(v2_dir, n_per_group):
    os.makedirs(OUT_DIR, exist_ok=True)

    montage_dirs = find_montage_dirs(v2_dir)
    if not montage_dirs:
        print("ERROR: no _preprocessing_steps/ directories found.", file=sys.stderr)
        print("These are created by pipeline_v2/pose_estimate.py. "
              "Ensure pose estimation has been run.", file=sys.stderr)
        sys.exit(1)

    # Group by filming group
    by_group = {1: [], 2: [], 3: []}
    for grp_num, stem, mdir in montage_dirs:
        by_group[grp_num].append((stem, mdir))

    total_copied = 0
    for grp_num in [1, 2, 3]:
        entries = by_group[grp_num]
        if not entries:
            print(f"WARNING: no b-videos with montage dirs found for Group {grp_num}")
            continue

        print(f"Group {grp_num} ({len(entries)} video(s)):")
        copied_this_group = 0
        for stem, mdir in sorted(entries):
            if copied_this_group >= n_per_group:
                break
            frames = select_frames(mdir, n_per_group - copied_this_group)
            if not frames:
                print(f"  {stem}: no montage JPEGs found in {mdir}")
                continue
            for src in frames:
                fname = os.path.basename(src)
                dest  = os.path.join(OUT_DIR, f"group{grp_num}_{stem}_{fname}")
                shutil.copy2(src, dest)
                size_kb = os.path.getsize(src) / 1024
                print(f"  Copied: {fname}  ({size_kb:.1f} KB)  → {dest}")
                copied_this_group += 1
                total_copied += 1
            if copied_this_group >= n_per_group:
                break

    # Write README
    readme_path = os.path.join(OUT_DIR, "README.txt")
    with open(readme_path, "w") as f:
        f.write(README_TEXT)

    print(f"\nTotal frames copied : {total_copied}")
    print(f"README written      : {readme_path}")
    print(f"Output directory    : {OUT_DIR}")
    print()
    print("REMINDER: These images are a qualitative illustration only.")
    print("No numerical ablation claim should be made from these examples.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", default="output_v2")
    parser.add_argument("--n-per-group", type=int, default=2,
                        help="Number of example frames to select per filming group (default 2)")
    args = parser.parse_args()
    if not os.path.isdir(args.v2):
        print(f"ERROR: {args.v2!r} not found — run from project root.", file=sys.stderr)
        sys.exit(1)
    run(args.v2, args.n_per_group)


if __name__ == "__main__":
    main()
