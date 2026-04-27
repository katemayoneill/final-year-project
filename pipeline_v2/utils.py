#!/usr/bin/env python3
"""Shared utilities for pipeline_v2 scripts."""
import math
import os

CONF_MIN = 0.1


def calc_angle(A, B, C):
    """Angle at B in degrees (law of cosines) for points A, B, C."""
    a2 = (B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2
    b2 = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
    c2 = (A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2
    denom = 2 * math.sqrt(a2 * b2)
    if denom < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (a2 + b2 - c2) / denom))))


def get_xy(keypoints, idx):
    """Returns (x, y) if joint idx has conf >= CONF_MIN, else None."""
    if not keypoints or idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    return (x, y) if c >= CONF_MIN and x > 0 and y > 0 else None


def video_stem(path, strip_suffix=""):
    """Returns the video stem from a pipeline file path, stripping a known suffix if given."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def print_progress(current, total, suffix=""):
    """Prints an in-place ASCII progress bar."""
    pct = current / total if total > 0 else 0
    bar = ("#" * int(pct * 40)).ljust(40)
    print(f"\r  [{bar}] {current}/{total} ({pct:.0%}){suffix}", end="", flush=True)
