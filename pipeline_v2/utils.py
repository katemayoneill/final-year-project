"""shared utilities."""
import math
import os

CONF_MIN = 0.1


def calc_angle(A, B, C):
    """return angle at B in degrees for points A, B, C."""
    a2 = (B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2
    b2 = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
    c2 = (A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2
    denom = 2 * math.sqrt(a2 * b2)
    if denom < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (a2 + b2 - c2) / denom))))


def get_xy(keypoints, idx):
    """return (x, y) if joint idx has conf >= CONF_MIN, else None."""
    if not keypoints or idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    return (x, y) if c >= CONF_MIN and x > 0 and y > 0 else None


def video_stem(path, suffix=""):
    """return video stem from a pipeline file path, strip suffix if given."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def print_progress(current, total, suffix=""):
    """print progress bar."""
    pct = current / total if total > 0 else 0
    bar = ("#" * int(pct * 40)).ljust(40)
    print(f"\r  [{bar}] {current}/{total} ({pct:.0%}){suffix}", end="", flush=True)
