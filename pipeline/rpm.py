#!/usr/bin/env python3
"""
Stage 4: RPM (cadence) calculation from keypoint time series.
Tracks knee angle across selected frames, finds peaks (maximum extension =
bottom of pedal stroke), and computes cadence in RPM from inter-peak periods.

Direction of travel is determined from front_wheel vs back_wheel x-positions
in the selection log — the camera-facing (non-occluded) knee is used as primary.

Uses only the longest contiguous run of selected frames (gap <= CONTIGUOUS_GAP_FRAMES)
to avoid false peaks caused by jumps between separate side-angle windows.

Usage: python3 rpm.py <video_keypoints.json>
Output: <video>_rpm.json
"""
import json
import math
import os
import sys

try:
    from scipy.signal import savgol_filter, find_peaks as scipy_find_peaks, correlate
    import numpy as _np
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not installed — falling back to simple moving average and manual peak detection")
    SCIPY_AVAILABLE = False

CONF_MIN              = 0.1
PEAK_PROMINENCE       = 20.0   # degrees — minimum rise above neighbouring troughs
CONTIGUOUS_GAP_FRAMES = 30     # max frame-index gap within one contiguous run (~0.5s at 60fps)
SMOOTH_WINDOW         = 5      # frames — moving-average window (fallback only)
MAX_CADENCE_RPM       = 130    # physical upper bound — sprint cyclists rarely exceed 130 RPM

SAVGOL_WINDOW = 11  # Savitzky-Golay window length (must be odd)
SAVGOL_ORDER  = 2   # polynomial order

# Body25 joint indices
R_HIP, R_KNEE, R_ANKLE = 9, 10, 11
L_HIP, L_KNEE, L_ANKLE = 12, 13, 14


def calc_angle(A, B, C):
    a2 = (B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2
    b2 = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
    c2 = (A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2
    denom = 2 * math.sqrt(a2 * b2)
    if denom < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (a2 + b2 - c2) / denom))))


def get_xy(keypoints, idx):
    if not keypoints or idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    return (x, y) if c >= CONF_MIN and x > 0 and y > 0 else None


def smooth(angles, window=SMOOTH_WINDOW):
    """Savitzky-Golay filter (scipy) when available; else simple centred moving average."""
    if SCIPY_AVAILABLE and len(angles) >= SAVGOL_WINDOW:
        return list(savgol_filter(angles, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_ORDER))
    # Fallback: simple centred moving average
    if window <= 1 or len(angles) < window:
        return angles[:]
    half = window // 2
    out = []
    for i in range(len(angles)):
        lo = max(0, i - half)
        hi = min(len(angles), i + half + 1)
        out.append(sum(angles[lo:hi]) / (hi - lo))
    return out


def autocorrelation_rpm(timestamps, smoothed_angles):
    """
    Estimate RPM from dominant periodicity via normalised autocorrelation.
    Used as fallback when peak detection finds fewer than 2 peaks.
    Returns (rpm, period_sec) or (None, None).
    """
    if not SCIPY_AVAILABLE or len(smoothed_angles) < 10:
        return None, None

    signal = _np.array(smoothed_angles, dtype=float)
    signal -= signal.mean()

    corr = correlate(signal, signal, mode='full')
    corr = corr[len(corr) // 2:]   # positive lags only
    if corr[0] == 0:
        return None, None
    corr /= corr[0]                 # normalise to 1 at lag 0

    # Minimum lag: enforce physical upper-bound on cadence
    if len(timestamps) < 2:
        return None, None
    mean_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    if mean_dt <= 0:
        return None, None
    min_lag = max(1, int((60.0 / MAX_CADENCE_RPM) / mean_dt))

    peaks, _ = scipy_find_peaks(corr[1:], height=0.1, distance=min_lag)
    if len(peaks) == 0:
        return None, None

    lag_frames = int(peaks[0]) + 1  # +1 because we searched corr[1:]
    period_sec = lag_frames * mean_dt
    rpm = 60.0 / period_sec

    if 20 <= rpm <= MAX_CADENCE_RPM:
        return round(rpm, 1), round(period_sec, 4)
    return None, None


def detect_peaks(angles, timestamps, prominence=PEAK_PROMINENCE):
    """Return indices of local maxima with sufficient prominence and minimum inter-peak gap."""
    min_gap_sec = 60.0 / MAX_CADENCE_RPM

    if SCIPY_AVAILABLE and len(angles) >= 3:
        arr = _np.array(angles, dtype=float)
        # Adaptive prominence: at least the fixed threshold or one full std dev of the signal
        adaptive_prom = max(prominence, float(arr.std()))

        mean_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) >= 2 else 0
        min_gap_samples = max(1, int(min_gap_sec / mean_dt)) if mean_dt > 0 else 1

        peak_idxs, _ = scipy_find_peaks(arr, prominence=adaptive_prom, distance=min_gap_samples)
        return list(peak_idxs)

    # Fallback: original manual implementation
    candidates = []
    n = len(angles)
    for i in range(1, n - 1):
        if angles[i] > angles[i - 1] and angles[i] > angles[i + 1]:
            left_min  = min(angles[:i])
            right_min = min(angles[i + 1:])
            if angles[i] - max(left_min, right_min) >= prominence:
                candidates.append(i)
    peaks = []
    for idx in candidates:
        if peaks and (timestamps[idx] - timestamps[peaks[-1]]) < min_gap_sec:
            if angles[idx] > angles[peaks[-1]]:
                peaks[-1] = idx
        else:
            peaks.append(idx)
    return peaks


def split_into_runs(series, gap):
    """Split (frame_idx, timestamp, angle) triples into contiguous runs."""
    if not series:
        return []
    runs = []
    current = [series[0]]
    for prev, curr in zip(series, series[1:]):
        if curr[0] - prev[0] <= gap:
            current.append(curr)
        else:
            runs.append(current)
            current = [curr]
    runs.append(current)
    return runs


def detect_direction(selection_log_path):
    """
    Return 'left' or 'right' (direction cyclist is travelling) by comparing
    median front_wheel vs back_wheel x-centre across all selected frames.
    front_wheel to the right of back_wheel  → moving right.
    front_wheel to the left  of back_wheel  → moving left.
    Returns None if log not found or insufficient data.
    """
    if not os.path.exists(selection_log_path):
        return None
    with open(selection_log_path) as f:
        log = json.load(f)

    deltas = []
    for entry in log.get("selected_frames", []):
        fb = entry.get("front_wheel_box")
        bb = entry.get("back_wheel_box")
        if fb and bb:
            fx = (fb[0] + fb[2]) / 2
            bx = (bb[0] + bb[2]) / 2
            deltas.append(fx - bx)

    if not deltas:
        return None
    deltas.sort()
    median = deltas[len(deltas) // 2]
    return "right" if median > 0 else "left"


if len(sys.argv) < 2:
    print("Usage: python3 rpm.py <video_keypoints.json>")
    sys.exit(1)

kp_path = sys.argv[1]
with open(kp_path) as f:
    data = json.load(f)

base     = os.path.splitext(kp_path)[0].replace("_keypoints", "")
out_path = base + "_rpm.json"
log_path = base + "_selection_log.json"


direction = detect_direction(log_path)

if direction == "right":
    primary   = (L_HIP, L_KNEE, L_ANKLE)
    fallback  = (R_HIP, R_KNEE, R_ANKLE)
    knee_used = "right"
else:
    primary   = (R_HIP, R_KNEE, R_ANKLE)
    fallback  = (L_HIP, L_KNEE, L_ANKLE)
    knee_used = "left"

if direction:
    print(f"Direction of travel    : {direction}  →  using {knee_used} knee (camera-facing side)")
else:
    print("Direction of travel    : unknown (selection log missing) — defaulting to right knee")

# Build time series: (frame_idx, timestamp, angle)
series = []
for entry in data["frames"]:
    kp  = entry.get("keypoints", [])
    t   = entry["timestamp"]
    idx = entry["frame_idx"]

    hip   = get_xy(kp, primary[0])
    knee  = get_xy(kp, primary[1])
    ankle = get_xy(kp, primary[2])

    if not (hip and knee and ankle):
        hip   = get_xy(kp, fallback[0])
        knee  = get_xy(kp, fallback[1])
        ankle = get_xy(kp, fallback[2])

    angle = calc_angle(hip, knee, ankle) if (hip and knee and ankle) else None

    if angle is not None:
        series.append((idx, t, angle))

series.sort(key=lambda x: x[0])

# Split into contiguous runs and pick the longest
runs = split_into_runs(series, CONTIGUOUS_GAP_FRAMES)
best_run = max(runs, key=len) if runs else []

timestamps = [s[1] for s in best_run]
angles     = [s[2] for s in best_run]
smoothed   = smooth(angles)

peak_indices    = detect_peaks(smoothed, timestamps) if len(smoothed) >= 3 else []
peak_timestamps = [timestamps[i] for i in peak_indices]

cadence_rpm    = None
std_dev_rpm    = None
cycle_periods  = []
rpm_method     = "peak_detection"

if len(peak_timestamps) >= 2:
    periods    = [peak_timestamps[i+1] - peak_timestamps[i] for i in range(len(peak_timestamps) - 1)]
    avg_period = sum(periods) / len(periods)
    cadence_rpm = round(60.0 / avg_period, 1) if avg_period > 0 else None

    if len(periods) > 1:
        mean_p = avg_period
        var    = sum((p - mean_p) ** 2 for p in periods) / len(periods)
        std_p  = math.sqrt(var)
        if avg_period > std_p:
            std_dev_rpm = round(
                (60.0 / (avg_period - std_p) - 60.0 / (avg_period + std_p)) / 2, 1
            )

    cycle_periods = [round(p, 4) for p in periods]

else:
    # Fewer than 2 peaks — try autocorrelation as fallback
    ac_rpm, ac_period = autocorrelation_rpm(timestamps, smoothed)
    if ac_rpm is not None:
        cadence_rpm  = ac_rpm
        rpm_method   = "autocorrelation"
        cycle_periods = [ac_period] if ac_period else []
        if ac_period and len(timestamps) >= 2:
            duration = timestamps[-1] - timestamps[0]
            peak_timestamps = []  # timestamps not available from autocorrelation
            estimated_cycles = int(duration / ac_period)
            # update peak_indices count for metrics
            peak_indices = [None] * estimated_cycles

run_info = {
    "frame_idx_start": best_run[0][0]  if best_run else None,
    "frame_idx_end":   best_run[-1][0] if best_run else None,
    "frame_count":     len(best_run),
    "duration_sec":    round(timestamps[-1] - timestamps[0], 2) if len(timestamps) >= 2 else 0,
    "total_runs":      len(runs),
}

output = {
    "video":             data["video"],
    "direction":         direction,
    "knee_used":         knee_used,
    "cadence_rpm":       cadence_rpm,
    "cycle_count":       len(peak_timestamps) if rpm_method == "peak_detection" else len(peak_indices),
    "cycle_timestamps":  [round(t, 4) for t in peak_timestamps],
    "cycle_periods_sec": cycle_periods,
    "std_dev_rpm":       std_dev_rpm,
    "rpm_method":        rpm_method,
    "best_run":          run_info,
    "metrics": {
        "frames_with_angle":    len(series),
        "frames_in_best_run":   len(best_run),
        "peaks_found":          len(peak_indices),
        "time_span_sec":        run_info["duration_sec"],
    },
    "angle_series": [[t, a] for t, a in zip(timestamps, smoothed)],
    "peak_timestamps": peak_timestamps,
}

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

m = output["metrics"]
print(f"Frames with knee angle : {m['frames_with_angle']}")
print(f"Contiguous runs found  : {run_info['total_runs']}")
print(f"Best run               : frames {run_info['frame_idx_start']}–{run_info['frame_idx_end']}  ({run_info['frame_count']} frames, {run_info['duration_sec']}s)")
print(f"Peaks (pedal cycles)   : {output['cycle_count']}  [method: {rpm_method}]")
if cadence_rpm is not None:
    suffix = f"  ±{std_dev_rpm}" if std_dev_rpm else ""
    print(f"Cadence                : {cadence_rpm} RPM{suffix}")
else:
    print("Cadence                : insufficient data (need >= 2 peaks or clear autocorrelation period)")
print(f"RPM saved              → {out_path}")
