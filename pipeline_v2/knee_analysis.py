"""
stage 3: knee angle cycle analysis.

usage: python3 knee_analysis.py <video_keypoints.json>
output: output_v2/<stem>/<stem>_knee_analysis.json
"""
import json
import os
import sys

import numpy as np
try:
    from scipy.signal import correlate, find_peaks, savgol_filter
except ImportError:
    print("[ERROR] scipy isn't installed.}")
    sys.exit(1)

from utils import calc_angle, get_xy, video_stem


SAVGOL_WINDOW = 11
SAVGOL_ORDER = 2

PEAK_PROMINENCE = 25.0
MAX_FRAME_GAP = 30 
SMOOTH_WINDOW = 5    

MAX_CADENCE_RPM = 130  
MIN_CADENCE_RPM = 20

AUTOCORR_MIN_HEIGHT = 0.1
AUTOCORR_MIN_FRAMES = 10

MIN_PEAKS = 3

R_HIP, R_KNEE, R_ANKLE = 9, 10, 11
L_HIP, L_KNEE, L_ANKLE = 12, 13, 14


def smooth(angles):
    """Savitzky-Golay filter; falls back to centred moving average for short series."""
    if len(angles) >= SAVGOL_WINDOW:
        return list(savgol_filter(angles, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_ORDER))
    if SMOOTH_WINDOW <= 1 or len(angles) < SMOOTH_WINDOW:
        return angles[:]
    half = SMOOTH_WINDOW // 2
    out = []
    for i in range(len(angles)):
        lo = max(0, i - half)
        hi = min(len(angles), i + half + 1)
        out.append(sum(angles[lo:hi]) / (hi - lo))
    return out


def detect_peaks(angles, timestamps):
    """return indices of local maxima with adaptive prominence and min inter-peak gap."""
    min_gap_sec = 60.0 / MAX_CADENCE_RPM

    if len(angles) >= MIN_PEAKS:
        arr = np.array(angles, dtype=float)
        adaptive_prom = max(PEAK_PROMINENCE, float(arr.std()))
        mean_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) >= 2 else 0
        min_gap_samples = max(1, int(min_gap_sec / mean_dt)) if mean_dt > 0 else 1
        peak_idxs, _ = find_peaks(arr, prominence=adaptive_prom, distance=min_gap_samples)
        return list(peak_idxs)

    candidates = []
    n = len(angles)
    for i in range(1, n - 1):
        if angles[i] > angles[i - 1] and angles[i] > angles[i + 1]:
            left_min = min(angles[:i])
            right_min = min(angles[i + 1:])
            if angles[i] - max(left_min, right_min) >= PEAK_PROMINENCE:
                candidates.append(i)
    peaks = []
    for idx in candidates:
        if peaks and (timestamps[idx] - timestamps[peaks[-1]]) < min_gap_sec:
            if angles[idx] > angles[peaks[-1]]:
                peaks[-1] = idx
        else:
            peaks.append(idx)
    return peaks


def split_runs(series, gap):
    """split (frame_idx, timestamp, angle) triples into runs."""
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


def get_direction(entries):
    """return direction from wheel x-positions."""
    deltas = []
    for entry in entries:
        fb = entry.get("fw_box")
        bb = entry.get("bw_box")
        if fb and bb:
            fx = (fb[0] + fb[2]) / 2
            bx = (bb[0] + bb[2]) / 2
            deltas.append(fx - bx)
    if not deltas:
        return None
    deltas.sort()
    return "right" if deltas[len(deltas) // 2] > 0 else "left"


def load_direction_map(log_path):
    """returns (global_direction, frame_direction_map) from selection log."""
    try:
        with open(log_path) as f:
            log = json.load(f)
    except (FileNotFoundError, OSError):
        return None, {}

    frames = log.get("selected_frames", [])
    bursts = log.get("selected_bursts", [])

    if not frames:
        return None, {}

    if not bursts:
        return get_direction(frames), {}

    burst_entries = {b["burst_id"]: [] for b in bursts}
    for entry in frames:
        fi = entry.get("frame_idx")
        if fi is None:
            continue
        for b in bursts:
            if b["start_frame_idx"] <= fi <= b["end_frame_idx"]:
                burst_entries[b["burst_id"]].append(entry)
                break

    burst_directions = {bid: get_direction(ents) for bid, ents in burst_entries.items()}

    frame_direction_map = {}
    for entry in frames:
        fi = entry.get("frame_idx")
        if fi is None:
            continue
        for b in bursts:
            if b["start_frame_idx"] <= fi <= b["end_frame_idx"]:
                frame_direction_map[fi] = burst_directions.get(b["burst_id"])
                break

    longest = max(bursts, key=lambda b: b.get("frame_count", 0))
    global_direction = burst_directions.get(longest["burst_id"]) or get_direction(frames)

    return global_direction, frame_direction_map


def direction_to_joints(direction):
    """map direction string to joints"""
    if direction == "left":
        return (L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE), "left"
    return (R_HIP, R_KNEE, R_ANKLE), (L_HIP, L_KNEE, L_ANKLE), "right"


def autocorrelation_period(timestamps, smoothed_angles):
    """return dominant period in seconds using normalised autocorrelation."""
    if len(smoothed_angles) < AUTOCORR_MIN_FRAMES:
        return None
    signal = np.array(smoothed_angles, dtype=float)
    signal -= signal.mean()
    corr = correlate(signal, signal, mode='full')
    corr = corr[len(corr) // 2:]
    if corr[0] == 0:
        return None
    corr /= corr[0]
    if len(timestamps) < 2:
        return None
    mean_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    if mean_dt <= 0:
        return None
    min_lag = max(1, int((60.0 / MAX_CADENCE_RPM) / mean_dt))
    ac_peaks, _ = find_peaks(corr[1:], height=AUTOCORR_MIN_HEIGHT, distance=min_lag)
    if len(ac_peaks) == 0:
        return None
    period_sec = (int(ac_peaks[0]) + 1) * mean_dt
    rpm = 60.0 / period_sec
    if MIN_CADENCE_RPM <= rpm <= MAX_CADENCE_RPM:
        return round(period_sec, 4)
    return None


def analyse_run(run_id, run):
    """smooth and find peaks for one run."""
    frame_indices = [s[0] for s in run]
    timestamps = [s[1] for s in run]
    raw = [s[2] for s in run]
    smoothed = smooth(raw)

    peak_indices = detect_peaks(smoothed, timestamps) if len(smoothed) >= MIN_PEAKS else []
    peak_method = "peak_detection"
    autocorr_period = None

    if len(peak_indices) < 2:
        autocorr_period = autocorrelation_period(timestamps, smoothed)
        if autocorr_period is not None:
            peak_method = "autocorrelation"

    peaks_out = [
        {
            "frame_idx": frame_indices[j],
            "timestamp": round(timestamps[j], 4),
            "angle": round(raw[j], 2),
        }
        for j in peak_indices
    ]

    duration_sec = round(timestamps[-1] - timestamps[0], 2) if len(timestamps) >= 2 else 0

    return {
        "run_id": run_id,
        "frame_idx_start": frame_indices[0],
        "frame_idx_end": frame_indices[-1],
        "frame_count": len(run),
        "duration_sec": duration_sec,
        "peaks": peaks_out,
        "peak_method": peak_method,
        "autocorr_period_sec": autocorr_period,
        "angle_series": [[round(t, 4), round(a, 2)] for t, a in zip(timestamps, smoothed)],
    }


def main():
    """build knee angle series, detect peaks per run, write knee_analysis.json."""
    if len(sys.argv) < 2:
        print("usage: python3 knee_analysis.py <video_keypoints.json>")
        sys.exit(1)

    kp_path = sys.argv[1]
    with open(kp_path) as f:
        data = json.load(f)

    stem = video_stem(kp_path, "_keypoints")
    out_dir = os.path.join("output_v2", stem)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, stem)
    out_path = base + "_knee_analysis.json"
    log_path = base + "_selection_log.json"

    direction, frame_direction_map = load_direction_map(log_path)
    _, _, knee_used = direction_to_joints(direction)

    if direction:
        unique_dirs = set(frame_direction_map.values()) - {None}
        if len(unique_dirs) > 1:
            print(f"direction: per-burst {sorted(unique_dirs)}  (knee selected per burst)")
        else:
            print(f"direction: {direction}  (using {knee_used} knee)")
    else:
        print("direction: unknown (defaulting to right knee)")

    series = []
    for entry in data["frames"]:
        kp = entry.get("keypoints", [])
        t = entry["timestamp"]
        idx = entry["frame_idx"]

        frame_dir = frame_direction_map.get(idx, direction)
        frame_primary, frame_fallback, _ = direction_to_joints(frame_dir)

        hip = get_xy(kp, frame_primary[0])
        knee = get_xy(kp, frame_primary[1])
        ankle = get_xy(kp, frame_primary[2])

        if not (hip and knee and ankle):
            hip = get_xy(kp, frame_fallback[0])
            knee = get_xy(kp, frame_fallback[1])
            ankle = get_xy(kp, frame_fallback[2])

        angle = calc_angle(hip, knee, ankle) if (hip and knee and ankle) else None
        if angle is not None:
            series.append((idx, t, angle))

    runs = split_runs(series, MAX_FRAME_GAP)

    run_results = [analyse_run(i, run) for i, run in enumerate(runs)]

    all_peaks = [p for r in run_results for p in r["peaks"]]
    all_angle_series = [pt for r in run_results for pt in r["angle_series"]]
    usable_runs = [r for r in run_results if len(r["peaks"]) >= 2]

    best_run_result = max(run_results, key=lambda r: r["frame_count"]) if run_results else {}
    top_method = best_run_result.get("peak_method", "peak_detection") if best_run_result else "peak_detection"
    top_autocorr = best_run_result.get("autocorr_period_sec") if best_run_result else None

    best_run_info = {
        "frame_idx_start": best_run_result.get("frame_idx_start"),
        "frame_idx_end": best_run_result.get("frame_idx_end"),
        "frame_count": best_run_result.get("frame_count", 0),
        "duration_sec": best_run_result.get("duration_sec", 0),
        "total_runs": len(runs),
    }

    total_time = sum(r["duration_sec"] for r in run_results)

    output = {
        "video": data["video"],
        "direction": direction,
        "knee_used": knee_used,
        "runs": run_results,
        "best_run": best_run_info,
        "peaks": all_peaks,
        "peak_method": top_method,
        "autocorr_period_sec": top_autocorr,
        "angle_series": all_angle_series,
        "metrics": {
            "frames_with_angle": len(series),
            "total_runs": len(runs),
            "usable_runs": len(usable_runs),
            "peaks_found": len(all_peaks),
            "time_span_sec": round(total_time, 2),
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    m = output["metrics"]
    print(f"frames with knee angle: {m['frames_with_angle']}")
    print(f"runs (total / usable): {m['total_runs']} / {m['usable_runs']}")
    for r in run_results:
        n_peaks = len(r["peaks"])
        usable  = "ok" if n_peaks >= 2 else "-"
        print(f"  Run {r['run_id']}  [{usable}]  : frames {r['frame_idx_start']}-{r['frame_idx_end']}  ({r['frame_count']} frames, {r['duration_sec']}s)  {n_peaks} peaks  [{r['peak_method']}]")
    print(f"total peaks: {m['peaks_found']}")
    print(f"knee analysis saved: {out_path}")


if __name__ == "__main__":
    main()
