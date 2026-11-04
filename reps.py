import numpy as np
from scipy.signal import find_peaks


def gap_fill(y):
    frames = np.arange(len(y))
    ok = ~np.isnan(y)
    return np.interp(frames, frames[ok], y[ok])

def thigh_angle(landmarks, frame, width, height):
    hx = landmarks[frame, 23, 0] * width
    hy = landmarks[frame, 23, 1] * height
    kx = landmarks[frame, 25, 0] * width
    ky = landmarks[frame, 25, 1] * height
    return np.degrees(np.arctan2(hy - ky, abs(kx - hx)))


def rep_stats(landmarks, fps, width, height):
    y = gap_fill(landmarks[:, 23, 1])
    peaks, _ = find_peaks(y, prominence=0.05, distance=20)
    tops, _ = find_peaks(-y, prominence=0.05, distance=20)

    reps = []
    for bottom in peaks:
        before = tops[tops < bottom]
        after = tops[tops > bottom]

        if len(before) == 0 or len(after) == 0:
            continue

        start = before[-1]
        end = after[0]

        reps.append({
            "start": start,
            "bottom": bottom,
            "end": end,
            "down": (bottom - start) / fps,
            "up": (end - bottom) / fps,
            "thigh": thigh_angle(landmarks, bottom, width, height),
        })

    return reps