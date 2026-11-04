import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


fps = 15
width, height = 810, 1440
good = np.load("output/subject_004_squat_good_side.npy")
bad = np.load("output/subject_004_squat_bad_side.npy")

def gap_fill(y):
    frames = np.arange(len(y))
    ok = ~np.isnan(y)
    return np.interp(frames, frames[ok], y[ok])

y = gap_fill(good[:, 23, 1])
peaks, props = find_peaks(y, prominence=0.05, distance=20)
tops, _ = find_peaks(-y, prominence=0.05, distance=20)
print(len(peaks), peaks)

y_b = gap_fill(bad[:, 23, 1])
b_peaks, b_props = find_peaks(y_b, prominence=0.05, distance=20)
b_tops, _ = find_peaks(-y_b, prominence=0.05, distance=20)
print(len(b_peaks), b_peaks)

def thigh_angle(landmarks, frame):
    hx = landmarks[frame, 23, 0] * width
    hy = landmarks[frame, 23, 1] * height
    kx = landmarks[frame, 25, 0] * width
    ky = landmarks[frame, 25, 1] * height
    return np.degrees(np.arctan2(hy - ky, abs(kx - hx)))

def rep_stats(landmarks, fps):
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
            "thigh": thigh_angle(landmarks, bottom),
        })

    return reps


for name, landmarks in [("good", good), ("bad", bad)]:
    for i, rep in enumerate(rep_stats(landmarks, fps)):
        print(f"{name} rep {i + 1}: down {rep['down']:.2f} s, up {rep['up']:.2f} s, thigh {rep['thigh']:.1f}°")

