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


for i, bottom in enumerate(peaks):
    before = tops[tops < bottom]
    after = tops[bottom < tops]

    if len(before) == 0 or len(after) == 0:
        continue
    
    start = before[-1]
    end = after[0]

    down = (bottom - start) / fps
    up = (end - bottom) / fps

    l_hip_y= good[bottom, 23, 1]
    l_knee_y = good[bottom, 25, 1]  

    angle = thigh_angle(good, bottom)
    print(f"good rep {i + 1}: down {down:.2f} s, up {up:.2f} s, thigh {angle:.1f}°")



for i, bottom in enumerate(b_peaks):
    before = b_tops[b_tops < bottom]
    after = b_tops[bottom < b_tops]

    if len(before) == 0 or len(after) == 0:
        continue

    start = before[-1]
    end = after[0]

    down = (bottom - start) / fps
    up = (end - bottom) / fps

    l_hip_y = bad[bottom, 23, 1]
    l_knee_y = bad[bottom, 25, 1]

    angle = thigh_angle(bad, bottom)
    print(f"bad rep {i + 1}: down {down:.2f} s, up {up:.2f} s, thigh {angle:.1f}°")

plt.plot(y, label="good")
plt.plot(y_b, label="bad")
plt.plot(tops, y[tops], "x")
plt.plot(b_tops, y_b[b_tops], "x")
plt.plot(peaks, y[peaks], "o")
plt.plot(b_peaks, y_b[b_peaks], "o")
plt.gca().invert_yaxis() 
plt.xlabel("frame")
plt.ylabel("hip height (y)")
plt.legend()
plt.show()
