import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Change later
fps = 15

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

for i, bottom in enumerate(peaks):
    before = tops[tops < bottom]
    after = tops[bottom < tops]

    if len(before) == 0 or len(after) == 0:
        continue

    start = before[-1]
    end = after[0]
    print(f"rep {i + 1}: start {start}, bottom {bottom}, end {end}")

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
