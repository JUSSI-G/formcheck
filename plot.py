import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

good = np.load("output/subject_004_squat_good_side.npy")
bad = np.load("output/subject_004_squat_bad_side.npy")

plt.plot(good[:, 23, 1], label="good")
plt.plot(bad[:, 23, 1], label="bad")

y = good[:, 23, 1]
peaks, props = find_peaks(y, prominence=0.05)
print(len(peaks), peaks)

y_b = bad[:, 23, 1]
b_peaks, b_props = find_peaks(y_b, prominence=0.05)
print(len(b_peaks), b_peaks)

plt.gca().invert_yaxis() 
plt.xlabel("frame")
plt.ylabel("hip height (y)")
plt.legend()
plt.show()
