import numpy as np
import matplotlib.pyplot as plt

landmarks = np.load("output/subject_001_squat_good_side.npy")

plt.plot(landmarks[:, 23, 1], label="left hip")
plt.plot(landmarks[:, 24, 1], label="right hip")

plt.gca().invert_yaxis()   # so down in the plot = down in real life
plt.xlabel("frame")
plt.ylabel("hip height (y)")
plt.legend()
plt.show()

missing = np.isnan(landmarks[:, 23, 1]).mean()
print(f"{missing:.0%} of frames have no detection")