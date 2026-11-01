import numpy as np
import matplotlib.pyplot as plt

good = np.load("output/subject_004_squat_good_side.npy")
bad = np.load("output/subject_004_squat_bad_side.npy")

plt.plot(good[:, 23, 1], label="good")
plt.plot(bad[:, 24, 1], label="bad")

plt.gca().invert_yaxis() 
plt.xlabel("frame")
plt.ylabel("hip height (y)")
plt.legend()
plt.show()
