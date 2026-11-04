import numpy as np
from reps import rep_stats

for quality in ["good", "bad"]:
    data = np.load(f"output/subject_004_squat_{quality}_side.npz")
    reps = rep_stats(data["landmarks"], data["fps"], data["width"], data["height"])

    for i, rep in enumerate(reps):
        print(f"{quality} rep {i + 1}: down {rep['down']:.2f} s, up {rep['up']:.2f} s, thigh {rep['thigh']:.1f}°")