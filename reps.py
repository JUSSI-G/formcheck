import numpy as np
from scipy.signal import find_peaks

LEFT = {"name": "left", "ear": 7, "shoulder": 11, "hip": 23, "knee": 25, "ankle": 27, "heel": 29, "toe": 31}
RIGHT = {"name": "right", "ear": 8, "shoulder": 12, "hip": 24, "knee": 26, "ankle": 28, "heel": 30, "toe": 32}


def gap_fill(y):
    frames = np.arange(len(y))
    ok = ~np.isnan(y)
    return np.interp(frames, frames[ok], y[ok])


def visible_side(landmarks):
    left = np.nanmean(landmarks[:, [23, 25, 27], 3])
    right = np.nanmean(landmarks[:, [24, 26, 28], 3])
    return LEFT if left >= right else RIGHT


def point(landmarks, frame, index, width, height):
    return landmarks[frame, index, :2] * np.array([width, height])


def joint_angle(landmarks, frame, a, b, c, width, height):
    pa = point(landmarks, frame, a, width, height)
    pb = point(landmarks, frame, b, width, height)
    pc = point(landmarks, frame, c, width, height)
    v1 = pa - pb
    v2 = pc - pb
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def thigh_angle(landmarks, frame, side, width, height):
    h = point(landmarks, frame, side["hip"], width, height)
    k = point(landmarks, frame, side["knee"], width, height)
    return np.degrees(np.arctan2(h[1] - k[1], abs(k[0] - h[0])))


def torso_lean(landmarks, frame, side, width, height):
    s = point(landmarks, frame, side["shoulder"], width, height)
    h = point(landmarks, frame, side["hip"], width, height)
    return np.degrees(np.arctan2(abs(s[0] - h[0]), h[1] - s[1]))


def knee_forward(landmarks, frame, side, width, height):
    k = point(landmarks, frame, side["knee"], width, height)
    a = point(landmarks, frame, side["ankle"], width, height)
    heel = point(landmarks, frame, side["heel"], width, height)
    toe = point(landmarks, frame, side["toe"], width, height)
    facing = np.sign(toe[0] - heel[0])
    return (k[0] - a[0]) * facing / np.linalg.norm(k - a)


def foot_angle(landmarks, frame, side, width, height, window=2):
    first = max(0, frame - window)
    last = min(len(landmarks) - 1, frame + window)

    angles = []
    for f in range(first, last + 1):
        heel = point(landmarks, f, side["heel"], width, height)
        toe = point(landmarks, f, side["toe"], width, height)
        angles.append(np.degrees(np.arctan2(heel[1] - toe[1], abs(toe[0] - heel[0]))))

    return np.nanmedian(angles)


def heel_lift(landmarks, start, bottom, side, width, height):
    standing = foot_angle(landmarks, start, side, width, height)
    squatting = foot_angle(landmarks, bottom, side, width, height)
    return standing - squatting

def head_forward(landmarks, frame, side, width, height):
    ear = point(landmarks, frame, side["ear"], width, height)
    s = point(landmarks, frame, side["shoulder"], width, height)
    h = point(landmarks, frame, side["hip"], width, height)
    heel = point(landmarks, frame, side["heel"], width, height)
    toe = point(landmarks, frame, side["toe"], width, height)
    facing = np.sign(toe[0] - heel[0])
    return (ear[0] - s[0]) * facing / np.linalg.norm(s - h)


def rep_stats(landmarks, fps, width, height):
    side = visible_side(landmarks)
    y = gap_fill(landmarks[:, side["hip"], 1])
    distance = int(1.3 * fps)
    peaks, _ = find_peaks(y, prominence=0.05, distance=distance)
    tops, _ = find_peaks(-y, prominence=0.05, distance=distance)

    reps = []
    for bottom in peaks:
        before = tops[tops < bottom]
        after = tops[tops > bottom]

        if len(before) == 0 or len(after) == 0:
            continue

        start = before[-1]
        end = after[0]

        rep = {
            "side": side["name"],
            "start": start,
            "bottom": bottom,
            "end": end,
            "down": (bottom - start) / fps,
            "up": (end - bottom) / fps,
            "thigh": thigh_angle(landmarks, bottom, side, width, height),
            "knee": joint_angle(landmarks, bottom, side["hip"], side["knee"], side["ankle"], width, height),
            "hip": joint_angle(landmarks, bottom, side["shoulder"], side["hip"], side["knee"], width, height),
            "lean": torso_lean(landmarks, bottom, side, width, height),
            "knee_fwd": knee_forward(landmarks, bottom, side, width, height),
            "heel_lift": heel_lift(landmarks, start, bottom, side, width, height),
            "lockout": joint_angle(landmarks, start, side["hip"], side["knee"], side["ankle"], width, height),
            "rom": joint_angle(landmarks, start, side["hip"], side["knee"], side["ankle"], width, height)
                   - joint_angle(landmarks, bottom, side["hip"], side["knee"], side["ankle"], width, height),
            "lean_change": torso_lean(landmarks, bottom, side, width, height)
                    - torso_lean(landmarks, start, side, width, height),
            "head_forward": head_forward(landmarks, bottom, side, width, height),
        }

        if reps and reps[-1]["start"] == start and reps[-1]["end"] == end:
            if y[bottom] > y[reps[-1]["bottom"]]:
                reps[-1] = rep
            continue

        reps.append(rep)

    return reps