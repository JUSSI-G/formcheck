"""Regenerate the README figures and demo GIF in docs/.

Run from the repo root after main.py and features.py:
    python scripts/make_figures.py
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reps import gap_fill, visible_side  # noqa: E402

DOCS = Path("docs")
GREEN = "#00a878"
RED = "#e04848"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def demo_gif():
    """docs/formcheck_1.mp4 (the app's own export) -> docs/demo.gif, ~12 fps, 360 px wide."""
    vid = cv2.VideoCapture(str(DOCS / "formcheck_1.mp4"))
    fps = vid.get(cv2.CAP_PROP_FPS)
    step = fps / 12
    frames, i, next_keep = [], 0, 0.0
    while True:
        ok, frame = vid.read()
        if not ok:
            break
        if i >= next_keep:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (360, round(h * 360 / w)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            next_keep += step
        i += 1
    vid.release()

    palette = frames[len(frames) // 3].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    gif[0].save(DOCS / "demo.gif", save_all=True, append_images=gif[1:],
                duration=round(1000 / 12), loop=0, optimize=True)


def rep_detection():
    """Hip height over time for subject 4, good vs. bad, with the peaks reps.py finds."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, quality in zip(axes, ["good", "bad"]):
        data = np.load(f"output/subject_004_squat_{quality}_side.npz")
        landmarks, fps = data["landmarks"], float(data["fps"])
        side = visible_side(landmarks)
        y = gap_fill(landmarks[:, side["hip"], 1])
        distance = int(1.3 * fps)
        bottoms, _ = find_peaks(y, prominence=0.05, distance=distance)
        tops, _ = find_peaks(-y, prominence=0.05, distance=distance)

        t = np.arange(len(y)) / fps
        ax.plot(t, y, color="#555", lw=1.4)
        ax.scatter(t[bottoms], y[bottoms], color=GREEN if quality == "good" else RED,
                   zorder=3, label="bottom of rep")
        ax.scatter(t[tops], y[tops], marker="x", color="#333", zorder=3, label="standing")
        ax.set_title(quality.capitalize())
        ax.set_xlabel("seconds")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("hip height\n(down = deeper)")
    axes[0].set_yticks([])
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(DOCS / "rep-detection.png")
    plt.close(fig)


def per_subject():
    """Accuracy per subject for the plain good/bad model, colored by each subject's main error."""
    features = ["thigh", "knee", "hip", "lean", "knee_fwd", "down", "up", "heel_lift"]
    df = pd.read_csv("output/reps_clean.csv").dropna(subset=features)
    df = df.merge(pd.read_csv("labels/errors.csv"), on="subject", how="left")

    model = make_pipeline(StandardScaler(), LogisticRegression())
    y = df["quality"] == "bad"
    pred = cross_val_predict(model, df[features], y, cv=GroupKFold(5), groups=df["subject"])
    df["correct"] = pred == y

    per = df.groupby("subject").agg(acc=("correct", "mean"), error=("error", "first")).sort_values("acc")
    colors = {"depth": GREEN, "back": "#4a78c2", "heels": "#e0a030", "unclear": "#aaaaaa"}

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([f"subject {s}" for s in per.index], per["acc"] * 100,
            color=[colors[e] for e in per["error"]])
    ax.axvline(50, color="#333", ls="--", lw=1)
    ax.text(51, -0.9, "coin flip", fontsize=8, color="#333")
    ax.set_xlabel("reps judged correctly (%)")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, [f"{k} error" for k in colors], title="main error in bad video",
              frameon=False, loc="lower right", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "per-subject.png")
    plt.close(fig)


def barbell_failure(frames=(205, 411, 514)):
    """A few bottom-of-rep frames from a barbell set, with the skeleton MediaPipe finds."""
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions("models/pose_landmarker_full.task"),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    vid = cv2.VideoCapture(str(DOCS / "barbell_failure.mp4"))
    fps = vid.get(cv2.CAP_PROP_FPS)

    connections = [(c.start, c.end) for c in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
                   if c.start >= 11 and c.end >= 11]
    picked, i = {}, 0
    while i <= max(frames):
        ok, frame = vid.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                             int(i / fps * 1000))
        if i in frames:
            h, w = rgb.shape[:2]
            img = rgb.copy()
            if result.pose_landmarks:
                pts = [(int(p.x * w), int(p.y * h)) for p in result.pose_landmarks[0]]
                for a, b in connections:
                    color = (224, 72, 72) if a >= 23 and b >= 23 else (0, 224, 160)
                    cv2.line(img, pts[a], pts[b], color, 4)
                for p in pts[11:]:
                    cv2.circle(img, p, 4, (255, 255, 255), -1)
                for hip in (23, 24):
                    cv2.circle(img, pts[hip], 14, (255, 220, 0), 3)
            picked[i] = img
        i += 1
    vid.release()
    landmarker.close()

    strip = np.hstack([picked[f] for f in frames if f in picked])
    strip = cv2.resize(strip, (strip.shape[1] // 2, strip.shape[0] // 2), interpolation=cv2.INTER_AREA)
    Image.fromarray(strip).save(DOCS / "barbell-failure.jpg", quality=85)


if __name__ == "__main__":
    demo_gif()
    rep_detection()
    per_subject()
    barbell_failure()
    for p in sorted(DOCS.glob("*.*")):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")
