import mediapipe as mp
import cv2
import numpy as np
import os
from pathlib import Path

models = {
    "full":  "models/pose_landmarker_full.task",
    "heavy": "models/pose_landmarker_heavy.task",
}


def extract_landmarks(path):
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(models["full"]),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    vid = cv2.VideoCapture(str(path))
    if not vid.isOpened():
        raise FileNotFoundError(f"Could not open {path}")
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    result = []

    while True:
        ok, frame = vid.read()
        if not ok:
            break

        frame_ms = int(frame_index / fps * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker_result = landmarker.detect_for_video(mp_image, frame_ms)
        result.append(landmarker_result.pose_landmarks)
        frame_index += 1

    vid.release()
    landmarker.close()

    frames = []
    for pose_landmarks in result:
        if len(pose_landmarks) == 0:
            frames.append(np.full((33, 4), np.nan))
        else:
            person = pose_landmarks[0]
            rows = []
            for i in person:
                rows.append([i.x, i.y, i.z, i.visibility])
            frames.append(rows)

    return np.array(frames)


folder = Path("data/Dataset Exercise-wise/squat/bad/side")
os.makedirs("output", exist_ok=True)

for path in sorted(folder.glob("*.mp4")):
    landmarks = extract_landmarks(path)
    missing = np.isnan(landmarks[:, 23, 1]).mean()
    print(f"{path.name}: {landmarks.shape[0]} frames, {missing:.0%} missing")
    np.save(f"output/{path.stem}.npy", landmarks)