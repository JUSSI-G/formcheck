import mediapipe as mp
import cv2


models = {
    "full":  "models/pose_landmarker_full.task",
    "heavy":  "models/pose_landmarker_heavy.task",
}

source = "data/Dataset Exercise-wise/bicep_curl/bad/diagonal/subject_001_bicep_curl_bad_diagonal.mp4"
vid = cv2.VideoCapture(source)
if not vid.isOpened():
    raise FileNotFoundError(f"Could not open the file")
fps = vid.get(cv2.CAP_PROP_FPS)
frame_index = 0
result = []

# Landmarke instance with video mode and the options
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(models["full"]),
    running_mode = mp.tasks.vision.RunningMode.VIDEO,
)

landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)


while True:
    ok, frame = vid.read()

    if not ok:
        print("End of video or error occurred.")
        break

    frame_ms = int(frame_index / fps * 1000)

    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker_result = landmarker.detect_for_video(mp_image, frame_ms)

    result.append(landmarker_result.pose_landmarks)

    frame_index += 1

vid.release()


print(len(result), "frames processed")
print(result[50])

