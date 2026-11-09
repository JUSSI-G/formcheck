import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

export type Variant = "full" | "heavy";

export interface PoseData {
  landmarks: number[][][];
  fps: number;
  width: number;
  height: number;
}

export async function createLandmarker(variant: Variant = "full"): Promise<PoseLandmarker> {
  const vision = await FilesetResolver.forVisionTasks("./mediapipe/wasm");
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: `./models/pose_landmarker_${variant}.task`, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

function seek(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    video.addEventListener("seeked", () => resolve(), { once: true });
    video.currentTime = time;
  });
}

export async function extractLandmarks(
  landmarker: PoseLandmarker,
  video: HTMLVideoElement,
  fps = 30,
  onProgress?: (fraction: number) => void,
): Promise<PoseData> {
  const total = Math.floor(video.duration * fps);
  const empty = Array.from({ length: 33 }, () => [NaN, NaN, NaN, NaN]);
  const landmarks: number[][][] = [];

  for (let i = 0; i < total; i++) {
    await seek(video, i / fps);
    const result = landmarker.detectForVideo(video, Math.round((i / fps) * 1000));
    const person = result.landmarks[0];
    landmarks.push(person ? person.map((lm) => [lm.x, lm.y, lm.z, lm.visibility ?? NaN]) : empty);
    onProgress?.((i + 1) / total);
  }

  return { landmarks, fps, width: video.videoWidth, height: video.videoHeight };
}