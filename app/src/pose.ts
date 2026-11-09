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
  onProgress?: (fraction: number) => void,
): Promise<PoseData> {
  await seek(video, 0);
  const empty = Array.from({ length: 33 }, () => [NaN, NaN, NaN, NaN]);
  const landmarks: number[][][] = [];
  const times: number[] = [];

    await new Promise<void>((resolve) => {
    let callbackId = 0;

    const onFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
      video.pause();
      const t = meta.mediaTime;

      if (!times.length || t > times[times.length - 1]) {
        const result = landmarker.detectForVideo(video, Math.round(t * 1000));
        const person = result.landmarks[0];
        landmarks.push(person ? person.map((lm) => [lm.x, lm.y, lm.z, lm.visibility ?? NaN]) : empty);
        times.push(t);
        onProgress?.(Math.min(1, t / video.duration));
      }

      callbackId = video.requestVideoFrameCallback(onFrame);
      video.play();
    };

    video.addEventListener(
      "ended",
      () => {
        video.cancelVideoFrameCallback(callbackId);
        resolve();
      },
      { once: true },
    );

    callbackId = video.requestVideoFrameCallback(onFrame);
    video.play();
  });

  const gaps = times.slice(1).map((t, i) => t - times[i]).sort((a, b) => a - b);
  const fps = gaps.length ? 1 / gaps[Math.floor(gaps.length / 2)] : 30;

  return { landmarks, fps, width: video.videoWidth, height: video.videoHeight };
}