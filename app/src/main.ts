import "./style.css";
import { createLandmarker, extractLandmarks } from "./pose";
import { repStats } from "./reps";
import { loadDepthModel, shallowProbability } from "./model";
import { drawSkeleton } from "./draw";

const fileInput = document.querySelector<HTMLInputElement>("#file")!;
const video = document.querySelector<HTMLVideoElement>("#video")!;
const canvas = document.querySelector<HTMLCanvasElement>("#overlay")!;
const button = document.querySelector<HTMLButtonElement>("#analyze")!;
const progressText = document.querySelector<HTMLElement>("#progress-text")!;

let loopId = 0;

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  video.cancelVideoFrameCallback(loopId);
  canvas.getContext("2d")!.clearRect(0, 0, canvas.width, canvas.height);
  progressText.textContent = "";

  video.src = URL.createObjectURL(file);
});

button.addEventListener("click", async () => {
  if (!video.src || isNaN(video.duration)) return;

  button.disabled = true;
  progressText.textContent = "Loading model…";
  const landmarker = await createLandmarker("full");

  const data = await extractLandmarks(landmarker, video, 30, (f) => {
    progressText.textContent = `Analyzing… ${Math.round(f * 100)}%`;
  });
  landmarker.close();
  button.disabled = false;
  progressText.textContent = `Done: ${data.landmarks.length} frames`;

  const reps = repStats(data.landmarks, data.fps, data.width, data.height);
  const model = await loadDepthModel();
  reps.forEach((rep, i) => {
    console.log(`rep ${i + 1}: ${Math.round(shallowProbability(model, rep) * 100)}% shallow`);
  });

  video.cancelVideoFrameCallback(loopId);

  function drawLoop() {
  const i = Math.round(video.currentTime * data.fps);
  const rep = reps.find((r) => i >= r.start && i <= r.end);
  const shallow =
    rep !== undefined &&
    Math.abs(i - rep.bottom) < data.fps * 0.5 &&
    shallowProbability(model, rep) > 0.5;
  drawSkeleton(canvas, data.landmarks[i], data.width, data.height, shallow);
  loopId = video.requestVideoFrameCallback(drawLoop);
}

  loopId = video.requestVideoFrameCallback(drawLoop);
  video.currentTime = 0;
  console.log("drawing frame", Math.round(video.currentTime * data.fps));
});