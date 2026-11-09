import "./style.css";
import { createLandmarker, extractLandmarks } from "./pose";
import { repStats } from "./reps";
import { loadDepthModel, shallowProbability } from "./model";

const fileInput = document.querySelector<HTMLInputElement>("#file")!;
const video = document.querySelector<HTMLVideoElement>("#video")!;
const button = document.querySelector<HTMLButtonElement>("#analyze")!;
const progressText = document.querySelector<HTMLElement>("#progress-text")!;

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) video.src = URL.createObjectURL(file);
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
  console.log(data);
  const reps = repStats(data.landmarks, data.fps, data.width, data.height);
  console.table(reps);

  const model = await loadDepthModel();
  reps.forEach((rep, i) => {
    console.log(`rep ${i + 1}: ${Math.round(shallowProbability(model, rep) * 100)}% shallow`);
  });
});