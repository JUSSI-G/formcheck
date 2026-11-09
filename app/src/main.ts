import "./style.css";
import { createLandmarker, extractLandmarks, type PoseData } from "./pose";
import { repStats, type Rep } from "./reps";
import { loadDepthModel, shallowProbability, type DepthModel } from "./model";
import { drawSkeleton } from "./draw";
import { exportVideo, type RepLabel } from "./export";

const fileInput = document.querySelector<HTMLInputElement>("#file")!;
const video = document.querySelector<HTMLVideoElement>("#video")!;
const canvas = document.querySelector<HTMLCanvasElement>("#overlay")!;
const button = document.querySelector<HTMLButtonElement>("#analyze")!;
const progressText = document.querySelector<HTMLElement>("#progress-text")!;
const results = document.querySelector<HTMLElement>("#results")!;
const summary = document.querySelector<HTMLElement>("#summary")!;
const repList = document.querySelector<HTMLOListElement>("#rep-list")!;
const exportButton = document.querySelector<HTMLButtonElement>("#export")!;

let loopId = 0;
let lastAnalysis: { data: PoseData; reps: Rep[]; labels: RepLabel[] } | null = null;

function depthText(thigh: number, deepest: number): string {
  const diff = Math.round(deepest - thigh);
  return diff <= 1 ? "your deepest rep" : `${diff}° shallower than your deepest rep`;
}

function makeLabels(reps: Rep[], model: DepthModel): RepLabel[] {
  const deepest = Math.max(...reps.map((r) => r.thigh));
  return reps.map((rep, i) => {
    const shallow = shallowProbability(model, rep) > 0.5;
    return {
      title: `Rep ${i + 1}: ${shallow ? "Looks shallow" : "Depth looks good"}${rep.cut_off ? " (possibly cut off)" : ""}`,
      detail: depthText(rep.thigh, deepest),
      shallow,
    };
  });
}

function showResults(reps: Rep[], model: DepthModel, fps: number) {
  repList.replaceChildren();

  const deepest = Math.max(...reps.map((r) => r.thigh));
  let deepEnough = 0;

  reps.forEach((rep, i) => {
    const shallow = shallowProbability(model, rep) > 0.5;
    if (!shallow) deepEnough++;

    const item = document.createElement("li");
    item.className = shallow ? "rep shallow" : "rep ok";

    const title = document.createElement("strong");
    title.textContent = `Rep ${i + 1}: ${shallow ? "Looks shallow" : "Depth looks good"}${rep.cut_off ? " (possibly cut off)" : ""}`;
    const details = document.createElement("span");
    details.textContent = `${depthText(rep.thigh, deepest)} · down ${rep.down.toFixed(1)} s · up ${rep.up.toFixed(1)} s`;

    item.append(title, details);
    item.addEventListener("click", () => {
      video.currentTime = rep.bottom / fps;
    });

    repList.append(item);
  });

  summary.textContent = reps.length
    ? `Depth looks good on ${deepEnough} of ${reps.length} reps`
    : "No complete reps found. Is the whole body visible from the side?";
  results.hidden = false;
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  video.cancelVideoFrameCallback(loopId);
  canvas.getContext("2d")!.clearRect(0, 0, canvas.width, canvas.height);
  progressText.textContent = "";
  results.hidden = true;
  repList.replaceChildren();
  lastAnalysis = null;

  video.src = URL.createObjectURL(file);
});

button.addEventListener("click", async () => {
  if (!video.src || isNaN(video.duration)) return;
  video.cancelVideoFrameCallback(loopId);
  button.disabled = true;
  progressText.textContent = "Loading model…";
  const landmarker = await createLandmarker("full");

  const data = await extractLandmarks(landmarker, video, (f) => {
    progressText.textContent = `Analyzing… ${Math.round(f * 100)}%`;
  });

  landmarker.close();
  button.disabled = false;
  progressText.textContent = `Done: ${data.landmarks.length} frames`;

  const reps = repStats(data.landmarks, data.fps, data.width, data.height);
  const model = await loadDepthModel();
  showResults(reps, model, data.fps);
  lastAnalysis = { data, reps, labels: makeLabels(reps, model) };

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
});

exportButton.addEventListener("click", async () => {
  if (!lastAnalysis) return;
  exportButton.disabled = true;

  const { blob, extension } = await exportVideo(
    video,
    lastAnalysis.data,
    lastAnalysis.reps,
    lastAnalysis.labels,
    (f) => {
      progressText.textContent = `Creating video… ${Math.round(f * 100)}%`;
    },
  );

  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `formcheck.${extension}`;
  link.click();

  progressText.textContent = "Video ready";
  exportButton.disabled = false;
});