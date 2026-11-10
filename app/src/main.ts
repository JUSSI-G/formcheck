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
const repList = document.querySelector<HTMLTableSectionElement>("#rep-list")!;
const exportButton = document.querySelector<HTMLButtonElement>("#export")!;
const sampleButton = document.querySelector<HTMLButtonElement>("#sample")!;
const progressBar = document.querySelector<HTMLElement>("#progress")!;

function setProgress(text: string, fraction: number) {
  progressText.textContent = text;
  const percent = Math.round(Math.min(Math.max(fraction, 0), 1) * 100);
  progressBar.style.setProperty("--progress", `${percent}%`);
  progressBar.setAttribute("aria-valuenow", String(percent));
}

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

    const item = document.createElement("tr");
    item.className = shallow ? "rep shallow" : "rep ok";
    item.tabIndex = 0;

    const cells = [
      `${i + 1}${rep.cut_off ? "*" : ""}`,
      `${shallow ? "✗ Shallow" : "✓ Deep enough"}`,
      Math.round(deepest - rep.thigh) <= 1 ? "deepest" : `${Math.round(deepest - rep.thigh)}° shallower`,
      `${rep.down.toFixed(1)} s`,
      `${rep.up.toFixed(1)} s`,
    ];
    cells.forEach((text, c) => {
      const cell = document.createElement("td");
      cell.textContent = text;
      if (c === 1) cell.className = "mark";
      if (c >= 3) cell.className = "num";
      item.append(cell);
    });
    if (rep.cut_off) item.title = "Possibly cut off by the start or end of the video";

    const select = () => {
      repList.querySelector(".selected")?.classList.remove("selected");
      item.classList.add("selected");
      video.currentTime = rep.bottom / fps;
    };
    item.addEventListener("click", select);
    item.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        select();
      }
    });

    repList.append(item);
  });

  const anyCutOff = reps.some((r) => r.cut_off);
  summary.textContent = reps.length
    ? `Depth looks good on ${deepEnough} of ${reps.length} reps${anyCutOff ? " (* possibly cut off)" : ""}`
    : "No complete reps found. Is the whole body visible from the side?";
  results.hidden = false;
}

function loadVideo(src: string) {
  video.cancelVideoFrameCallback(loopId);
  canvas.getContext("2d")!.clearRect(0, 0, canvas.width, canvas.height);
  setProgress("", 0);
  results.hidden = true;
  repList.replaceChildren();
  lastAnalysis = null;

  video.src = src;
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  loadVideo(URL.createObjectURL(file));
});

sampleButton.addEventListener("click", () => {
  fileInput.value = "";
  loadVideo("sample.mp4");
  video.addEventListener("loadedmetadata", () => analyze(), { once: true });
});

button.addEventListener("click", () => analyze());

async function analyze() {
  if (!video.src || isNaN(video.duration)) return;
  video.cancelVideoFrameCallback(loopId);
  button.disabled = true;
  sampleButton.disabled = true;
  setProgress("Loading model…", 0);
  const landmarker = await createLandmarker("full");

  const data = await extractLandmarks(landmarker, video, (f) => {
    setProgress(`Analyzing… ${Math.round(f * 100)}%`, f);
  });

  landmarker.close();
  button.disabled = false;
  sampleButton.disabled = false;
  setProgress(`Done: ${data.landmarks.length} frames`, 1);

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
}

exportButton.addEventListener("click", async () => {
  if (!lastAnalysis) return;
  exportButton.disabled = true;

  const { blob, extension } = await exportVideo(
    video,
    lastAnalysis.data,
    lastAnalysis.reps,
    lastAnalysis.labels,
    (f) => {
      setProgress(`Creating video… ${Math.round(f * 100)}%`, f);
    },
  );

  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `formcheck.${extension}`;
  link.click();

  setProgress("Video ready", 1);
  exportButton.disabled = false;
});