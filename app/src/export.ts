import type { PoseData } from "./pose";
import type { Rep } from "./reps";
import { drawSkeleton } from "./draw";

export interface RepLabel {
  title: string;
  detail: string;
  shallow: boolean;
}

function pickMimeType(): string {
  const types = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"];
  return types.find((t) => MediaRecorder.isTypeSupported(t)) ?? "";
}

export async function exportVideo(
  video: HTMLVideoElement,
  data: PoseData,
  reps: Rep[],
  labels: RepLabel[],
  onProgress?: (fraction: number) => void,
): Promise<{ blob: Blob; extension: string }> {
  const canvas = document.createElement("canvas");
  canvas.width = data.width;
  canvas.height = data.height;
  const ctx = canvas.getContext("2d")!;
  const skeleton = document.createElement("canvas");
  const fontSize = Math.round(data.width / 24);

  const mimeType = pickMimeType();
  const recorder = new MediaRecorder(
    canvas.captureStream(30),
    mimeType ? { mimeType, videoBitsPerSecond: 5_000_000 } : undefined,
  );
  const chunks: Blob[] = [];
  recorder.ondataavailable = (e) => {
    if (e.data.size) chunks.push(e.data);
  };
  const stopped = new Promise<void>((resolve) => (recorder.onstop = () => resolve()));

  function drawFrame(time: number) {
    const i = Math.round(time * data.fps);
    const r = reps.findIndex((rep) => i >= rep.start && i <= rep.end);
    const rep = reps[r];
    const label = labels[r];

    ctx.drawImage(video, 0, 0, data.width, data.height);
    const shallow = rep !== undefined && label.shallow && Math.abs(i - rep.bottom) < data.fps * 0.5;
    drawSkeleton(skeleton, data.landmarks[i], data.width, data.height, shallow);
    ctx.drawImage(skeleton, 0, 0);

    if (!rep) return;
    const pad = fontSize * 0.5;
    ctx.textBaseline = "top";
    ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
    const titleWidth = ctx.measureText(label.title).width;
    ctx.font = `${fontSize * 0.8}px system-ui, sans-serif`;
    const detailWidth = ctx.measureText(label.detail).width;

    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(pad, pad, Math.max(titleWidth, detailWidth) + pad * 2, fontSize * 1.9 + pad * 2);

    ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
    ctx.fillStyle = label.shallow ? "#ff4d4d" : "#00e0a0";
    ctx.fillText(label.title, pad * 2, pad * 2);
    ctx.font = `${fontSize * 0.8}px system-ui, sans-serif`;
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label.detail, pad * 2, pad * 2 + fontSize * 1.1);
  }

  await new Promise<void>((resolve) => {
    video.addEventListener("seeked", () => resolve(), { once: true });
    video.currentTime = 0;
  });

  await new Promise<void>((resolve) => {
    let id = 0;
    const onFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
      drawFrame(meta.mediaTime);
      onProgress?.(Math.min(1, meta.mediaTime / video.duration));
      id = video.requestVideoFrameCallback(onFrame);
    };
    video.addEventListener(
      "ended",
      () => {
        video.cancelVideoFrameCallback(id);
        resolve();
      },
      { once: true },
    );
    drawFrame(0);
    recorder.start();
    id = video.requestVideoFrameCallback(onFrame);
    video.play();
  });

  recorder.stop();
  await stopped;
  const type = recorder.mimeType || "video/webm";
  return { blob: new Blob(chunks, { type }), extension: type.includes("mp4") ? "mp4" : "webm" };
}