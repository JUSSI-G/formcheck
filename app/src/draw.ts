import { DrawingUtils, PoseLandmarker } from "@mediapipe/tasks-vision";

type Connection = { start: number; end: number };

const BODY = PoseLandmarker.POSE_CONNECTIONS.filter((c) => c.start >= 11 && c.end >= 11);
const isLeg = (c: Connection) => c.start >= 23 && c.end >= 23;
const LEGS = BODY.filter(isLeg);
const UPPER = BODY.filter((c) => !isLeg(c));

export function drawSkeleton(
  canvas: HTMLCanvasElement,
  frame: number[][] | undefined,
  width: number,
  height: number,
  shallow = false,
) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, width, height);
  if (!frame || Number.isNaN(frame[23][0])) return;

  const landmarks = frame.map(([x, y, z, visibility]) => ({ x, y, z, visibility }));
  const utils = new DrawingUtils(ctx);

  utils.drawConnectors(landmarks, UPPER, { color: "#00e0a0", lineWidth: 4 });
  utils.drawConnectors(landmarks, LEGS, { color: shallow ? "#ff4d4d" : "#00e0a0", lineWidth: 4 });
  utils.drawLandmarks(landmarks.slice(11), { color: "#ffffff", radius: 3 });
}