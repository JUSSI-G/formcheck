export type Landmarks = number[][][];

interface Side {
  name: "left" | "right";
  ear: number;
  shoulder: number;
  hip: number;
  knee: number;
  ankle: number;
  heel: number;
  toe: number;
}

export interface Rep {
  side: "left" | "right";
  start: number;
  bottom: number;
  end: number;
  down: number;
  up: number;
  thigh: number;
  knee: number;
  hip: number;
  lean: number;
  knee_fwd: number;
  heel_lift: number;
  lockout: number;
  rom: number;
  lean_change: number;
  head_forward: number;
}

const LEFT: Side = { name: "left", ear: 7, shoulder: 11, hip: 23, knee: 25, ankle: 27, heel: 29, toe: 31 };
const RIGHT: Side = { name: "right", ear: 8, shoulder: 12, hip: 24, knee: 26, ankle: 28, heel: 30, toe: 32 };

const degrees = (radians: number) => (radians * 180) / Math.PI;

function nanMean(values: number[]): number {
  const ok = values.filter((v) => !Number.isNaN(v));
  return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : NaN;
}

function nanMedian(values: number[]): number {
  const ok = values.filter((v) => !Number.isNaN(v)).sort((a, b) => a - b);
  if (!ok.length) return NaN;
  const mid = Math.floor(ok.length / 2);
  return ok.length % 2 ? ok[mid] : (ok[mid - 1] + ok[mid]) / 2;
}

export function gapFill(y: number[]): number[] {
  const known = y.map((v, i) => (Number.isNaN(v) ? -1 : i)).filter((i) => i >= 0);
  if (!known.length) return y.slice();

  return y.map((v, i) => {
    if (!Number.isNaN(v)) return v;
    if (i < known[0]) return y[known[0]];
    if (i > known[known.length - 1]) return y[known[known.length - 1]];

    let next = known.findIndex((k) => k > i);
    const right = known[next];
    const left = known[next - 1];
    const t = (i - left) / (right - left);
    return y[left] + t * (y[right] - y[left]);
  });
}

export function findPeaks(x: number[], prominence: number, distance: number): number[] {
  const n = x.length;
  let peaks: number[] = [];

  let i = 1;
  while (i < n - 1) {
    if (x[i - 1] < x[i]) {
      let ahead = i + 1;
      while (ahead < n - 1 && x[ahead] === x[i]) ahead++;
      if (x[ahead] < x[i]) {
        peaks.push(Math.floor((i + ahead - 1) / 2));
        i = ahead;
      }
    }
    i++;
  }

  const keep = peaks.map(() => true);
  const order = peaks.map((_, j) => j).sort((a, b) => x[peaks[a]] - x[peaks[b]]);
  for (let o = order.length - 1; o >= 0; o--) {
    const j = order[o];
    if (!keep[j]) continue;
    for (let k = j - 1; k >= 0 && peaks[j] - peaks[k] < distance; k--) keep[k] = false;
    for (let k = j + 1; k < peaks.length && peaks[k] - peaks[j] < distance; k++) keep[k] = false;
  }
  peaks = peaks.filter((_, j) => keep[j]);

  return peaks.filter((p) => {
    let leftMin = x[p];
    for (let k = p; k >= 0 && x[k] <= x[p]; k--) leftMin = Math.min(leftMin, x[k]);
    let rightMin = x[p];
    for (let k = p; k < n && x[k] <= x[p]; k++) rightMin = Math.min(rightMin, x[k]);
    return x[p] - Math.max(leftMin, rightMin) >= prominence;
  });
}

function visibleSide(landmarks: Landmarks): Side {
  const left = nanMean(landmarks.flatMap((f) => [f[23][3], f[25][3], f[27][3]]));
  const right = nanMean(landmarks.flatMap((f) => [f[24][3], f[26][3], f[28][3]]));
  return left >= right ? LEFT : RIGHT;
}

function point(landmarks: Landmarks, frame: number, index: number, width: number, height: number): [number, number] {
  const lm = landmarks[frame][index];
  return [lm[0] * width, lm[1] * height];
}

function jointAngle(landmarks: Landmarks, frame: number, a: number, b: number, c: number, width: number, height: number): number {
  const pa = point(landmarks, frame, a, width, height);
  const pb = point(landmarks, frame, b, width, height);
  const pc = point(landmarks, frame, c, width, height);
  const v1 = [pa[0] - pb[0], pa[1] - pb[1]];
  const v2 = [pc[0] - pb[0], pc[1] - pb[1]];
  const cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (Math.hypot(v1[0], v1[1]) * Math.hypot(v2[0], v2[1]));
  return degrees(Math.acos(Math.min(1, Math.max(-1, cos))));
}

function thighAngle(landmarks: Landmarks, frame: number, side: Side, width: number, height: number): number {
  const h = point(landmarks, frame, side.hip, width, height);
  const k = point(landmarks, frame, side.knee, width, height);
  return degrees(Math.atan2(h[1] - k[1], Math.abs(k[0] - h[0])));
}

function torsoLean(landmarks: Landmarks, frame: number, side: Side, width: number, height: number): number {
  const s = point(landmarks, frame, side.shoulder, width, height);
  const h = point(landmarks, frame, side.hip, width, height);
  return degrees(Math.atan2(Math.abs(s[0] - h[0]), h[1] - s[1]));
}

function facing(landmarks: Landmarks, frame: number, side: Side, width: number, height: number): number {
  const heel = point(landmarks, frame, side.heel, width, height);
  const toe = point(landmarks, frame, side.toe, width, height);
  return Math.sign(toe[0] - heel[0]);
}

function kneeForward(landmarks: Landmarks, frame: number, side: Side, width: number, height: number): number {
  const k = point(landmarks, frame, side.knee, width, height);
  const a = point(landmarks, frame, side.ankle, width, height);
  return ((k[0] - a[0]) * facing(landmarks, frame, side, width, height)) / Math.hypot(k[0] - a[0], k[1] - a[1]);
}

function footAngle(landmarks: Landmarks, frame: number, side: Side, width: number, height: number, window = 2): number {
  const first = Math.max(0, frame - window);
  const last = Math.min(landmarks.length - 1, frame + window);
  const angles: number[] = [];
  for (let f = first; f <= last; f++) {
    const heel = point(landmarks, f, side.heel, width, height);
    const toe = point(landmarks, f, side.toe, width, height);
    angles.push(degrees(Math.atan2(heel[1] - toe[1], Math.abs(toe[0] - heel[0]))));
  }
  return nanMedian(angles);
}

function heelLift(landmarks: Landmarks, start: number, bottom: number, side: Side, width: number, height: number): number {
  return footAngle(landmarks, start, side, width, height) - footAngle(landmarks, bottom, side, width, height);
}

function headForward(landmarks: Landmarks, frame: number, side: Side, width: number, height: number): number {
  const ear = point(landmarks, frame, side.ear, width, height);
  const s = point(landmarks, frame, side.shoulder, width, height);
  const h = point(landmarks, frame, side.hip, width, height);
  return ((ear[0] - s[0]) * facing(landmarks, frame, side, width, height)) / Math.hypot(s[0] - h[0], s[1] - h[1]);
}

export function repStats(landmarks: Landmarks, fps: number, width: number, height: number): Rep[] {
  const side = visibleSide(landmarks);
  const y = gapFill(landmarks.map((f) => f[side.hip][1]));
  const distance = Math.floor(1.3 * fps);
  const peaks = findPeaks(y, 0.05, distance);
  const tops = findPeaks(y.map((v) => -v), 0.05, distance);

  const reps: Rep[] = [];
  for (const bottom of peaks) {
    const before = tops.filter((t) => t < bottom);
    const after = tops.filter((t) => t > bottom);
    if (!before.length || !after.length) continue;

    const start = before[before.length - 1];
    const end = after[0];
    const kneeTop = jointAngle(landmarks, start, side.hip, side.knee, side.ankle, width, height);
    const kneeBottom = jointAngle(landmarks, bottom, side.hip, side.knee, side.ankle, width, height);

    const rep: Rep = {
      side: side.name,
      start,
      bottom,
      end,
      down: (bottom - start) / fps,
      up: (end - bottom) / fps,
      thigh: thighAngle(landmarks, bottom, side, width, height),
      knee: kneeBottom,
      hip: jointAngle(landmarks, bottom, side.shoulder, side.hip, side.knee, width, height),
      lean: torsoLean(landmarks, bottom, side, width, height),
      knee_fwd: kneeForward(landmarks, bottom, side, width, height),
      heel_lift: heelLift(landmarks, start, bottom, side, width, height),
      lockout: kneeTop,
      rom: kneeTop - kneeBottom,
      lean_change: torsoLean(landmarks, bottom, side, width, height) - torsoLean(landmarks, start, side, width, height),
      head_forward: headForward(landmarks, bottom, side, width, height),
    };

    const last = reps[reps.length - 1];
    if (last && last.start === start && last.end === end) {
      if (y[bottom] > y[last.bottom]) reps[reps.length - 1] = rep;
      continue;
    }

    reps.push(rep);
  }

  return reps;
}