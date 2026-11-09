import type { Rep } from "./reps";

export interface DepthModel {
  features: (keyof Rep)[];
  mean: number[];
  scale: number[];
  weights: number[];
  intercept: number;
}

export async function loadDepthModel(): Promise<DepthModel> {
  const response = await fetch("./depth_model.json");
  return response.json();
}

export function shallowProbability(model: DepthModel, rep: Rep): number {
  let z = model.intercept;
  model.features.forEach((name, i) => {
    const scaled = ((rep[name] as number) - model.mean[i]) / model.scale[i];
    z += model.weights[i] * scaled;
  });
  return 1 / (1 + Math.exp(-z));
}