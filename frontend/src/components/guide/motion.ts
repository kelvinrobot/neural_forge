import { useAnimationFrame, useMotionValue, useReducedMotion, type MotionValue } from "framer-motion";
import { useRef } from "react";

/** A looping 0..1 progress value; frozen at `holdAt` when not playing or when reduced motion is requested. */
export function useLoopProgress(durationSeconds: number, playing: boolean, holdAt: number): MotionValue<number> {
  const progress = useMotionValue(holdAt);
  const startRef = useRef<number | null>(null);
  const reduceMotion = useReducedMotion();
  useAnimationFrame((time) => {
    if (!playing || reduceMotion) {
      startRef.current = null;
      if (progress.get() !== holdAt) progress.set(holdAt);
      return;
    }
    if (startRef.current === null) startRef.current = time - holdAt * durationSeconds * 1000;
    progress.set((((time - startRef.current) / 1000) / durationSeconds) % 1);
  });
  return progress;
}

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Piecewise interpolation between [time, value] pairs with smooth easing; times must ascend within 0..1. */
export function keys(p: number, frames: ReadonlyArray<readonly [number, number]>): number {
  if (p <= frames[0][0]) return frames[0][1];
  for (let index = 1; index < frames.length; index += 1) {
    const [time, value] = frames[index];
    const [previousTime, previousValue] = frames[index - 1];
    if (p <= time) {
      const span = time - previousTime;
      const t = span <= 0 ? 1 : (p - previousTime) / span;
      return previousValue + (value - previousValue) * smooth(t);
    }
  }
  return frames[frames.length - 1][1];
}

export function wave(p: number, cycles: number, phase = 0): number {
  return Math.sin((p * cycles + phase) * Math.PI * 2);
}

export function radians(degrees: number): number {
  return (degrees * Math.PI) / 180;
}

export function point(x: number, y: number): string {
  return `${x.toFixed(1)} ${y.toFixed(1)}`;
}

export const SKIN_TONES = {
  light: "#f1c7a8",
  warm: "#e2a982",
  medium: "#c68a5e",
  deep: "#8d5a3b",
} as const;

export type SkinTone = keyof typeof SKIN_TONES;

export function shade(hex: string, factor: number): string {
  const value = Number.parseInt(hex.slice(1), 16);
  const channel = (shift: number) => Math.max(0, Math.min(255, Math.round(((value >> shift) & 255) * factor)));
  return `rgb(${channel(16)}, ${channel(8)}, ${channel(0)})`;
}
