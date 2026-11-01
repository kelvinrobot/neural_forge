import { useCallback, useEffect, useState } from "react";

const RING_INTERVAL_MS = 2000;

function beep(context: AudioContext, start: number): void {
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  oscillator.type = "sine";
  oscillator.frequency.value = 880;
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(0.3, start + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.25);
  oscillator.connect(gain).connect(context.destination);
  oscillator.start(start);
  oscillator.stop(start + 0.3);
}

/** Browsers only allow audio after a user gesture, so `enable` must be called from a click. */
export function useRingtone(ringing: boolean) {
  const [context, setContext] = useState<AudioContext | null>(null);

  const enable = useCallback(async () => {
    const audio = context ?? new AudioContext();
    await audio.resume();
    setContext(audio);
    if ("Notification" in window && Notification.permission === "default") {
      await Notification.requestPermission();
    }
  }, [context]);

  useEffect(() => {
    if (!context || !ringing) return undefined;
    const ring = () => {
      beep(context, context.currentTime);
      beep(context, context.currentTime + 0.35);
    };
    ring();
    const timer = window.setInterval(ring, RING_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [context, ringing]);

  useEffect(() => () => void context?.close(), [context]);

  return { enabled: context !== null, enable };
}
