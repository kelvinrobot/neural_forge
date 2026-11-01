import { useEffect, useRef } from "react";

/** Runs `callback` now and then every `intervalMs` after it settles, while `active` is true. */
export function usePolling(callback: () => Promise<void>, intervalMs: number, active: boolean): void {
  const saved = useRef(callback);

  useEffect(() => {
    saved.current = callback;
  });

  useEffect(() => {
    if (!active) return undefined;
    let cancelled = false;
    let timer: number | null = null;
    const tick = async () => {
      try {
        await saved.current();
      } finally {
        if (!cancelled) timer = window.setTimeout(tick, intervalMs);
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [active, intervalMs]);
}
