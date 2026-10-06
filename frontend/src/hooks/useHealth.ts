import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { HealthResponse } from "../api/types";

const POLL_INTERVAL_MS = 5000;

interface HealthState {
  health: HealthResponse | null;
  unreachable: boolean;
  attempts: number;
}

/** Polls /health every 5 s until the engine reports "ready"; the Space may be waking from sleep. */
export function useHealth() {
  const [state, setState] = useState<HealthState>({ health: null, unreachable: false, attempts: 0 });
  const timerRef = useRef<number | null>(null);
  const activeRef = useRef(true);

  const check = useCallback(async () => {
    try {
      const health = await api.health();
      if (!activeRef.current) return;
      setState((previous) => ({ health, unreachable: false, attempts: previous.attempts + 1 }));
      if (health.status !== "ready") {
        timerRef.current = window.setTimeout(check, POLL_INTERVAL_MS);
      }
    } catch {
      if (!activeRef.current) return;
      setState((previous) => ({ ...previous, unreachable: true, attempts: previous.attempts + 1 }));
      timerRef.current = window.setTimeout(check, POLL_INTERVAL_MS);
    }
  }, []);

  useEffect(() => {
    activeRef.current = true;
    void check();
    return () => {
      activeRef.current = false;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, [check]);

  return state;
}
