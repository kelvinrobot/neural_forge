import { useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus = "requesting" | "ready" | "denied" | "unavailable" | "insecure" | "error";

interface CameraState {
  stream: MediaStream | null;
  status: CameraStatus;
  message: string;
  devices: MediaDeviceInfo[];
}

function describeError(error: unknown): Pick<CameraState, "status" | "message"> {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return { status: "denied", message: "Camera access was blocked. Allow camera access in your browser settings and try again." };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return { status: "unavailable", message: "No camera was found. Connect a camera and try again." };
  }
  if (name === "NotReadableError") {
    return { status: "error", message: "The camera is being used by another application. Close it and try again." };
  }
  return { status: "error", message: "The camera could not be started. Please try again." };
}

/** Opens the selected camera and stops every track on unmount or when the device changes. */
export function useCamera(deviceId: string | null, enabled = true) {
  const [state, setState] = useState<CameraState>({ stream: null, status: "requesting", message: "", devices: [] });
  const [attempt, setAttempt] = useState(0);
  const streamRef = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setState((previous) => ({ ...previous, stream: null }));
  }, []);

  useEffect(() => {
    if (!enabled) return undefined;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setState((previous) => ({
        ...previous,
        status: "insecure",
        message: "The camera needs a secure (HTTPS) connection and a supported browser.",
      }));
      return undefined;
    }
    let cancelled = false;
    setState((previous) => ({ ...previous, status: "requesting", message: "" }));
    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
        },
      })
      .then(async (stream) => {
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput");
        if (!cancelled) setState({ stream, status: "ready", message: "", devices });
      })
      .catch((error: unknown) => {
        if (!cancelled) setState((previous) => ({ ...previous, stream: null, ...describeError(error) }));
      });
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [deviceId, enabled, attempt]);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);

  return { ...state, retry, stop };
}
