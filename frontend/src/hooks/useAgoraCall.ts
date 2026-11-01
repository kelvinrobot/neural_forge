import type { IAgoraRTCClient, ICameraVideoTrack, IMicrophoneAudioTrack, NetworkQuality } from "agora-rtc-sdk-ng";
import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

export const AGORA_APP_ID = import.meta.env.VITE_AGORA_APP_ID?.trim() ?? "";

export type CallState = "joining" | "joined" | "error";

export type ConnectionQuality = "unknown" | "good" | "fair" | "poor" | "down";

function qualityFrom(stats: NetworkQuality): ConnectionQuality {
  const worst = Math.max(stats.uplinkNetworkQuality, stats.downlinkNetworkQuality);
  if (worst === 0) return "unknown";
  if (worst <= 2) return "good";
  if (worst <= 4) return "fair";
  return worst === 5 ? "poor" : "down";
}

function describeError(error: unknown): string {
  const code = typeof error === "object" && error !== null && "code" in error ? String((error as { code: unknown }).code) : "";
  const message = error instanceof Error ? error.message : "";
  if (code === "PERMISSION_DENIED" || /permission/i.test(message)) {
    return "Camera or microphone access was denied. Allow access in your browser settings, then reload this page.";
  }
  if (code === "DEVICE_NOT_FOUND") return "No camera or microphone was found on this device.";
  if (code === "NOT_READABLE") return "The camera or microphone is being used by another app. Close it and reload this page.";
  return "The video call could not connect. Check your internet connection and try again.";
}

interface AgoraCallOptions {
  channel: string;
  localRef: RefObject<HTMLDivElement>;
  remoteRef: RefObject<HTMLDivElement>;
}

export function useAgoraCall({ channel, localRef, remoteRef }: AgoraCallOptions) {
  const [state, setState] = useState<CallState>("joining");
  const [error, setError] = useState<string | null>(null);
  const [remoteVideo, setRemoteVideo] = useState(false);
  const [remotePresent, setRemotePresent] = useState(false);
  const [micOn, setMicOn] = useState(true);
  const [cameraOn, setCameraOn] = useState(true);
  const [quality, setQuality] = useState<ConnectionQuality>("unknown");
  const tracksRef = useRef<[IMicrophoneAudioTrack, ICameraVideoTrack] | null>(null);

  useEffect(() => {
    if (!AGORA_APP_ID) return undefined;
    let cancelled = false;
    let client: IAgoraRTCClient | null = null;

    const release = () => {
      tracksRef.current?.forEach((track) => {
        track.stop();
        track.close();
      });
      tracksRef.current = null;
      if (client) {
        client.removeAllListeners();
        void client.leave().catch(() => undefined);
        client = null;
      }
    };

    const start = async () => {
      try {
        const { default: AgoraRTC } = await import("agora-rtc-sdk-ng");
        if (cancelled) return;
        AgoraRTC.setLogLevel(3);
        const tracks = await AgoraRTC.createMicrophoneAndCameraTracks(undefined, { encoderConfig: { width: 640, height: 480 } });
        if (cancelled) {
          tracks.forEach((track) => track.close());
          return;
        }
        tracksRef.current = tracks;
        if (localRef.current) tracks[1].play(localRef.current);

        const rtc = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });
        client = rtc;
        rtc.on("user-published", async (user, mediaType) => {
          await rtc.subscribe(user, mediaType);
          setRemotePresent(true);
          if (mediaType === "video" && remoteRef.current) {
            user.videoTrack?.play(remoteRef.current);
            setRemoteVideo(true);
          }
          if (mediaType === "audio") user.audioTrack?.play();
        });
        rtc.on("user-joined", () => setRemotePresent(true));
        rtc.on("user-unpublished", (_user, mediaType) => {
          if (mediaType === "video") setRemoteVideo(false);
        });
        rtc.on("user-left", () => {
          setRemoteVideo(false);
          setRemotePresent(false);
        });
        rtc.on("network-quality", (stats) => setQuality(qualityFrom(stats)));
        rtc.on("connection-state-change", (current, _previous, reason) => {
          if (current === "DISCONNECTED" && reason !== "LEAVE" && !cancelled) {
            setError("The video connection was lost. Check your internet connection.");
            setState("error");
          }
        });

        await rtc.join(AGORA_APP_ID, channel, null, null);
        if (cancelled) return;
        await rtc.publish(tracks);
        if (!cancelled) setState("joined");
      } catch (caught) {
        if (cancelled) return;
        setError(describeError(caught));
        setState("error");
        release();
      }
    };

    void start();
    window.addEventListener("pagehide", release);
    return () => {
      cancelled = true;
      window.removeEventListener("pagehide", release);
      release();
    };
  }, [channel, localRef, remoteRef]);

  const toggleMic = useCallback(() => {
    const microphone = tracksRef.current?.[0];
    if (!microphone) return;
    void microphone.setMuted(micOn);
    setMicOn(!micOn);
  }, [micOn]);

  const toggleCamera = useCallback(() => {
    const camera = tracksRef.current?.[1];
    if (!camera) return;
    void camera.setEnabled(!cameraOn);
    setCameraOn(!cameraOn);
  }, [cameraOn]);

  return { state, error, remoteVideo, remotePresent, micOn, cameraOn, quality, toggleMic, toggleCamera };
}
