import { useEffect, type RefObject } from "react";
import type { ExamSocket } from "../api/examSocket";

const FRAME_WIDTH = 640;
const FRAME_INTERVAL_MS = 45;
const JPEG_QUALITY = 0.8;

/** Sends unmirrored JPEG frames (about 22 fps) while active; frames are dropped when the socket is congested. */
export function useFrameStreamer(videoRef: RefObject<HTMLVideoElement>, socket: ExamSocket | null, active: boolean) {
  useEffect(() => {
    if (!active || !socket) return undefined;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return undefined;
    const timer = window.setInterval(() => {
      const video = videoRef.current;
      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) return;
      const height = Math.round((FRAME_WIDTH * video.videoHeight) / video.videoWidth);
      if (canvas.width !== FRAME_WIDTH || canvas.height !== height) {
        canvas.width = FRAME_WIDTH;
        canvas.height = height;
      }
      context.drawImage(video, 0, 0, FRAME_WIDTH, height);
      socket.sendFrame(canvas.toDataURL("image/jpeg", JPEG_QUALITY));
    }, FRAME_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [videoRef, socket, active]);
}
