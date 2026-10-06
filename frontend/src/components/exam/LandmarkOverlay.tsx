import { useRef, type RefObject } from "react";
import type { StepView } from "../../api/types";
import { useLandmarkOverlay, type OverlayHighlight } from "../../hooks/useLandmarkOverlay";

interface LandmarkOverlayProps {
  videoRef: RefObject<HTMLVideoElement>;
  examId: string;
  view: StepView | null;
  highlight: OverlayHighlight;
  enabled: boolean;
}

export function LandmarkOverlay({ videoRef, examId, view, highlight, enabled }: LandmarkOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const status = useLandmarkOverlay(videoRef, canvasRef, { examId, view, highlight, enabled });
  return (
    <>
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100 object-cover" aria-hidden="true" />
      {status === "loading" && (
        <span className="absolute bottom-3 right-3 rounded-full bg-black/50 px-2.5 py-1 text-[11px] font-medium text-white">
          Loading tracking…
        </span>
      )}
    </>
  );
}
