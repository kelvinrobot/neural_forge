import { AnimatePresence, motion } from "framer-motion";
import { useEffect, type ReactNode, type RefObject } from "react";
import type { StepView } from "../../api/types";
import type { OverlayHighlight } from "../../hooks/useLandmarkOverlay";
import { useExamStore } from "../../state/examStore";
import { useSettingsStore } from "../../state/settingsStore";
import { Icon } from "../ui/Icon";
import { ActivityLayer } from "./ActivityLayer";
import { CountdownRing } from "./CountdownRing";
import { FramingGuide } from "./FramingGuide";
import { LandmarkOverlay } from "./LandmarkOverlay";
import { QualityBanner } from "./QualityBanner";

interface VideoStageProps {
  videoRef: RefObject<HTMLVideoElement>;
  stream: MediaStream | null;
  examId: string;
  view: StepView | null;
  highlight: OverlayHighlight;
  stepTitle: string;
  overlayEnabled: boolean;
  children?: ReactNode;
}

export function VideoStage({ videoRef, stream, examId, view, highlight, stepTitle, overlayEnabled, children }: VideoStageProps) {
  const qualityCode = useExamStore((state) => state.qualityCode);
  const qualityOk = useExamStore((state) => state.qualityOk);
  const countdown = useExamStore((state) => state.countdown);
  const phase = useExamStore((state) => state.phase);
  const paused = useExamStore((state) => state.paused);
  const voiceEnabled = useSettingsStore((state) => state.voiceEnabled);
  const setVoiceEnabled = useSettingsStore((state) => state.setVoiceEnabled);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = stream;
    if (stream) void video.play().catch(() => undefined);
  }, [stream, videoRef]);

  return (
    <div className="relative aspect-[3/4] w-full overflow-hidden rounded-2xl bg-ink-950 shadow-soft sm:aspect-video">
      <video ref={videoRef} className="absolute inset-0 h-full w-full -scale-x-100 object-cover" playsInline muted autoPlay aria-label="Live camera view" />
      <LandmarkOverlay videoRef={videoRef} examId={examId} view={view} highlight={highlight} enabled={overlayEnabled} />
      {view && <FramingGuide view={view} ok={qualityOk} />}
      <ActivityLayer stepTitle={stepTitle} />
      <QualityBanner code={phase === "countdown" ? null : qualityCode} />
      <CountdownRing value={phase === "countdown" ? countdown : null} />
      <AnimatePresence>
        {paused && (
          <motion.div
            className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-ink-950/70 text-white"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <Icon name="pause" size={48} />
            <span className="text-lg font-semibold">Paused</span>
          </motion.div>
        )}
      </AnimatePresence>
      <button
        type="button"
        onClick={() => setVoiceEnabled(!voiceEnabled)}
        className="absolute bottom-3 right-3 rounded-full bg-black/55 p-2.5 text-white backdrop-blur hover:bg-black/70 focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
        aria-label={voiceEnabled ? "Mute voice guidance" : "Unmute voice guidance"}
        aria-pressed={!voiceEnabled}
      >
        <Icon name={voiceEnabled ? "speaker" : "speakerOff"} />
      </button>
      {children}
    </div>
  );
}
