import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, ApiError, websocketUrl } from "../api/client";
import { ExamSocket } from "../api/examSocket";
import type { ControlAction, ServerMessage } from "../api/types";
import { AnalysisScreen } from "../components/exam/AnalysisScreen";
import { CaptionBar } from "../components/exam/CaptionBar";
import { ControlBar } from "../components/exam/ControlBar";
import { StepPanel } from "../components/exam/StepPanel";
import { StepTimeline } from "../components/exam/StepTimeline";
import { VideoStage } from "../components/exam/VideoStage";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Icon } from "../components/ui/Icon";
import { Spinner } from "../components/ui/Spinner";
import { toast } from "../components/ui/Toast";
import { useCamera } from "../hooks/useCamera";
import { useFrameStreamer } from "../hooks/useFrameStreamer";
import type { OverlayHighlight } from "../hooks/useLandmarkOverlay";
import { useSpeech } from "../hooks/useSpeech";
import { useExamStore, type ExamPhase } from "../state/examStore";
import { useSettingsStore } from "../state/settingsStore";

const SILENT_TYPES = new Set(["step_progress", "analysis_progress", "pong"]);
const QUALITY_REPEAT_MS = 6000;
const ANALYSIS_PHASES: ExamPhase[] = ["finishing", "analysing", "report"];

function highlightFor(examId: string): OverlayHighlight {
  if (examId === "eyes") return "eyes";
  if (examId === "mouth") return "mouth";
  return "face";
}

function errorText(message: Extract<ServerMessage, { type: "error" }>): string {
  return message.display_text || `The examination reported an error (${message.code}).`;
}

export function ExamPage() {
  const { examId = "" } = useParams();
  const navigate = useNavigate();
  const session = useExamStore((state) => state.session);
  const exam = useExamStore((state) => state.exam);
  const patient = useExamStore((state) => state.patient);
  const phase = useExamStore((state) => state.phase);
  const paused = useExamStore((state) => state.paused);
  const steps = useExamStore((state) => state.steps);
  const stepStatus = useExamStore((state) => state.stepStatus);
  const currentStepId = useExamStore((state) => state.currentStepId);
  const caption = useExamStore((state) => state.caption);
  const areasCaptured = useExamStore((state) => state.areasCaptured);
  const analysisStage = useExamStore((state) => state.analysisStage);
  const applyMessage = useExamStore((state) => state.applyMessage);
  const setPhase = useExamStore((state) => state.setPhase);
  const setCaption = useExamStore((state) => state.setCaption);
  const tickCountdown = useExamStore((state) => state.tickCountdown);
  const countdown = useExamStore((state) => state.countdown);
  const beginSession = useExamStore((state) => state.beginSession);
  const cameraDeviceId = useSettingsStore((state) => state.cameraDeviceId);

  const [socket, setSocket] = useState<ExamSocket | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const qualitySpokenRef = useRef(new Map<string, number>());
  const countdownSpokenRef = useRef<{ value: number; at: number } | null>(null);

  const cameraWanted = phase !== "report" && !ANALYSIS_PHASES.includes(phase) && phase !== "connection_lost";
  const camera = useCamera(cameraDeviceId, cameraWanted);
  const { speak, cancel } = useSpeech(setCaption);
  const speakRef = useRef(speak);
  speakRef.current = speak;

  useEffect(() => {
    if (!session) navigate(`/exam/${encodeURIComponent(examId)}`, { replace: true });
  }, [session, examId, navigate]);

  const handleSpeech = useCallback(
    (message: ServerMessage) => {
      const text = message.speech_text?.trim();
      if (!text || SILENT_TYPES.has(message.type)) return;
      if (message.type === "quality_feedback") {
        const last = qualitySpokenRef.current.get(message.code) ?? 0;
        if (performance.now() - last < QUALITY_REPEAT_MS) return;
        qualitySpokenRef.current.set(message.code, performance.now());
      }
      if (message.type === "get_ready") {
        const spoken = countdownSpokenRef.current;
        if (spoken && spoken.value === message.seconds && performance.now() - spoken.at < 1200) return;
        countdownSpokenRef.current = { value: message.seconds, at: performance.now() };
      }
      if (message.type === "step_preview") cancel();
      speakRef.current(text, { interrupt: message.type === "get_ready" });
    },
    [cancel],
  );

  useEffect(() => {
    if (!session) return undefined;
    const examSocket = new ExamSocket(websocketUrl(session.websocket_path));
    let active = true;
    examSocket.onAny((message) => {
      applyMessage(message);
      handleSpeech(message);
      if (message.type === "error") toast(errorText(message), "error");
      if (message.type === "examination_stopped") setStreaming(false);
      if (message.type === "report_ready") {
        setStreaming(false);
        examSocket.close();
        navigate(`/report/${encodeURIComponent(message.report.session_id || session.session_id)}`);
      }
    });
    examSocket.onState((state) => {
      if (state === "lost" && active) {
        setStreaming(false);
        setPhase("connection_lost");
        cancel();
      }
    });
    examSocket
      .connect()
      .then(() => {
        if (!active) return;
        setSocket(examSocket);
        setStreaming(true);
      })
      .catch(() => {
        if (!active) return;
        setPhase("connection_lost");
      });
    return () => {
      active = false;
      setStreaming(false);
      setSocket(null);
      examSocket.close();
    };
  }, [session, applyMessage, handleSpeech, navigate, setPhase, cancel]);

  useEffect(() => {
    if (phase !== "countdown" || countdown === null || countdown <= 1) return undefined;
    const timer = window.setTimeout(() => {
      tickCountdown();
      const next = useExamStore.getState().countdown;
      if (next !== null && next > 0) {
        countdownSpokenRef.current = { value: next, at: performance.now() };
        speakRef.current(String(next), { interrupt: true });
      }
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [phase, countdown, tickCountdown]);

  useFrameStreamer(videoRef, socket, streaming && camera.status === "ready");

  const sendControl = useCallback(
    (action: ControlAction) => {
      socket?.sendControl(action);
      if (action === "stop") setStreaming(false);
    },
    [socket],
  );

  const restart = async () => {
    if (!exam || !patient) {
      navigate(`/exam/${encodeURIComponent(examId)}`);
      return;
    }
    setRestarting(true);
    try {
      const fresh = await api.createSession({ exam_id: exam.exam_id, consent: true, camera_mirrored: false, patient });
      beginSession(fresh);
    } catch (error) {
      toast(error instanceof ApiError ? error.detail : "The examination could not be restarted.", "error");
    } finally {
      setRestarting(false);
    }
  };

  const currentStep = useMemo(() => steps.find((step) => step.id === currentStepId) ?? null, [steps, currentStepId]);
  const finishedSteps = Object.values(stepStatus).filter((status) => status !== "pending" && status !== "active").length;
  const examTitle = exam?.title ?? "";

  if (!session || !exam) return null;

  if (phase === "connection_lost") {
    return (
      <div className="mx-auto max-w-lg py-16">
        <Card>
          <div className="flex flex-col items-center gap-4 text-center">
            <span className="rounded-full bg-red-50 p-4 text-red-600 dark:bg-red-950">
              <Icon name="wifiOff" size={32} />
            </span>
            <h1 className="text-xl font-semibold">Connection lost</h1>
            <p className="text-sm text-ink-600 dark:text-ink-300">
              The connection to the diagnostic engine was interrupted, so the examination has stopped. You can restart it from the beginning.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              <Button icon="repeat" onClick={restart} disabled={restarting}>
                {restarting ? "Restarting…" : "Restart examination"}
              </Button>
              <Button variant="secondary" onClick={() => navigate("/")}>
                Back to home
              </Button>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  if (ANALYSIS_PHASES.includes(phase)) {
    return <AnalysisScreen examId={exam.exam_id} examTitle={examTitle} stage={phase === "finishing" ? "Finishing the examination…" : analysisStage} />;
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,65fr)_minmax(0,35fr)]">
      <div className="flex flex-col gap-3">
        <VideoStage
          videoRef={videoRef}
          stream={camera.stream}
          examId={exam.exam_id}
          view={currentStep?.view ?? exam.steps[0]?.view ?? null}
          highlight={highlightFor(exam.exam_id)}
          stepTitle={currentStep?.title ?? examTitle}
          overlayEnabled={camera.status === "ready"}
        >
          {camera.status !== "ready" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-ink-950/90 p-6 text-center text-white">
              {camera.status === "requesting" ? (
                <Spinner label="Starting camera" />
              ) : (
                <>
                  <Icon name="camera" size={36} />
                  <p className="max-w-sm text-sm">{camera.message}</p>
                  <Button variant="secondary" onClick={camera.retry}>
                    Try again
                  </Button>
                </>
              )}
            </div>
          )}
          {phase === "connecting" && camera.status === "ready" && (
            <div className="absolute inset-x-0 bottom-16 flex justify-center">
              <span className="flex items-center gap-2 rounded-full bg-black/60 px-4 py-2 text-sm text-white">
                <Spinner size={16} label="Connecting" /> Connecting to the diagnostic engine…
              </span>
            </div>
          )}
        </VideoStage>
        <CaptionBar text={caption} />
      </div>

      <aside className="flex flex-col gap-4">
        <Card>
          {currentStep ? (
            <StepPanel examId={exam.exam_id} step={currentStep} totalSteps={steps.length} finishedSteps={finishedSteps} />
          ) : (
            <div className="space-y-2">
              <h2 className="text-xl font-semibold">{examTitle}</h2>
              <p className="text-sm text-ink-600 dark:text-ink-300">
                {phase === "ready_to_start"
                  ? "Position yourself as shown in the preview, then start the examination."
                  : "Preparing the examination…"}
              </p>
            </div>
          )}
          <div className="mt-5">
            <ControlBar phase={phase} paused={paused} step={currentStep} areasCaptured={areasCaptured} onControl={sendControl} />
          </div>
        </Card>
        <Card title="Steps">
          <StepTimeline steps={steps} status={stepStatus} currentStepId={currentStepId} />
        </Card>
      </aside>
    </div>
  );
}
