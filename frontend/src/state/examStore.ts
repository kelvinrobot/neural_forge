import { create } from "zustand";
import type {
  CreateSessionResponse,
  ExamDetail,
  LiveValue,
  PatientInfo,
  QualityCode,
  Report,
  ServerMessage,
  Step,
} from "../api/types";
import { isBilateral, normalizeReport, normalizeStep, type RawStep } from "../api/types";

export type ExamPhase =
  | "connecting"
  | "ready_to_start"
  | "preview"
  | "waiting_ready"
  | "positioning"
  | "countdown"
  | "collecting"
  | "area_complete"
  | "between_steps"
  | "finishing"
  | "analysing"
  | "report"
  | "connection_lost";

export type StepRunStatus = "pending" | "active" | "completed" | "skipped" | "incomplete" | "repeated";

export interface StepProgressState {
  percent: number;
  elapsed: number;
  remaining: number;
  fps: number;
}

export interface LivePoint {
  t: number;
  values: number[];
}

const LIVE_HISTORY_SECONDS = 10;

interface ExamState {
  exam: ExamDetail | null;
  patient: PatientInfo | null;
  session: CreateSessionResponse | null;
  phase: ExamPhase;
  paused: boolean;
  steps: Step[];
  stepStatus: Record<string, StepRunStatus>;
  currentStepId: string | null;
  repeatNumber: number;
  displayText: string;
  qualityCode: QualityCode | null;
  qualityOk: boolean;
  countdown: number | null;
  progress: StepProgressState | null;
  live: LiveValue | null;
  liveHistory: LivePoint[];
  areasCaptured: number;
  analysisStage: string | null;
  stepNote: string | null;
  report: Report | null;
  caption: string;
  setExam: (exam: ExamDetail) => void;
  setPatient: (patient: PatientInfo) => void;
  beginSession: (session: CreateSessionResponse) => void;
  applyMessage: (message: ServerMessage) => void;
  tickCountdown: () => void;
  setPhase: (phase: ExamPhase) => void;
  setCaption: (caption: string) => void;
  setReport: (report: Report) => void;
  resetRun: () => void;
  clearAll: () => void;
}

const runDefaults = {
  session: null,
  phase: "connecting" as ExamPhase,
  paused: false,
  stepStatus: {} as Record<string, StepRunStatus>,
  currentStepId: null,
  repeatNumber: 1,
  displayText: "",
  qualityCode: null,
  qualityOk: false,
  countdown: null,
  progress: null,
  live: null,
  liveHistory: [] as LivePoint[],
  areasCaptured: 0,
  analysisStage: null,
  stepNote: null,
  report: null,
  caption: "",
};

function outcomeStatus(type: string, current: StepRunStatus | undefined): StepRunStatus {
  if (type === "step_skipped") return "skipped";
  if (type === "step_incomplete") return "incomplete";
  return current === "repeated" ? "repeated" : "completed";
}

/** Only numeric live values are plotted; text states such as "seated" are shown as chips only. */
function livePoint(live: LiveValue): LivePoint | null {
  const values = isBilateral(live) ? [live.right, live.left] : [live.value];
  if (!values.every((value): value is number => typeof value === "number" && Number.isFinite(value))) return null;
  return { t: performance.now() / 1000, values };
}

export const useExamStore = create<ExamState>((set, get) => ({
  exam: null,
  patient: null,
  steps: [],
  ...runDefaults,

  setExam: (exam) => set({ exam, steps: exam.steps }),
  setPatient: (patient) => set({ patient }),

  beginSession: (session) =>
    set({
      ...runDefaults,
      session,
      exam: session.exam,
      steps: session.exam.steps,
      stepStatus: Object.fromEntries(session.exam.steps.map((step) => [step.id, "pending" as StepRunStatus])),
    }),

  applyMessage: (message) => {
    const text = message.display_text ?? null;
    switch (message.type) {
      case "session_ready":
        set({ phase: "ready_to_start", displayText: text ?? "" });
        return;
      case "step_preview": {
        const step = normalizeStep(message.step as unknown as RawStep);
        const repeat_number = message.repeat_number ?? 1;
        const steps = get().steps.some((existing) => existing.id === step.id)
          ? get().steps.map((existing) => (existing.id === step.id ? step : existing))
          : [...get().steps, step];
        set((state) => ({
          steps,
          phase: "preview",
          currentStepId: step.id,
          repeatNumber: repeat_number,
          displayText: text ?? step.display_text,
          stepStatus: { ...state.stepStatus, [step.id]: repeat_number > 1 ? "repeated" : "active" },
          qualityCode: null,
          qualityOk: false,
          countdown: null,
          progress: null,
          live: null,
          liveHistory: [],
          stepNote: null,
        }));
        return;
      }
      case "waiting_for_ready":
        set({ phase: "waiting_ready", displayText: text ?? get().displayText });
        return;
      case "quality_feedback":
        set((state) => ({
          qualityCode: message.code,
          qualityOk: false,
          phase: state.phase === "collecting" || state.phase === "countdown" ? state.phase : "positioning",
        }));
        return;
      case "position_ok":
        set({ qualityCode: null, qualityOk: true });
        return;
      case "get_ready":
        set({ phase: "countdown", countdown: Math.max(0, Math.round(message.seconds)), qualityCode: null });
        return;
      case "step_started":
        set({
          phase: "collecting",
          countdown: null,
          progress: { percent: 0, elapsed: 0, remaining: message.max_seconds, fps: 0 },
        });
        return;
      case "step_progress": {
        const point = message.live ? livePoint(message.live) : null;
        const previous = get().liveHistory;
        const sameSeries = point !== null && previous.every((item) => item.values.length === point.values.length);
        const history = point
          ? [...(sameSeries ? previous : []), point].filter((item) => performance.now() / 1000 - item.t <= LIVE_HISTORY_SECONDS)
          : previous;
        set({
          phase: get().phase === "countdown" || get().phase === "positioning" ? "collecting" : get().phase,
          progress: {
            percent: message.progress_percent,
            elapsed: message.elapsed_seconds,
            remaining: message.seconds_remaining,
            fps: message.frames_per_second,
          },
          qualityOk: message.quality_ok,
          qualityCode: message.quality_ok ? null : get().qualityCode,
          live: message.live,
          liveHistory: history,
        });
        return;
      }
      case "paused":
        set({ paused: true });
        return;
      case "resumed":
        set({ paused: false });
        return;
      case "step_repeat":
        set((state) => ({
          stepStatus: { ...state.stepStatus, [message.step_id]: "repeated" },
          stepNote: text ?? message.reason,
        }));
        return;
      case "area_complete":
        set({ phase: "area_complete", areasCaptured: message.areas_captured, displayText: text ?? get().displayText });
        return;
      case "step_complete":
      case "step_skipped":
      case "step_incomplete":
        set((state) => ({
          phase: "between_steps",
          stepStatus: {
            ...state.stepStatus,
            [message.step_id]: outcomeStatus(message.type, state.stepStatus[message.step_id]),
          },
          stepNote: message.note ?? text,
          progress: null,
          live: null,
        }));
        return;
      case "examination_complete":
      case "examination_stopped":
      case "examination_incomplete":
        set({ phase: "finishing", displayText: text ?? get().displayText, paused: false });
        return;
      case "analysis_started":
        set({ phase: "analysing", analysisStage: text });
        return;
      case "analysis_progress":
        set({ phase: "analysing", analysisStage: text ?? message.stage });
        return;
      case "report_ready":
        set({ phase: "report", report: normalizeReport(message.report) });
        return;
      case "error":
      case "pong":
        return;
    }
  },

  tickCountdown: () => {
    const current = get().countdown;
    if (current !== null && current > 0) set({ countdown: current - 1 });
  },

  setPhase: (phase) => set({ phase }),
  setCaption: (caption) => set({ caption }),
  setReport: (report) => set({ report }),

  resetRun: () => set({ ...runDefaults }),

  clearAll: () => set({ exam: null, patient: null, steps: [], ...runDefaults }),
}));
