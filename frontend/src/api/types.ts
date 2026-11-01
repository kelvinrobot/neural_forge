export type HealthStatus = "loading" | "ready" | "degraded";

export interface ModelStatusDetail {
  status?: string;
  name?: string;
  loaded?: boolean;
  error?: string | null;
  [key: string]: unknown;
}

export interface GpuDetail {
  name?: string;
  available?: boolean;
  memory_total_mb?: number;
  [key: string]: unknown;
}

export interface HealthResponse {
  status: HealthStatus;
  models?: Record<string, string | ModelStatusDetail> | null;
  errors?: string[] | Record<string, string> | null;
  gpu?: string | boolean | GpuDetail | null;
  installed_exams?: string[];
  [key: string]: unknown;
}


export type StepView =
  | "face"
  | "face_close"
  | "mouth"
  | "hands"
  | "hand_close"
  | "upper_body"
  | "full_body"
  | "skin";

export interface ExamSummary {
  exam_id: string;
  title: string;
  description: string;
  total_steps: number;
  estimated_minutes: number;
}

export interface ExamsResponse {
  exams: ExamSummary[];
}

export interface Step {
  id: string;
  number: number;
  total: number;
  title: string;
  display_text: string;
  speech_text: string;
  tips: string[];
  safety_note: string | null;
  view: StepView;
  min_seconds: number;
  max_seconds: number;
  requires_ready: boolean;
  skippable: boolean;
  repeatable: boolean;
}

/** Step as sent over the wire: tips and safety_note are strings where "" means none (a tips list is also accepted). */
export type RawStep = Omit<Step, "tips" | "safety_note"> & {
  tips?: string | string[] | null;
  safety_note?: string | null;
};

export interface ExamDetail extends ExamSummary {
  steps: Step[];
}

/** Normalises a step from the backend so the UI can rely on its shape. */
export function normalizeStep(raw: RawStep): Step {
  const tips = Array.isArray(raw.tips)
    ? raw.tips
    : (raw.tips ?? "")
        .split(/(?<=[.!?])\s+/)
        .map((tip) => tip.trim());
  const safety = typeof raw.safety_note === "string" ? raw.safety_note.trim() : "";
  return {
    ...raw,
    title: raw.title ?? raw.id,
    display_text: raw.display_text ?? "",
    speech_text: raw.speech_text ?? "",
    tips: tips.filter((tip) => tip.length > 0),
    safety_note: safety.length > 0 ? safety : null,
  };
}

function asList(value: unknown): string[] {
  if (Array.isArray(value)) return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

/** Normalises list fields of a report so rendering never depends on the exact wire shape. */
export function normalizeReport(raw: Report): Report {
  const interpretation: Partial<Interpretation> = raw.interpretation ?? {};
  const quality: Partial<ReportQuality> = raw.quality ?? {};
  return {
    ...raw,
    findings: (raw.findings ?? []).map((finding) => ({
      ...finding,
      measurements: Array.isArray(finding.measurements) ? finding.measurements : [],
      confidence: typeof finding.confidence === "number" ? finding.confidence : 0,
      explanation: finding.explanation ?? "",
    })),
    interpretation: {
      summary: interpretation.summary ?? "",
      patient_message: interpretation.patient_message ?? "",
      possible_conditions: (interpretation.possible_conditions ?? []).map((condition) => ({
        ...condition,
        likelihood: condition.likelihood ?? "",
        reasoning: condition.reasoning ?? "",
        supporting_findings: asList(condition.supporting_findings),
      })),
      recommendations: asList(interpretation.recommendations),
      referral_urgency: interpretation.referral_urgency ?? raw.referral_urgency ?? "none",
      source: interpretation.source === "ai_model" ? "ai_model" : "rule_based",
    },
    steps: Array.isArray(raw.steps) ? raw.steps : [],
    quality: {
      completed_steps: quality.completed_steps ?? 0,
      total_steps: quality.total_steps ?? 0,
      skipped_steps: quality.skipped_steps ?? 0,
      incomplete_steps: quality.incomplete_steps ?? 0,
      notes: asList(quality.notes),
    },
    patient: raw.patient ?? { age: null, sex: "unspecified", symptoms: [], medical_history: [], notes: "" },
    disclaimer: raw.disclaimer ?? "",
    speech_summary: raw.speech_summary ?? "",
  };
}

export function normalizeExam(raw: ExamSummary & { steps?: RawStep[] | null }): ExamDetail {
  return { ...raw, steps: (raw.steps ?? []).map(normalizeStep) };
}

export type Sex = "male" | "female" | "other" | "unspecified";

export interface PatientInfo {
  age: number | null;
  sex: Sex;
  symptoms: string[];
  medical_history: string[];
  notes: string;
}

export interface CreateSessionRequest {
  exam_id: string;
  consent: true;
  camera_mirrored: boolean;
  patient: PatientInfo;
}

export interface CreateSessionResponse {
  session_id: string;
  exam: ExamDetail;
  websocket_path: string;
  state: string;
}

export interface QuestionResponse {
  answer: string;
  speech_text: string;
}

export type QualityCode =
  | "too_dark"
  | "too_bright"
  | "blurry"
  | "no_face"
  | "face_too_far"
  | "face_too_close"
  | "need_closer_eyes"
  | "need_closer_mouth"
  | "head_turned"
  | "head_tilted"
  | "hands_missing_one"
  | "hands_missing_two"
  | "hand_too_far"
  | "upper_body_missing"
  | "full_body_missing"
  | "full_body_too_small"
  | "full_body_cut"
  | "no_frames";

export type ControlAction = "start" | "ready" | "pause" | "resume" | "skip" | "stop" | "add_area" | "next";

export interface LiveSingle {
  label: string;
  value: number | string;
  unit: string;
}

export interface LiveBilateral {
  label: string;
  right: number;
  left: number;
  unit: string;
}

export type LiveValue = LiveSingle | LiveBilateral;

export function isBilateral(live: LiveValue): live is LiveBilateral {
  return "right" in live && "left" in live;
}

interface MessageBase {
  session_id: string;
  timestamp: number | string;
  display_text?: string | null;
  speech_text?: string | null;
}

export interface SessionReadyMessage extends MessageBase {
  type: "session_ready";
  exam_title: string;
  total_steps: number;
  estimated_minutes: number;
}

export interface StepPreviewMessage extends MessageBase {
  type: "step_preview";
  step: Step;
  repeat_number: number;
}

export interface WaitingForReadyMessage extends MessageBase {
  type: "waiting_for_ready";
  step_id: string;
}

export interface QualityFeedbackMessage extends MessageBase {
  type: "quality_feedback";
  step_id: string;
  code: QualityCode;
}

export interface PositionOkMessage extends MessageBase {
  type: "position_ok";
  step_id: string;
}

export interface GetReadyMessage extends MessageBase {
  type: "get_ready";
  step_id: string;
  seconds: number;
}

export interface StepStartedMessage extends MessageBase {
  type: "step_started";
  step_id: string;
  min_seconds: number;
  max_seconds: number;
}

export interface StepProgressMessage extends MessageBase {
  type: "step_progress";
  step_id: string;
  step_number: number;
  total_steps: number;
  progress_percent: number;
  elapsed_seconds: number;
  seconds_remaining: number;
  quality_ok: boolean;
  frames_per_second: number;
  live: LiveValue | null;
}

export interface PausedMessage extends MessageBase {
  type: "paused";
}

export interface ResumedMessage extends MessageBase {
  type: "resumed";
}

export interface StepRepeatMessage extends MessageBase {
  type: "step_repeat";
  step_id: string;
  reason: string;
}

export interface AreaCompleteMessage extends MessageBase {
  type: "area_complete";
  step_id: string;
  areas_captured: number;
}

export type StepOutcomeType = "step_complete" | "step_skipped" | "step_incomplete";

export interface StepOutcomeMessage extends MessageBase {
  type: StepOutcomeType;
  step_id: string;
  status: string;
  note: string | null;
  next_step: Step | null;
}

export type ExamEndType = "examination_complete" | "examination_stopped" | "examination_incomplete";

export interface ExamEndMessage extends MessageBase {
  type: ExamEndType;
}

export interface AnalysisStartedMessage extends MessageBase {
  type: "analysis_started";
}

export interface AnalysisProgressMessage extends MessageBase {
  type: "analysis_progress";
  stage: string;
}

export interface ReportReadyMessage extends MessageBase {
  type: "report_ready";
  report: Report;
}

export interface ErrorMessage extends MessageBase {
  type: "error";
  code: string;
}

export interface PongMessage extends MessageBase {
  type: "pong";
}

export type ServerMessage =
  | SessionReadyMessage
  | StepPreviewMessage
  | WaitingForReadyMessage
  | QualityFeedbackMessage
  | PositionOkMessage
  | GetReadyMessage
  | StepStartedMessage
  | StepProgressMessage
  | PausedMessage
  | ResumedMessage
  | StepRepeatMessage
  | AreaCompleteMessage
  | StepOutcomeMessage
  | ExamEndMessage
  | AnalysisStartedMessage
  | AnalysisProgressMessage
  | ReportReadyMessage
  | ErrorMessage
  | PongMessage;

export type ServerMessageType = ServerMessage["type"];

export type MessageOf<T extends ServerMessageType> = Extract<ServerMessage, { type: T }>;

export type OverallStatus =
  | "urgent_attention"
  | "findings_detected"
  | "borderline_findings"
  | "no_significant_findings"
  | "inconclusive";

export type ReferralUrgency = "none" | "routine" | "soon" | "urgent";

export type FindingStatus = "normal" | "borderline" | "abnormal" | "inconclusive";

export interface Measurement {
  name: string;
  value: number | string | null;
  unit: string | null;
  reference: string | null;
  side: string | null;
}

export interface Finding {
  code: string;
  title: string;
  status: FindingStatus;
  confidence: number;
  measurements: Measurement[];
  explanation: string;
  red_flag: boolean;
  evidence_image: string | null;
}

export interface PossibleCondition {
  name: string;
  likelihood: string;
  supporting_findings: string[];
  reasoning: string;
}

export interface Interpretation {
  summary: string;
  patient_message: string;
  possible_conditions: PossibleCondition[];
  recommendations: string[];
  referral_urgency: ReferralUrgency;
  source: "ai_model" | "rule_based";
}

export interface ReportStep {
  step_id: string;
  title: string;
  status: string;
  attempts: number;
  valid_frames: number;
  duration_seconds: number;
  note: string | null;
}

export interface ReportQuality {
  completed_steps: number;
  total_steps: number;
  skipped_steps: number;
  incomplete_steps: number;
  notes: string[];
}

export interface Report {
  report_id: string;
  session_id: string;
  exam_id: string;
  exam_title: string;
  generated_at: string;
  patient: PatientInfo;
  overall_status: OverallStatus;
  overall_status_label: string;
  referral_urgency: ReferralUrgency;
  findings: Finding[];
  interpretation: Interpretation;
  steps: ReportStep[];
  quality: ReportQuality;
  models: Record<string, string> | string[] | string | null;
  speech_summary: string;
  disclaimer: string;
  stopped_early: boolean;
}

export type ConsultationStatus = "requested" | "accepted" | "declined" | "ended" | "expired";

export interface Consultation {
  consultation_id: string;
  session_id: string;
  professional_id: string;
  professional_name: string;
  patient_display_name: string;
  patient_note: string;
  exam_title: string;
  overall_status_label: string;
  referral_urgency: string;
  status: ConsultationStatus;
  channel_name: string;
  created_at: number;
  accepted_at: number | null;
  ended_at: number | null;
  updated_at: number;
}

export interface CreateConsultationRequest {
  session_id: string;
  professional_id: string;
  professional_name: string;
  patient_display_name: string;
  patient_note: string;
  consent: true;
}

export interface PortalLoginResponse {
  professional_id: string;
}

export interface CallToken {
  app_id: string;
  channel_name: string;
  uid: number;
  token: string;
  expires_at: number;
}
