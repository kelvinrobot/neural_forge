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

export interface ExamDetail extends ExamSummary {
  steps: Step[];
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
  value: number;
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
  source: string;
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
