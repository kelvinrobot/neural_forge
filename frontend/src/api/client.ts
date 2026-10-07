import {
  normalizeExam,
  normalizeReport,
  type CreateSessionRequest,
  type CreateSessionResponse,
  type ExamDetail,
  type ExamsResponse,
  type ExamSummary,
  type HealthResponse,
  type QuestionResponse,
  type RawStep,
  type Report,
} from "./types";

type RawExam = ExamSummary & { steps?: RawStep[] | null };

const DEFAULT_BASE_URL = "https://drrobot9-neural-forge.hf.space";

export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail || `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      return typeof detail === "string" ? detail : JSON.stringify(detail);
    }
    return JSON.stringify(body);
  } catch {
    return response.statusText;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "The diagnostic engine could not be reached.");
  }
  if (!response.ok) {
    throw new ApiError(response.status, await readDetail(response));
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function websocketUrl(path: string): string {
  const url = new URL(path, `${API_BASE_URL}/`);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export const api = {
  health: () => request<HealthResponse>("/api/v1/health"),
  exams: () => request<ExamsResponse>("/api/v1/exams"),
  exam: async (examId: string): Promise<ExamDetail> =>
    normalizeExam(await request<RawExam>(`/api/v1/exams/${encodeURIComponent(examId)}`)),
  createSession: async (body: CreateSessionRequest): Promise<CreateSessionResponse> => {
    const response = await request<Omit<CreateSessionResponse, "exam"> & { exam: RawExam }>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return { ...response, exam: normalizeExam(response.exam) };
  },
  report: async (sessionId: string): Promise<Report> =>
    normalizeReport(await request<Report>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/report`)),
  question: (sessionId: string, question: string) =>
    request<QuestionResponse>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/question`, {
      method: "POST",
      body: JSON.stringify({ question }),
    }),
  deleteSession: (sessionId: string) =>
    request<unknown>(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" }),
};
