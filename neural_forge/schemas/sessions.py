"""Session API and stored session state schemas."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from neural_forge.schemas.findings import PatientContext, SessionFindings, StepRecord
from neural_forge.schemas.protocols import AreaName, ProtocolStepView
from neural_forge.schemas.reports import FinalReport

__all__ = [
    "CreateSessionRequest",
    "CreateSessionResponse",
    "DeleteSessionResponse",
    "PatientContext",
    "QAEntry",
    "QuestionRequest",
    "QuestionResponse",
    "SessionState",
    "SessionStatus",
    "SessionSummaryResponse",
]


class SessionStatus(str, Enum):
    """Lifecycle status of an examination session."""

    CREATED = "created"
    ACTIVE = "active"
    ANALYSING = "analysing"
    COMPLETED = "completed"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class CreateSessionRequest(BaseModel):
    """Request body for creating an examination session."""

    area: AreaName
    consent: bool
    patient_context: PatientContext = Field(default_factory=PatientContext)
    preferred_language: str = Field(default="en", pattern=r"^[a-z]{2,3}$")

    @field_validator("consent")
    @classmethod
    def require_consent(cls, value: bool) -> bool:
        """Reject requests that do not explicitly give consent."""
        if value is not True:
            raise ValueError("consent must be true to start an educational screening session")
        return value


class CreateSessionResponse(BaseModel):
    """Response returned when a session is created."""

    session_id: str
    area: AreaName
    language: str
    total_steps: int
    steps: list[ProtocolStepView]
    websocket_path: str
    educational_notice: str


class QAEntry(BaseModel):
    """A follow-up question and its answer."""

    question: str
    answer: str
    asked_at: datetime


class SessionState(BaseModel):
    """Persistent state of one session held in the session repository."""

    session_id: str
    area: AreaName
    language: str
    patient_context: PatientContext
    consent_given_at: datetime
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    step_records: list[StepRecord] = Field(default_factory=list)
    findings: SessionFindings | None = None
    report: FinalReport | None = None
    questions: list[QAEntry] = Field(default_factory=list)


class QuestionRequest(BaseModel):
    """Follow-up question about a completed session."""

    question: str = Field(min_length=1, max_length=2000)


class QuestionResponse(BaseModel):
    """Answer to a follow-up question."""

    session_id: str
    question: str
    answer: str
    related_findings: list[str]
    confidence: float
    suggest_professional_follow_up: bool
    disclaimer: str


class DeleteSessionResponse(BaseModel):
    """Result of deleting a session."""

    session_id: str
    deleted: bool


class SessionSummaryResponse(BaseModel):
    """Aggregated findings returned by the summary endpoint."""

    session_id: str
    area: AreaName
    status: SessionStatus
    findings: SessionFindings
