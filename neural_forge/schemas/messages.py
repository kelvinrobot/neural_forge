"""WebSocket message protocol with a `type` discriminator."""

import time
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from neural_forge.schemas.findings import ReferralUrgency, StepStatus
from neural_forge.schemas.protocols import CameraSetup


class ControlAction(str, Enum):
    """Patient control actions sent by the client."""

    START = "start"
    READY = "ready"
    PAUSE = "pause"
    RESUME = "resume"
    SKIP_STEP = "skip_step"
    ADD_AREA = "add_area"
    STOP = "stop"


class FrameMessage(BaseModel):
    """A single camera frame sent by the client."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["frame"]
    session_id: str
    timestamp_ms: float = Field(ge=0.0)
    image_base64: str = Field(min_length=16)


class ControlMessage(BaseModel):
    """A patient control action."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["control"]
    action: ControlAction


class ReferenceCardMessage(BaseModel):
    """Client hint that a white or grey reference card is visible."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reference_card_present"]
    present: bool = True


ClientMessage = Annotated[
    Union[FrameMessage, ControlMessage, ReferenceCardMessage],
    Field(discriminator="type"),
]

CLIENT_MESSAGE_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def current_time_ms() -> float:
    """Wall-clock time in milliseconds."""
    return time.time() * 1000.0


class ServerMessageBase(BaseModel):
    """Common fields of all server messages."""

    server_time_ms: float = Field(default_factory=current_time_ms)


class NextStepPreview(BaseModel):
    """Short preview of the next step."""

    step_id: str
    step_number: int
    title: str
    instruction: str


class StepPreviewMessage(ServerMessageBase):
    """Announces the next step before it begins."""

    type: Literal["step_preview"] = "step_preview"
    step_id: str
    step_instance: str
    step_number: int
    total_steps: int
    title: str
    instruction: str
    tips: list[str]
    safety_note: str | None
    camera_setup: CameraSetup
    requires_patient_ready: bool
    skippable: bool
    repetition: int = 1
    attempt: int = 1


class QualityFeedbackMessage(ServerMessageBase):
    """Live camera quality guidance."""

    type: Literal["quality_feedback"] = "quality_feedback"
    step_id: str
    acceptable: bool
    issues: list[str]
    guidance: list[str]
    brightness: float
    blur_variance: float
    subject_present: bool
    estimated_distance_m: float | None = None


class GetReadyMessage(ServerMessageBase):
    """Countdown tick before data collection starts."""

    type: Literal["get_ready"] = "get_ready"
    step_id: str
    countdown_seconds: int
    seconds_remaining: int


class StepStartedMessage(ServerMessageBase):
    """Data collection for the step has started."""

    type: Literal["step_started"] = "step_started"
    step_id: str
    step_instance: str
    step_number: int
    min_duration_seconds: float
    max_duration_seconds: float
    prompt: str | None = None


class StepProgressMessage(ServerMessageBase):
    """Periodic progress update while a step is running."""

    type: Literal["step_progress"] = "step_progress"
    step_id: str
    percent_complete: float
    seconds_elapsed: float
    seconds_remaining: float
    current_prompt: str | None = None
    paused: bool = False


class LiveReadingMessage(ServerMessageBase):
    """A live measurement for display during the step."""

    type: Literal["live_reading"] = "live_reading"
    step_id: str
    agent: str
    metric: str
    value: float | None
    unit: str
    confidence: float
    description: str


class StepRepeatMessage(ServerMessageBase):
    """The step will be repeated once with guidance."""

    type: Literal["step_repeat"] = "step_repeat"
    step_id: str
    attempt: int
    reason: str
    guidance: list[str]


class StepCompleteMessage(ServerMessageBase):
    """The step has finished."""

    type: Literal["step_complete"] = "step_complete"
    step_id: str
    step_instance: str
    status: StepStatus
    message: str
    next_step: NextStepPreview | None = None
    can_add_area: bool = False


class ExaminationCompleteMessage(ServerMessageBase):
    """All steps have finished or the patient stopped."""

    type: Literal["examination_complete"] = "examination_complete"
    completed_steps: int
    repeated_steps: int
    skipped_steps: int
    incomplete_steps: int
    stopped: bool
    message: str


class AnalysisStartedMessage(ServerMessageBase):
    """Analysis of the collected data has started."""

    type: Literal["analysis_started"] = "analysis_started"
    message: str


class ReportReadyMessage(ServerMessageBase):
    """The final report is available."""

    type: Literal["report_ready"] = "report_ready"
    session_id: str
    report_path: str
    summary: str
    referral_urgency: ReferralUrgency
    red_flag_alerts: list[str]
    disclaimer: str


class ErrorMessage(ServerMessageBase):
    """An error that occurred while handling the session."""

    type: Literal["error"] = "error"
    code: str
    message: str
    recoverable: bool = True


class KeepaliveMessage(ServerMessageBase):
    """Application-level keep-alive for proxies that drop idle connections."""

    type: Literal["keepalive"] = "keepalive"


ServerMessage = Union[
    StepPreviewMessage,
    QualityFeedbackMessage,
    GetReadyMessage,
    StepStartedMessage,
    StepProgressMessage,
    LiveReadingMessage,
    StepRepeatMessage,
    StepCompleteMessage,
    ExaminationCompleteMessage,
    AnalysisStartedMessage,
    ReportReadyMessage,
    ErrorMessage,
    KeepaliveMessage,
]
