"""Measurement, quality, flag and agent result schemas."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class QualityFlag(str, Enum):
    """Quality classification attached to every reading."""

    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


class FlagSeverity(str, Enum):
    """Severity of an educational screening flag."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


Side = Literal["left", "right", "both", "midline"]


class Reading(BaseModel):
    """A single timestamped measurement produced by an agent."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    metric: str
    value: float | None
    unit: str
    confidence: float = Field(ge=0.0, le=1.0)
    quality_flag: QualityFlag
    protocol_step: str
    timestamp: float
    description: str
    side: Side | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    """Output of an image classifier for one image and one task."""

    task: str
    method: Literal["zero_shot", "trained_head"]
    label_scores: dict[str, float]
    top_label: str
    top_score: float = Field(ge=0.0, le=1.0)
    margin: float = Field(ge=0.0, le=1.0)


class EvidenceCrop(BaseModel):
    """An image crop that supports a finding and may be sent to the medical consultant."""

    crop_id: str
    agent: str
    label: str
    protocol_step: str
    timestamp: float
    image_jpeg_base64: str = Field(default="", exclude=True, repr=False)
    classification: ClassificationResult | None = None
    needs_consultation: bool = False
    consultation_question: str | None = None
    saved_path: str | None = None


class ScreeningFlag(BaseModel):
    """An educational screening flag raised when a Python threshold is crossed."""

    code: str
    title: str
    severity: FlagSeverity
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    agent: str
    protocol_steps: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    side: Side | None = None


class AgentResult(BaseModel):
    """Final summarised output of one area agent for one session."""

    agent: str
    area: str
    value: float | None = None
    unit: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    quality_flag: QualityFlag
    protocol_step: str
    timestamp: float
    description: str
    metrics: list[Reading] = Field(default_factory=list)
    flags: list[ScreeningFlag] = Field(default_factory=list)
    evidence_crops: list[EvidenceCrop] = Field(default_factory=list)
    steps_with_data: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    status: Literal["ok", "insufficient_data", "error"] = "ok"


class QualityAssessment(BaseModel):
    """Per-frame image quality and subject framing assessment."""

    usable: bool
    quality_flag: QualityFlag
    score: float = Field(ge=0.0, le=1.0)
    brightness: float = 0.0
    contrast: float = 0.0
    blur_variance: float = 0.0
    motion: float = 0.0
    subject: str | None = None
    subject_present: bool = False
    subject_fraction: float = 0.0
    estimated_distance_m: float | None = None
    issues: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)


class CalibrationInfo(BaseModel):
    """Colour calibration applied to a frame."""

    method: Literal["reference_card", "session_reference", "grey_world", "none"]
    gains_bgr: tuple[float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    card_detected: bool = False
