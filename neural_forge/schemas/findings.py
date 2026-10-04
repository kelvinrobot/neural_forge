"""Aggregated findings, reasoning output and consultant schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from neural_forge.schemas.readings import AgentResult, EvidenceCrop, ScreeningFlag


class PatientContext(BaseModel):
    """Optional patient context supplied when a session is created."""

    model_config = ConfigDict(extra="forbid")

    age: int | None = Field(default=None, ge=0, le=120)
    sex: Literal["female", "male", "other", "unspecified"] = "unspecified"
    symptoms: list[str] = Field(default_factory=list, max_length=30)


class StepStatus(str, Enum):
    """Outcome of a protocol step."""

    COMPLETED = "completed"
    REPEATED = "repeated"
    SKIPPED = "skipped"
    INCOMPLETE = "incomplete"
    NOT_REACHED = "not_reached"


class StepRecord(BaseModel):
    """What happened during one protocol step instance."""

    step_id: str
    step_instance: str
    step_number: int
    title: str
    status: StepStatus
    attempts: int = 0
    reason: str | None = None
    duration_seconds: float = 0.0
    valid_frames: int = 0
    quality_issues: list[str] = Field(default_factory=list)


class MetricStatistics(BaseModel):
    """Robust statistics of one metric for one agent and one step."""

    agent: str
    metric: str
    unit: str
    protocol_step: str
    side: str | None = None
    median: float
    q1: float
    q3: float
    iqr: float
    minimum: float
    maximum: float
    count: int
    discarded: int
    mean_confidence: float


class AgentFindings(BaseModel):
    """Agent result together with statistics over its per-frame readings."""

    agent: str
    result: AgentResult
    statistics: list[MetricStatistics] = Field(default_factory=list)


class QualitySummary(BaseModel):
    """Session-level frame and image quality summary."""

    frames_received: int = 0
    frames_dropped: int = 0
    frames_rejected: int = 0
    frames_processed: int = 0
    frames_usable: int = 0
    usable_fraction: float = 0.0
    measured_fps: float = 0.0
    mean_brightness: float = 0.0
    mean_blur_variance: float = 0.0
    common_issues: dict[str, int] = Field(default_factory=dict)
    calibration_method: str = "none"
    calibration_confidence: float = 0.0


class SessionFindings(BaseModel):
    """The structured findings JSON built for one examination session."""

    session_id: str
    area: str
    generated_at: datetime
    patient_context: PatientContext
    steps: list[StepRecord]
    agents: list[AgentFindings]
    flags: list[ScreeningFlag]
    evidence: list[EvidenceCrop] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    quality: QualitySummary
    stopped_early: bool = False
    educational_notice: str

    def to_llm_payload(self) -> dict[str, Any]:
        """Return a compact dictionary for the reasoning model without raw image data."""
        return {
            "area": self.area,
            "patient_context": self.patient_context.model_dump(),
            "overall_confidence": round(self.overall_confidence, 3),
            "stopped_early": self.stopped_early,
            "steps": [
                {
                    "step_id": step.step_id,
                    "instance": step.step_instance,
                    "status": step.status.value,
                    "attempts": step.attempts,
                    "reason": step.reason,
                }
                for step in self.steps
            ],
            "flags": [
                {
                    "code": flag.code,
                    "title": flag.title,
                    "severity": flag.severity.value,
                    "confidence": round(flag.confidence, 3),
                    "side": flag.side,
                    "description": flag.description,
                    "evidence": flag.evidence,
                }
                for flag in self.flags
            ],
            "agents": [
                {
                    "agent": agent_findings.agent,
                    "status": agent_findings.result.status,
                    "confidence": round(agent_findings.result.confidence, 3),
                    "summary": agent_findings.result.description,
                    "metrics": [
                        {
                            "metric": metric.metric,
                            "value": None if metric.value is None else round(metric.value, 4),
                            "unit": metric.unit,
                            "side": metric.side,
                            "step": metric.protocol_step,
                            "confidence": round(metric.confidence, 3),
                        }
                        for metric in agent_findings.result.metrics
                    ],
                    "limitations": agent_findings.result.limitations,
                }
                for agent_findings in self.agents
            ],
            "evidence": [
                {
                    "crop_id": crop.crop_id,
                    "agent": crop.agent,
                    "label": crop.label,
                    "step": crop.protocol_step,
                    "classification": None
                    if crop.classification is None
                    else {
                        "task": crop.classification.task,
                        "top_label": crop.classification.top_label,
                        "top_score": round(crop.classification.top_score, 3),
                        "margin": round(crop.classification.margin, 3),
                    },
                    "needs_consultation": crop.needs_consultation,
                }
                for crop in self.evidence
            ],
            "quality": {
                "usable_fraction": round(self.quality.usable_fraction, 3),
                "measured_fps": round(self.quality.measured_fps, 2),
                "calibration_method": self.quality.calibration_method,
                "calibration_confidence": round(self.quality.calibration_confidence, 3),
                "common_issues": self.quality.common_issues,
            },
        }


class ReferralUrgency(str, Enum):
    """How soon an in-person assessment is recommended."""

    NONE = "none"
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    EMERGENCY = "emergency"


URGENCY_ORDER: list[ReferralUrgency] = [
    ReferralUrgency.NONE,
    ReferralUrgency.ROUTINE,
    ReferralUrgency.SOON,
    ReferralUrgency.URGENT,
    ReferralUrgency.EMERGENCY,
]


def max_urgency(first: ReferralUrgency, second: ReferralUrgency) -> ReferralUrgency:
    """Return the more urgent of two referral urgencies."""
    return first if URGENCY_ORDER.index(first) >= URGENCY_ORDER.index(second) else second


class PossibleCondition(BaseModel):
    """A condition that the screening signs may be consistent with."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=200)
    supporting_signs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=1500)


class ReasoningOutput(BaseModel):
    """Strict JSON schema the reasoning model must return."""

    model_config = ConfigDict(extra="ignore")

    findings_summary: str = Field(min_length=1, max_length=4000)
    possible_conditions: list[PossibleCondition] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_next_tests: list[str] = Field(default_factory=list, max_length=10)
    repeat_steps: list[str] = Field(default_factory=list, max_length=20)
    referral_urgency: ReferralUrgency


class ConsultationRequest(BaseModel):
    """A single-image question routed to the medical consultant."""

    model_config = ConfigDict(extra="ignore")

    crop_id: str
    question: str = Field(min_length=3, max_length=400)


class ConsultationPlan(BaseModel):
    """Which evidence crops the reasoning model wants a consultant opinion on."""

    model_config = ConfigDict(extra="ignore")

    consultations: list[ConsultationRequest] = Field(default_factory=list, max_length=10)


class ConsultantResponse(BaseModel):
    """Strict JSON schema the medical consultant model must return."""

    model_config = ConfigDict(extra="ignore")

    observation: str = Field(min_length=1, max_length=1200)
    likely_finding: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    recommend_in_person_review: bool


class ConsultantOpinion(BaseModel):
    """A structured opinion returned by the medical consultant agent."""

    crop_id: str
    question: str
    observation: str
    likely_finding: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommend_in_person_review: bool
    parsed: bool = True


class ReasoningResult(BaseModel):
    """Reasoning output together with provenance information."""

    output: ReasoningOutput
    source: Literal["llm", "rule_based"]
    consultations: list[ConsultantOpinion] = Field(default_factory=list)
    rule_matches: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QuestionAnswerOutput(BaseModel):
    """Strict JSON schema for follow-up question answers."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1, max_length=3000)
    related_findings: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)
    suggest_professional_follow_up: bool = True
