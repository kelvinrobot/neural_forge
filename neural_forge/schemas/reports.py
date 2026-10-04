"""Safety-reviewed interpretation and final report schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from neural_forge.schemas.findings import (
    ConsultantOpinion,
    PatientContext,
    PossibleCondition,
    ReferralUrgency,
    StepStatus,
)
from neural_forge.schemas.readings import EvidenceCrop, QualityFlag, Reading, ScreeningFlag


class RedFlagAlert(BaseModel):
    """An urgent-referral alert forced by a safety rule."""

    code: str
    urgency: ReferralUrgency
    message: str


class SafeInterpretation(BaseModel):
    """Reasoning output after the safety agent post-check."""

    findings_summary: str
    possible_conditions: list[PossibleCondition]
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_next_tests: list[str]
    repeat_steps: list[str]
    referral_urgency: ReferralUrgency
    red_flag_alerts: list[RedFlagAlert] = Field(default_factory=list)
    consultations: list[ConsultantOpinion] = Field(default_factory=list)
    source: Literal["llm", "rule_based"]
    safety_modifications: list[str] = Field(default_factory=list)
    disclaimer: str


class StepReportEntry(BaseModel):
    """Report line describing one step outcome."""

    step_number: int
    step_id: str
    step_instance: str
    title: str
    status: StepStatus
    attempts: int
    reason: str | None = None


class FindingReportEntry(BaseModel):
    """Report entry summarising one agent's findings."""

    agent: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    quality_flag: QualityFlag
    key_metrics: list[Reading] = Field(default_factory=list)
    flags: list[ScreeningFlag] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    """Final structured report and plain-English summary for one area examination."""

    session_id: str
    area: str
    area_title: str
    language: str
    generated_at: datetime
    patient_context: PatientContext
    steps: list[StepReportEntry]
    steps_completed: int
    steps_repeated: int
    steps_skipped: int
    steps_incomplete: int
    stopped_early: bool
    findings: list[FindingReportEntry]
    screening_flags: list[ScreeningFlag]
    interpretation: SafeInterpretation
    plain_english_summary: str
    quality_notes: list[str]
    limitations: list[str]
    evidence_crops: list[EvidenceCrop] = Field(default_factory=list)
    disclaimer: str
