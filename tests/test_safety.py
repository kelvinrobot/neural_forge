"""Tests for the safety agent rules."""

from datetime import datetime, timezone

from neural_forge.agents.core.safety import SafetyAgent
from neural_forge.schemas.findings import (
    PatientContext,
    PossibleCondition,
    QualitySummary,
    ReasoningOutput,
    ReasoningResult,
    ReferralUrgency,
    SessionFindings,
)
from neural_forge.schemas.readings import FlagSeverity, ScreeningFlag
from tests.synthetic import config_repository


def make_findings(flags: list[ScreeningFlag]) -> SessionFindings:
    return SessionFindings(
        session_id="s",
        area="face",
        generated_at=datetime.now(timezone.utc),
        patient_context=PatientContext(),
        steps=[],
        agents=[],
        flags=flags,
        overall_confidence=0.6,
        quality=QualitySummary(),
        educational_notice="notice",
    )


def make_result(summary: str, conditions: list[PossibleCondition], urgency: ReferralUrgency) -> ReasoningResult:
    return ReasoningResult(
        output=ReasoningOutput(
            findings_summary=summary,
            possible_conditions=conditions,
            confidence=0.55,
            recommended_next_tests=["This test confirms the diagnosis"],
            repeat_steps=[],
            referral_urgency=urgency,
        ),
        source="llm",
    )


def test_definitive_language_is_removed() -> None:
    safety = SafetyAgent(config_repository().safety_rules())
    text, changes = safety.sanitize_text("You have Bell's palsy and this confirms the diagnosis.")
    assert "You have" not in text
    assert "confirms" not in text
    assert "diagnosis" not in text
    assert changes


def test_review_adds_confidence_hedging_and_disclaimer() -> None:
    safety = SafetyAgent(config_repository().safety_rules())
    condition = PossibleCondition(name="Bell's palsy", supporting_signs=["whole_side_facial_weakness"], confidence=0.42, rationale="The patient has weakness.")
    interpretation = safety.review(make_result("The patient has facial weakness.", [condition], ReferralUrgency.ROUTINE), make_findings([]), "en")
    assert interpretation.possible_conditions[0].name.startswith("Possible")
    assert "42%" in interpretation.possible_conditions[0].rationale
    assert "Overall screening confidence: 55%" in interpretation.findings_summary
    assert interpretation.disclaimer
    assert "confirms" not in interpretation.recommended_next_tests[0]


def test_red_flags_force_urgent_referral() -> None:
    safety = SafetyAgent(config_repository().safety_rules())
    flag = ScreeningFlag(
        code="facial_asymmetry_forehead_sparing",
        title="Lower facial weakness with forehead sparing",
        severity=FlagSeverity.HIGH,
        confidence=0.7,
        description="d",
        agent="facial_symmetry",
    )
    interpretation = safety.review(make_result("Summary.", [], ReferralUrgency.ROUTINE), make_findings([flag]), "en")
    assert interpretation.referral_urgency == ReferralUrgency.EMERGENCY
    assert interpretation.red_flag_alerts and interpretation.red_flag_alerts[0].code == "stroke_pattern"
    assert interpretation.findings_summary.startswith("URGENT")


def test_low_confidence_red_flag_is_ignored() -> None:
    safety = SafetyAgent(config_repository().safety_rules())
    flag = ScreeningFlag(code="seizure_like_movement", title="t", severity=FlagSeverity.HIGH, confidence=0.1, description="d", agent="involuntary_movement")
    assert safety.red_flag_alerts([flag], "en") == []
