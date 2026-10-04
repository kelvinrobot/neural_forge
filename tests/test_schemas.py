"""Tests for Pydantic schemas and the WebSocket message protocol."""

import pytest
from pydantic import ValidationError

from neural_forge.schemas.findings import PossibleCondition, ReasoningOutput, ReferralUrgency, max_urgency
from neural_forge.schemas.messages import CLIENT_MESSAGE_ADAPTER, ControlAction, ControlMessage, FrameMessage, StepProgressMessage
from neural_forge.schemas.protocols import CompletionCriteria
from neural_forge.schemas.readings import QualityFlag, Reading
from neural_forge.schemas.sessions import CreateSessionRequest


def test_consent_must_be_true() -> None:
    with pytest.raises(ValidationError):
        CreateSessionRequest(area="face", consent=False)
    request = CreateSessionRequest(area="face", consent=True)
    assert request.preferred_language == "en"


def test_client_messages_use_discriminator() -> None:
    frame = CLIENT_MESSAGE_ADAPTER.validate_json('{"type": "frame", "session_id": "abc", "timestamp_ms": 12.5, "image_base64": "aGVsbG8gd29ybGQgaGVsbG8="}')
    assert isinstance(frame, FrameMessage)
    control = CLIENT_MESSAGE_ADAPTER.validate_json('{"type": "control", "action": "skip_step"}')
    assert isinstance(control, ControlMessage) and control.action == ControlAction.SKIP_STEP
    with pytest.raises(ValidationError):
        CLIENT_MESSAGE_ADAPTER.validate_json('{"type": "unknown"}')
    with pytest.raises(ValidationError):
        CLIENT_MESSAGE_ADAPTER.validate_json('{"type": "control", "action": "explode"}')


def test_server_message_serialises_type() -> None:
    message = StepProgressMessage(step_id="relaxed_face", percent_complete=50.0, seconds_elapsed=5.0, seconds_remaining=5.0)
    assert '"type":"step_progress"' in message.model_dump_json()


def test_reasoning_output_requires_supporting_signs() -> None:
    with pytest.raises(ValidationError):
        PossibleCondition(name="Example", supporting_signs=[], confidence=0.5)
    output = ReasoningOutput.model_validate_json(
        '{"findings_summary": "text", "possible_conditions": [], "confidence": 0.4, "recommended_next_tests": [], "repeat_steps": [], "referral_urgency": "routine", "extra": 1}'
    )
    assert output.referral_urgency == ReferralUrgency.ROUTINE


def test_max_urgency_ordering() -> None:
    assert max_urgency(ReferralUrgency.ROUTINE, ReferralUrgency.EMERGENCY) == ReferralUrgency.EMERGENCY
    assert max_urgency(ReferralUrgency.URGENT, ReferralUrgency.NONE) == ReferralUrgency.URGENT


def test_completion_criteria_validation() -> None:
    with pytest.raises(ValidationError):
        CompletionCriteria(type="events", min_valid_frames=10, description="missing event")
    CompletionCriteria(type="state", state="seated_after_walk", min_valid_frames=5, description="ok")


def test_reading_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Reading(agent="a", metric="m", value=1.0, unit="u", confidence=1.5, quality_flag=QualityFlag.GOOD, protocol_step="s", timestamp=0.0, description="d")
