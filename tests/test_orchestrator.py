"""Tests for the session orchestrator step flow with a fake pipeline."""

import asyncio
from typing import Any

import pytest

from neural_forge.agents.core.frame_ingest import FrameIngestAgent, RawFrame
from neural_forge.agents.core.orchestrator import SessionOrchestratorAgent
from neural_forge.core.pipeline import FrameProcessingResult, StepProgressSnapshot
from neural_forge.core.settings import Settings
from neural_forge.schemas.findings import StepStatus
from neural_forge.schemas.messages import ControlAction, FrameMessage, StepPreviewMessage
from neural_forge.schemas.protocols import AreaProtocol, ProtocolStep
from neural_forge.schemas.readings import QualityAssessment, QualityFlag
from tests.synthetic import config_repository

SESSION_ID = "0123456789abcdef0123456789abcdef"


def step_definition(step_id: str, number: int, skippable: bool = False, repeatable: bool = False) -> dict[str, Any]:
    return {
        "id": step_id,
        "step_number": number,
        "title": {"en": step_id.replace("_", " ")},
        "instruction": {"en": f"Do {step_id}."},
        "camera_setup": "face",
        "get_ready_seconds": 0,
        "min_duration_seconds": 0.1,
        "max_duration_seconds": 0.4,
        "completion_criteria": {"type": "valid_frames", "min_valid_frames": 1, "description": "frames"},
        "active_agents": [],
        "skippable": skippable,
        "repeatable": repeatable,
        "max_repetitions": 3 if repeatable else 1,
    }


def make_protocol(repeatable_last: bool = False) -> AreaProtocol:
    return AreaProtocol.model_validate(
        {
            "area": "face",
            "version": "test",
            "title": {"en": "Test"},
            "description": {"en": "Test protocol"},
            "steps": [
                step_definition("get_positioned", 1),
                step_definition("middle", 2, skippable=True),
                step_definition("last", 3, repeatable=repeatable_last),
            ],
        }
    )


class FakeQuality:
    """Quality stand-in that accepts every frame."""

    confirmation_frames = 1

    def guidance_for(self, issues: list[str]) -> list[str]:
        return [f"guidance for {issue}" for issue in issues]


class FakePipeline:
    """Minimal pipeline used to exercise the orchestrator without models."""

    def __init__(self, protocol: AreaProtocol, settings: Settings, never_complete: set[str] | None = None) -> None:
        self.protocol = protocol
        self.frame_ingest = FrameIngestAgent(SESSION_ID, settings)
        self.messages = config_repository().ui_messages()
        self.language = "en"
        self.quality = FakeQuality()
        self.never_complete = never_complete or set()
        self.begun: list[str] = []
        self.ended: list[tuple[str, bool]] = []

    async def process_frame(self, raw_frame: RawFrame, step: ProtocolStep, step_instance: str, collect: bool) -> FrameProcessingResult:
        quality = QualityAssessment(usable=True, quality_flag=QualityFlag.GOOD, score=1.0, subject_present=True)
        return FrameProcessingResult(frame_index=raw_frame.sequence, quality=quality, readings=[])

    def step_progress(self, step: ProtocolStep, step_instance: str, valid_frames: int) -> StepProgressSnapshot:
        if step.id in self.never_complete:
            return StepProgressSnapshot(fraction=0.2, complete=False, reasons=["The test agent never has enough data."])
        fraction = min(1.0, valid_frames / step.completion_criteria.min_valid_frames)
        return StepProgressSnapshot(fraction=fraction, complete=fraction >= 1.0, reasons=[])

    def begin_step(self, step: ProtocolStep, step_instance: str) -> None:
        self.begun.append(step_instance)

    def end_step(self, step: ProtocolStep, step_instance: str, completed: bool) -> None:
        self.ended.append((step_instance, completed))

    def set_reference_card_hint(self, present: bool) -> None:
        self.card_hint = present


def fast_settings() -> Settings:
    return Settings(
        start_timeout_seconds=5.0,
        ready_timeout_seconds=5.0,
        quality_wait_timeout_seconds=2.0,
        progress_interval_seconds=0.05,
        quality_feedback_interval_seconds=0.05,
        repeatable_offer_timeout_seconds=0.3,
        frame_starvation_seconds=5.0,
    )


async def feed_frames(ingest: FrameIngestAgent) -> None:
    timestamp = 1000.0
    while True:
        timestamp += 33.0
        ingest.submit(FrameMessage(type="frame", session_id=SESSION_ID, timestamp_ms=timestamp, image_base64="aGVsbG8gd29ybGQgaGVsbG8="))
        await asyncio.sleep(0.02)


async def run_examination(pipeline: FakePipeline, settings: Settings, on_message: Any = None) -> tuple[Any, list[Any]]:
    sent: list[Any] = []
    orchestrator_holder: dict[str, SessionOrchestratorAgent] = {}

    async def send(message: Any) -> None:
        sent.append(message)
        if on_message is not None:
            on_message(message, orchestrator_holder["agent"])

    orchestrator = SessionOrchestratorAgent(pipeline, send, settings)  # type: ignore[arg-type]
    orchestrator_holder["agent"] = orchestrator
    feeder = asyncio.create_task(feed_frames(pipeline.frame_ingest))
    orchestrator.submit_control(ControlAction.START)
    try:
        outcome = await asyncio.wait_for(orchestrator.run(), timeout=15.0)
    finally:
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
    return outcome, sent


@pytest.mark.asyncio
async def test_all_steps_complete_in_order() -> None:
    pipeline = FakePipeline(make_protocol(), fast_settings())
    outcome, sent = await run_examination(pipeline, fast_settings())
    assert outcome.started and not outcome.stopped
    assert [record.status for record in outcome.step_records] == [StepStatus.COMPLETED] * 3
    types = [message.type for message in sent]
    assert types[0] == "step_preview"
    assert "step_started" in types and "step_progress" in types and "step_complete" in types
    assert types[-1] == "examination_complete"
    assert pipeline.begun == ["get_positioned", "middle", "last"]


@pytest.mark.asyncio
async def test_skippable_step_can_be_skipped() -> None:
    def on_message(message: Any, orchestrator: SessionOrchestratorAgent) -> None:
        if isinstance(message, StepPreviewMessage) and message.step_id == "middle":
            orchestrator.submit_control(ControlAction.SKIP_STEP)

    outcome, _ = await run_examination(FakePipeline(make_protocol(), fast_settings()), fast_settings(), on_message)
    statuses = {record.step_id: record.status for record in outcome.step_records}
    assert statuses["middle"] == StepStatus.SKIPPED
    assert statuses["last"] == StepStatus.COMPLETED


@pytest.mark.asyncio
async def test_failed_step_is_repeated_once_then_incomplete() -> None:
    pipeline = FakePipeline(make_protocol(), fast_settings(), never_complete={"middle"})
    outcome, sent = await run_examination(pipeline, fast_settings())
    middle = next(record for record in outcome.step_records if record.step_id == "middle")
    assert middle.status == StepStatus.INCOMPLETE
    assert middle.attempts == 2
    assert any(message.type == "step_repeat" for message in sent)
    assert outcome.step_records[-1].status == StepStatus.COMPLETED


@pytest.mark.asyncio
async def test_stop_marks_remaining_steps() -> None:
    def on_message(message: Any, orchestrator: SessionOrchestratorAgent) -> None:
        if isinstance(message, StepPreviewMessage) and message.step_id == "middle":
            orchestrator.submit_control(ControlAction.STOP)

    outcome, sent = await run_examination(FakePipeline(make_protocol(), fast_settings()), fast_settings(), on_message)
    statuses = {record.step_id: record.status for record in outcome.step_records}
    assert outcome.stopped
    assert statuses["get_positioned"] == StepStatus.COMPLETED
    assert statuses["middle"] == StepStatus.INCOMPLETE
    assert statuses["last"] == StepStatus.NOT_REACHED
    assert sent[-1].type == "examination_complete" and sent[-1].stopped


@pytest.mark.asyncio
async def test_repeatable_step_runs_again_on_add_area() -> None:
    added = {"count": 0}

    def on_message(message: Any, orchestrator: SessionOrchestratorAgent) -> None:
        if message.type == "step_complete" and message.step_id == "last" and message.can_add_area and added["count"] == 0:
            added["count"] += 1
            orchestrator.submit_control(ControlAction.ADD_AREA)

    outcome, _ = await run_examination(FakePipeline(make_protocol(repeatable_last=True), fast_settings()), fast_settings(), on_message)
    instances = [record.step_instance for record in outcome.step_records if record.step_id == "last"]
    assert instances == ["last", "last__2"]


@pytest.mark.asyncio
async def test_pause_freezes_step_timer() -> None:
    paused = {"done": False}

    def on_message(message: Any, orchestrator: SessionOrchestratorAgent) -> None:
        if message.type == "step_started" and message.step_id == "middle" and not paused["done"]:
            paused["done"] = True
            orchestrator.submit_control(ControlAction.PAUSE)
            asyncio.get_running_loop().call_later(0.3, orchestrator.submit_control, ControlAction.RESUME)

    outcome, sent = await run_examination(FakePipeline(make_protocol(), fast_settings()), fast_settings(), on_message)
    assert any(message.type == "step_progress" and message.paused for message in sent)
    assert all(record.status == StepStatus.COMPLETED for record in outcome.step_records)
