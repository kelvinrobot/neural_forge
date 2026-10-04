"""Per-session examination pipeline: ingest, quality, calibration, landmarks, area agents and analysis."""

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from neural_forge.agents.area_registry import get_area_specification
from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.agents.core.aggregator import AggregatorAgent
from neural_forge.agents.core.colour_calibration import ColourCalibrationAgent
from neural_forge.agents.core.frame_ingest import FrameIngestAgent, RawFrame
from neural_forge.agents.core.quality import QualityAgent
from neural_forge.agents.core.reasoning import ReasoningAgent
from neural_forge.agents.core.report import ReportAgent
from neural_forge.agents.core.safety import SafetyAgent
from neural_forge.core.exceptions import FrameRejectedError, ModelInferenceError, ModelNotAvailableError, ProtocolError
from neural_forge.core.executor import InferenceExecutor
from neural_forge.core.logging import get_logger, log_event
from neural_forge.core.protocol_loader import ConfigRepository
from neural_forge.core.settings import Settings
from neural_forge.models.registry import ModelRegistry
from neural_forge.schemas.findings import QualitySummary, SessionFindings, StepRecord
from neural_forge.schemas.protocols import ProtocolStep
from neural_forge.schemas.readings import AgentResult, QualityAssessment, QualityFlag, Reading
from neural_forge.schemas.reports import FinalReport
from neural_forge.schemas.sessions import SessionState
from neural_forge.services.landmark_service import LandmarkService, TrackerName

logger = get_logger(__name__)


@dataclass(frozen=True)
class FrameProcessingResult:
    """Outcome of processing one frame."""

    frame_index: int
    quality: QualityAssessment
    readings: list[Reading]


@dataclass(frozen=True)
class StepProgressSnapshot:
    """How far the completion criteria of a running step are met."""

    fraction: float
    complete: bool
    reasons: list[str]


class ExaminationPipeline:
    """Owns all per-session processing state; frames for one session are processed sequentially."""

    def __init__(
        self,
        session: SessionState,
        settings: Settings,
        config: ConfigRepository,
        registry: ModelRegistry,
        executor: InferenceExecutor,
        reasoning_agent: ReasoningAgent,
        safety_agent: SafetyAgent,
        report_agent: ReportAgent,
    ) -> None:
        self._session_id = session.session_id
        self._language = session.language
        self._patient_context = session.patient_context
        self._settings = settings
        self._config = config
        self._executor = executor
        self._reasoning = reasoning_agent
        self._safety = safety_agent
        self._report = report_agent
        self.area = session.area
        self.protocol = config.protocol(session.area)
        self.messages = config.ui_messages()
        quality_config = config.quality_config()
        specification = get_area_specification(session.area)
        self.frame_ingest = FrameIngestAgent(session.session_id, settings)
        self.quality = QualityAgent(quality_config, session.language)
        self.colour: ColourCalibrationAgent | None = (
            ColourCalibrationAgent(quality_config["colour_calibration"]) if specification.needs_colour_calibration else None
        )
        context = AgentContext(
            area=session.area.value,
            protocol=self.protocol,
            thresholds=config.thresholds(session.area),
            zero_shot_prompts=config.zero_shot_prompts(),
            registry=registry,
            settings=settings,
        )
        self.agents: list[AreaAgent] = specification.build_agents(context)
        self._agents_by_name = {agent.name: agent for agent in self.agents}
        for step in self.protocol.steps:
            for agent_name in step.active_agents:
                if agent_name not in self._agents_by_name:
                    raise ProtocolError(f"Step '{step.id}' references unknown agent '{agent_name}' for area '{self.area.value}'")
        enabled_trackers: set[TrackerName] = set()
        for agent in self.agents:
            enabled_trackers.update(agent.required_trackers)
        for step in self.protocol.steps:
            enabled_trackers.update(self.quality.trackers_for_setup(step.camera_setup))
        self.landmarks = LandmarkService(registry, frozenset(enabled_trackers))
        self.aggregator = AggregatorAgent(
            session.session_id,
            session.area.value,
            quality_config["aggregation"],
            self._safety.disclaimer(session.language),
        )
        self._frame_index = 0
        self._card_hint = False
        self._tracker_failures: set[str] = set()

    @property
    def session_id(self) -> str:
        """Session identifier."""
        return self._session_id

    @property
    def language(self) -> str:
        """Patient's preferred language."""
        return self._language

    def set_reference_card_hint(self, present: bool) -> None:
        """Record the client's hint that a reference card is visible."""
        self._card_hint = present

    def active_agents(self, step: ProtocolStep) -> list[AreaAgent]:
        """Agents that collect data during a step."""
        return [self._agents_by_name[name] for name in step.active_agents if name in self._agents_by_name]

    def _trackers_for(self, step: ProtocolStep, collect: bool) -> set[TrackerName]:
        trackers = set(self.quality.trackers_for_setup(step.camera_setup))
        if collect:
            for agent in self.active_agents(step):
                trackers.update(agent.required_trackers)
        return trackers

    async def process_frame(self, raw_frame: RawFrame, step: ProtocolStep, step_instance: str, collect: bool) -> FrameProcessingResult:
        """Process one frame in the worker pool."""
        self._frame_index += 1
        return await self._executor.run(self._process_frame_sync, raw_frame, self._frame_index, step, step_instance, collect)

    def _process_frame_sync(
        self, raw_frame: RawFrame, frame_index: int, step: ProtocolStep, step_instance: str, collect: bool
    ) -> FrameProcessingResult:
        try:
            frame = self.frame_ingest.decode(raw_frame, frame_index)
        except FrameRejectedError as error:
            quality = self.quality.rejected_assessment(error.message)
            self.aggregator.record_frame_quality(quality)
            return FrameProcessingResult(frame_index=frame_index, quality=quality, readings=[])
        frame.protocol_step = step.id
        frame.step_instance = step_instance
        frame.allow_motion = step.allow_motion
        try:
            self.landmarks.run(frame, self._trackers_for(step, collect))
        except (ModelNotAvailableError, ModelInferenceError) as error:
            if error.message not in self._tracker_failures:
                self._tracker_failures.add(error.message)
                log_event(logger, logging.ERROR, "tracker_failure", session_id=self._session_id, error=error.message)
        frame.quality = self.quality.assess(frame, step.camera_setup, step.allow_motion)
        if self.colour is not None:
            self.colour.apply(frame, capture_reference=step.captures_colour_reference and collect, card_hint=self._card_hint)
        readings: list[Reading] = []
        if collect and frame.is_usable:
            for agent in self.active_agents(step):
                try:
                    readings.extend(agent.process_frame(frame))
                except (ModelNotAvailableError, ModelInferenceError) as error:
                    log_event(logger, logging.WARNING, "agent_model_error", agent=agent.name, error=error.message)
                except (ValueError, IndexError, FloatingPointError, cv2.error) as error:
                    log_event(logger, logging.WARNING, "agent_measurement_error", agent=agent.name, error=str(error))
            self.aggregator.add_readings(readings)
        self.aggregator.record_frame_quality(frame.quality)
        return FrameProcessingResult(frame_index=frame_index, quality=frame.quality, readings=readings)

    def begin_step(self, step: ProtocolStep, step_instance: str) -> None:
        """Notify active agents that data collection for a step instance starts."""
        start_timestamp = self.frame_ingest.latest_timestamp_s
        for agent in self.active_agents(step):
            agent.on_step_start(step_instance, start_timestamp)

    def end_step(self, step: ProtocolStep, step_instance: str, completed: bool) -> None:
        """Notify active agents that a step instance ended."""
        for agent in self.active_agents(step):
            agent.on_step_end(step_instance, completed)

    def step_progress(self, step: ProtocolStep, step_instance: str, valid_frames: int) -> StepProgressSnapshot:
        """Combine orchestrator frame counts, calibration progress and agent sufficiency."""
        criteria = step.completion_criteria
        fractions = [min(1.0, valid_frames / max(1, criteria.min_valid_frames))]
        reasons: list[str] = []
        if valid_frames < criteria.min_valid_frames:
            reasons.append(
                self.messages.text(
                    "progress_frames_missing", self._language, captured=valid_frames, required=criteria.min_valid_frames
                )
            )
        if step.captures_colour_reference and self.colour is not None:
            reference_fraction = self.colour.reference_progress()
            fractions.append(reference_fraction)
            if reference_fraction < 1.0:
                reasons.append(self.messages.text("reference_card_missing", self._language))
        for agent in self.active_agents(step):
            progress = agent.data_progress(step_instance)
            fractions.append(progress)
            if progress < 1.0:
                reason = agent.insufficiency_reason(step_instance)
                if reason:
                    reasons.append(reason)
        fraction = float(min(fractions)) if fractions else 1.0
        return StepProgressSnapshot(fraction=fraction, complete=fraction >= 1.0, reasons=reasons)

    def _finalize_agents(self) -> list[AgentResult]:
        results: list[AgentResult] = []
        for agent in self.agents:
            try:
                results.append(agent.finalize())
            except Exception as error:
                log_event(logger, logging.ERROR, "agent_finalize_failed", agent=agent.name, error=str(error))
                results.append(
                    AgentResult(
                        agent=agent.name,
                        area=agent.area,
                        confidence=0.0,
                        quality_flag=QualityFlag.UNUSABLE,
                        protocol_step="none",
                        timestamp=0.0,
                        description=f"The {agent.display_name} analysis could not be completed.",
                        limitations=[f"Analysis error: {error}"],
                        status="error",
                    )
                )
        return results

    def _quality_summary(self) -> QualitySummary:
        statistics = self.frame_ingest.statistics()
        assessed, usable, mean_brightness, mean_blur, common_issues = self.aggregator.quality_counts()
        method, calibration_confidence = self.colour.summary() if self.colour is not None else ("none", 0.0)
        return QualitySummary(
            frames_received=statistics.received,
            frames_dropped=statistics.dropped,
            frames_rejected=statistics.rejected + statistics.decode_errors,
            frames_processed=assessed,
            frames_usable=usable,
            usable_fraction=usable / assessed if assessed else 0.0,
            measured_fps=statistics.measured_fps,
            mean_brightness=mean_brightness,
            mean_blur_variance=mean_blur,
            common_issues=common_issues,
            calibration_method=method,
            calibration_confidence=calibration_confidence,
        )

    def _save_crops(self, findings: SessionFindings) -> None:
        target_directory = Path(self._settings.crops_dir) / self._session_id
        target_directory.mkdir(parents=True, exist_ok=True)
        for crop in findings.evidence:
            if not crop.image_jpeg_base64:
                continue
            image_bytes = base64.b64decode(crop.image_jpeg_base64)
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            path = target_directory / f"{crop.crop_id}.jpg"
            cv2.imwrite(str(path), image)
            crop.saved_path = str(path)

    async def analyse(self, step_records: list[StepRecord], stopped_early: bool) -> tuple[SessionFindings, FinalReport]:
        """Finalise agents, aggregate, reason, apply safety rules and build the report."""
        agent_results = await self._executor.run(self._finalize_agents)
        findings = self.aggregator.build_findings(
            self._patient_context, step_records, agent_results, self._quality_summary(), stopped_early
        )
        rule_summary_texts = {
            "flags": self.messages.template("rule_summary_flags", self._language),
            "none": self.messages.template("rule_summary_none", self._language),
        }
        reasoning_result = await self._reasoning.interpret(findings, self.protocol, rule_summary_texts)
        interpretation = self._safety.review(reasoning_result, findings, self._language)
        if self._settings.save_report_crops:
            await self._executor.run(self._save_crops, findings)
        report = self._report.build(findings, interpretation, self.protocol, self._language)
        return findings, report

    def close(self) -> None:
        """Release per-session trackers and clear buffers."""
        self.landmarks.close()
        for agent in self.agents:
            agent.reset()
