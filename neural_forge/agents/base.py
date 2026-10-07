"""base class and shared helpers for area agents."""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from neural_forge.core.exceptions import ConfigurationError, ModelInferenceError, ModelNotAvailableError
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.core.settings import Settings
from neural_forge.models.registry import ModelRegistry
from neural_forge.schemas.protocols import AreaProtocol, CompletionType, ProtocolStep
from neural_forge.schemas.readings import (
    AgentResult,
    ClassificationResult,
    EvidenceCrop,
    FlagSeverity,
    QualityFlag,
    Reading,
    ScreeningFlag,
    Side,
)
from neural_forge.services.colour_utils import encode_jpeg_base64
from neural_forge.services.landmark_service import TrackerName


@dataclass
class AgentContext:
    """Everything an area agent needs at construction time."""

    area: str
    protocol: AreaProtocol
    thresholds: dict[str, Any]
    zero_shot_prompts: dict[str, Any]
    registry: ModelRegistry
    settings: Settings


@dataclass
class CropCandidate:
    """An image crop kept for later classification or as evidence."""

    image: np.ndarray
    sharpness: float
    protocol_step: str
    timestamp: float
    label: str


class BestCropBuffer:
    """Keeps the sharpest crops offered, up to a fixed capacity."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._items: list[CropCandidate] = []

    def offer(self, candidate: CropCandidate) -> None:
        """Add a candidate, keeping only the sharpest ones."""
        if candidate.image.size == 0:
            return
        self._items.append(candidate)
        self._items.sort(key=lambda item: item.sharpness, reverse=True)
        del self._items[self._capacity :]

    def items(self) -> list[CropCandidate]:
        """Return kept crops, sharpest first."""
        return list(self._items)

    def clear(self) -> None:
        """Remove all crops."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


def finite_or_none(value: float | None) -> float | None:
    """Return the value as float, or None when missing or not finite."""
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def clip_unit(value: float) -> float:
    """Clip a value to [0, 1], mapping NaN to 0."""
    if value is None or math.isnan(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def robust_median(values: list[float]) -> float:
    """Median of finite values, NaN when empty."""
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.median(finite_values))


class AreaAgent(ABC):
    """Common interface for all area agents; the orchestrator drives agents only through it."""

    name: ClassVar[str]
    area: ClassVar[str]
    required_trackers: ClassVar[frozenset[TrackerName]] = frozenset()
    supported_events: ClassVar[frozenset[str]] = frozenset()
    supported_states: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, context: AgentContext) -> None:
        self._context = context
        self._settings = context.settings
        self._registry = context.registry
        agent_config = context.thresholds.get(self.name)
        if not isinstance(agent_config, dict):
            raise ConfigurationError(f"Thresholds for agent '{self.name}' are missing in thresholds/{self.area}.yaml")
        self._config: dict[str, Any] = agent_config
        self._step_definitions: dict[str, ProtocolStep] = {
            step.id: step for step in context.protocol.steps if self.name in step.active_agents
        }
        self._valid_frames: dict[str, int] = defaultdict(int)
        self._quality_scores: dict[str, list[float]] = defaultdict(list)
        self._step_start_times: dict[str, float] = {}
        self._step_outcomes: dict[str, bool] = {}
        self._last_timestamp: float = 0.0
        self._peers: dict[str, "AreaAgent"] = {}
        self._crop_counter = 0

    @property
    def active_protocol_steps(self) -> list[str]:
        """Protocol step ids during which this agent collects data."""
        return list(self._step_definitions.keys())

    @property
    def display_name(self) -> str:
        """Human-readable agent name."""
        return self.name.replace("_", " ")

    def is_active_for(self, step_id: str) -> bool:
        """Return whether the agent collects data during the step (or a repetition of it)."""
        return base_step_id(step_id) in self._step_definitions

    def step_definition(self, step_instance: str) -> ProtocolStep | None:
        """Return the protocol step definition for a step instance."""
        return self._step_definitions.get(base_step_id(step_instance))

    def bind_peers(self, peers: dict[str, "AreaAgent"]) -> None:
        """Receive references to the other agents of the same area."""
        self._peers = {agent_name: agent for agent_name, agent in peers.items() if agent_name != self.name}

    def threshold(self, key: str) -> Any:
        """Return a configured value for this agent or raise if it is missing."""
        if key not in self._config:
            raise ConfigurationError(f"Threshold '{key}' for agent '{self.name}' is missing in thresholds/{self.area}.yaml")
        return self._config[key]

    def threshold_float(self, key: str) -> float:
        """Return a configured numeric value as float."""
        return float(self.threshold(key))

    def threshold_int(self, key: str) -> int:
        """Return a configured numeric value as int."""
        return int(self.threshold(key))

    def threshold_range(self, key: str) -> tuple[float, float]:
        """Return a configured two-element range."""
        value = self.threshold(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ConfigurationError(f"Threshold '{key}' for agent '{self.name}' must be a two-element list")
        return float(value[0]), float(value[1])

    def threshold_list(self, key: str) -> list[Any]:
        """Return a configured list."""
        value = self.threshold(key)
        if not isinstance(value, list):
            raise ConfigurationError(f"Threshold '{key}' for agent '{self.name}' must be a list")
        return value

    def process_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure one usable frame during an active step and count it as valid when it yields readings."""
        if frame_context.protocol_step is None or frame_context.step_instance is None:
            return []
        if not self.is_active_for(frame_context.protocol_step) or not frame_context.is_usable:
            return []
        readings = self.measure_frame(frame_context)
        if readings:
            self._valid_frames[frame_context.step_instance] += 1
            self._quality_scores[frame_context.step_instance].append(frame_context.quality_score)
            self._last_timestamp = frame_context.timestamp_s
        return readings

    @abstractmethod
    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Extract measurements from one frame; return an empty list when the frame cannot be used."""

    def on_step_start(self, step_instance: str, timestamp: float) -> None:
        """Record the start time of a step instance (called again on a repeat attempt)."""
        self._step_start_times[step_instance] = timestamp

    def on_step_end(self, step_instance: str, completed: bool) -> None:
        """Record whether a step instance completed."""
        self._step_outcomes[step_instance] = completed

    def step_elapsed(self, step_instance: str, timestamp: float) -> float:
        """Seconds since the step instance started, using measured frame timestamps."""
        start_time = self._step_start_times.get(step_instance)
        if start_time is None:
            return 0.0
        return max(0.0, timestamp - start_time)

    def valid_frame_count(self, step_instance: str) -> int:
        """Number of frames that produced readings during a step instance."""
        return self._valid_frames.get(step_instance, 0)

    def required_valid_frames(self, step_instance: str) -> int:
        """Number of valid frames required by the step's completion criteria."""
        step = self.step_definition(step_instance)
        if step is None:
            return 0
        return step.completion_criteria.min_valid_frames

    def count_events(self, step_instance: str, event: str) -> int:
        """Number of detected events of a kind during a step; agents with events override this."""
        return 0

    def state_reached(self, step_instance: str, state: str) -> bool:
        """Whether a completion state was detected; agents with states override this."""
        return False

    def data_progress(self, step_instance: str) -> float:
        """Fraction (0-1) of this agent's completion criteria met for the step instance."""
        step = self.step_definition(step_instance)
        if step is None:
            return 1.0
        criteria = step.completion_criteria
        required_frames = max(1, criteria.min_valid_frames)
        frame_progress = min(1.0, self.valid_frame_count(step_instance) / required_frames)
        if criteria.type == CompletionType.EVENTS and criteria.event in self.supported_events:
            required_events = max(1, criteria.min_events or 1)
            event_progress = min(1.0, self.count_events(step_instance, criteria.event or "") / required_events)
            return min(frame_progress, event_progress)
        if criteria.type == CompletionType.STATE and criteria.state in self.supported_states:
            if self.state_reached(step_instance, criteria.state or ""):
                return frame_progress
            return min(frame_progress, 0.95)
        return frame_progress

    def has_sufficient_data(self, step_id: str) -> bool:
        """Whether the agent has collected enough data to end the step instance."""
        return self.data_progress(step_id) >= 1.0

    def insufficiency_reason(self, step_instance: str) -> str | None:
        """Plain-English explanation of what is still missing for the step instance."""
        step = self.step_definition(step_instance)
        if step is None:
            return None
        criteria = step.completion_criteria
        frames = self.valid_frame_count(step_instance)
        if frames < criteria.min_valid_frames:
            return f"The {self.display_name} check captured {frames} of {criteria.min_valid_frames} clear frames."
        if criteria.type == CompletionType.EVENTS and criteria.event in self.supported_events:
            detected = self.count_events(step_instance, criteria.event or "")
            if detected < (criteria.min_events or 0):
                return f"The {self.display_name} check detected {detected} of {criteria.min_events} {criteria.event}."
        if criteria.type == CompletionType.STATE and criteria.state in self.supported_states:
            if not self.state_reached(step_instance, criteria.state or ""):
                return f"The {self.display_name} check did not detect '{(criteria.state or '').replace('_', ' ')}'."
        return None

    @abstractmethod
    def finalize(self) -> AgentResult:
        """Summarise all collected data into an agent result with screening flags."""

    def data_confidence(self, step_ids: list[str]) -> float:
        """Confidence in the data of the given steps: mean frame quality scaled by frame sufficiency."""
        base_ids = {base_step_id(step_id) for step_id in step_ids}
        scores: list[float] = []
        sufficiency: list[float] = []
        for step_instance, instance_scores in self._quality_scores.items():
            if base_step_id(step_instance) not in base_ids or not instance_scores:
                continue
            scores.extend(instance_scores)
            required = max(1, self.required_valid_frames(step_instance))
            sufficiency.append(min(1.0, len(instance_scores) / required))
        if not scores:
            return 0.0
        return clip_unit(float(np.mean(scores)) * float(np.mean(sufficiency)))

    @staticmethod
    def exceedance_confidence(value: float, threshold: float, data_confidence: float) -> float:
        """Flag confidence growing with how far a value exceeds its threshold, scaled by data confidence."""
        if threshold == 0 or value is None or math.isnan(value):
            return 0.0
        margin = abs(value) / abs(threshold) - 1.0
        return clip_unit((0.5 + 0.5 * min(1.0, max(0.0, margin))) * data_confidence)

    @staticmethod
    def shortfall_confidence(value: float, threshold: float, data_confidence: float) -> float:
        """Flag confidence growing with how far a value falls below its threshold, scaled by data confidence."""
        if threshold == 0 or value is None or math.isnan(value):
            return 0.0
        margin = 1.0 - abs(value) / abs(threshold)
        return clip_unit((0.5 + 0.5 * min(1.0, max(0.0, margin * 2.0))) * data_confidence)

    def reset(self) -> None:
        """Clear all collected data and counters."""
        self._valid_frames.clear()
        self._quality_scores.clear()
        self._step_start_times.clear()
        self._step_outcomes.clear()
        self._last_timestamp = 0.0
        self._crop_counter = 0
        self._reset_measurements()

    @abstractmethod
    def _reset_measurements(self) -> None:
        """Clear agent-specific buffers."""

    @property
    def steps_with_data(self) -> list[str]:
        """Step instances that produced at least one valid frame."""
        return [step for step, count in self._valid_frames.items() if count > 0]

    @staticmethod
    def quality_from_confidence(confidence: float) -> QualityFlag:
        """Map a confidence score to a quality flag."""
        if confidence >= 0.7:
            return QualityFlag.GOOD
        if confidence >= 0.4:
            return QualityFlag.FAIR
        return QualityFlag.POOR

    def make_reading(
        self,
        frame: FrameContext,
        metric: str,
        value: float | None,
        unit: str,
        description: str,
        confidence: float = 1.0,
        side: Side | None = None,
        details: dict[str, Any] | None = None,
    ) -> Reading:
        """Create a per-frame reading weighted by the frame quality score."""
        quality_flag = frame.quality.quality_flag if frame.quality is not None else QualityFlag.POOR
        return Reading(
            agent=self.name,
            metric=metric,
            value=finite_or_none(value),
            unit=unit,
            confidence=clip_unit(confidence * frame.quality_score),
            quality_flag=quality_flag,
            protocol_step=frame.step_instance or frame.protocol_step or "",
            timestamp=frame.timestamp_s,
            description=description,
            side=side,
            details=details or {},
        )

    def summary_reading(
        self,
        metric: str,
        value: float | None,
        unit: str,
        description: str,
        confidence: float,
        protocol_step: str,
        side: Side | None = None,
        details: dict[str, Any] | None = None,
    ) -> Reading:
        """Create a session-level summary reading for an agent result."""
        bounded_confidence = clip_unit(confidence)
        return Reading(
            agent=self.name,
            metric=metric,
            value=finite_or_none(value),
            unit=unit,
            confidence=bounded_confidence,
            quality_flag=self.quality_from_confidence(bounded_confidence),
            protocol_step=protocol_step,
            timestamp=self._last_timestamp,
            description=description,
            side=side,
            details=details or {},
        )

    def make_flag(
        self,
        code: str,
        title: str,
        severity: FlagSeverity,
        confidence: float,
        description: str,
        protocol_steps: list[str],
        evidence: list[str] | None = None,
        side: Side | None = None,
    ) -> ScreeningFlag:
        """Create a screening flag."""
        return ScreeningFlag(
            code=code,
            title=title,
            severity=severity,
            confidence=clip_unit(confidence),
            description=description,
            agent=self.name,
            protocol_steps=protocol_steps,
            evidence=evidence or [],
            side=side,
        )

    def make_evidence_crop(
        self,
        label: str,
        candidate: CropCandidate,
        classification: ClassificationResult | None = None,
        needs_consultation: bool = False,
        consultation_question: str | None = None,
    ) -> EvidenceCrop:
        """Encode a crop as evidence for the report and the medical consultant."""
        self._crop_counter += 1
        return EvidenceCrop(
            crop_id=f"{self.name}-{self._crop_counter}",
            agent=self.name,
            label=label,
            protocol_step=candidate.protocol_step,
            timestamp=candidate.timestamp,
            image_jpeg_base64=encode_jpeg_base64(candidate.image, self._settings.report_crop_jpeg_quality),
            classification=classification,
            needs_consultation=needs_consultation,
            consultation_question=consultation_question,
        )

    def safe_classify(
        self, images: list[np.ndarray], task: str, limitations: list[str]
    ) -> list[ClassificationResult]:
        """Classify crops, recording a limitation instead of failing when the classifier is unavailable."""
        if not images:
            return []
        try:
            return self._registry.classifier().classify(images, task)
        except (ModelNotAvailableError, ModelInferenceError, ConfigurationError) as error:
            limitations.append(f"Image classification for '{task}' was unavailable: {error.message}")
            return []

    @staticmethod
    def mean_label_scores(results: list[ClassificationResult]) -> dict[str, float]:
        """Average label probabilities across several crops."""
        if not results:
            return {}
        labels = list(results[0].label_scores.keys())
        return {label: float(np.mean([result.label_scores.get(label, 0.0) for result in results])) for label in labels}

    def needs_consultation(self, result: ClassificationResult) -> bool:
        """Whether a classification is doubtful enough to ask the medical consultant."""
        low = self.threshold_float("consultation_probability_low")
        high = self.threshold_float("consultation_probability_high")
        return low <= result.top_score <= high

    def build_result(
        self,
        description: str,
        metrics: list[Reading],
        flags: list[ScreeningFlag],
        confidence: float,
        value: float | None = None,
        unit: str = "",
        evidence: list[EvidenceCrop] | None = None,
        limitations: list[str] | None = None,
    ) -> AgentResult:
        """Assemble an agent result with provenance fields."""
        steps = self.steps_with_data
        bounded_confidence = clip_unit(confidence) if steps else 0.0
        return AgentResult(
            agent=self.name,
            area=self.area,
            value=finite_or_none(value),
            unit=unit,
            confidence=bounded_confidence,
            quality_flag=self.quality_from_confidence(bounded_confidence) if steps else QualityFlag.UNUSABLE,
            protocol_step=",".join(steps) if steps else "none",
            timestamp=self._last_timestamp,
            description=description,
            metrics=metrics,
            flags=flags,
            evidence_crops=evidence or [],
            steps_with_data=steps,
            limitations=limitations or [],
            status="ok" if steps else "insufficient_data",
        )

    def insufficient_result(self, reason: str) -> AgentResult:
        """Agent result used when no usable data was collected."""
        return AgentResult(
            agent=self.name,
            area=self.area,
            confidence=0.0,
            quality_flag=QualityFlag.UNUSABLE,
            protocol_step="none",
            timestamp=self._last_timestamp,
            description=reason,
            steps_with_data=self.steps_with_data,
            limitations=[reason],
            status="insufficient_data",
        )
