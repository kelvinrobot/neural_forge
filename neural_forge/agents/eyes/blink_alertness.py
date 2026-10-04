"""Blink and alertness agent: eye aspect ratio, blink rate, blink duration and PERCLOS."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import eye_aspect_ratio
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import LEFT_EYE_EAR_POINTS, RIGHT_EYE_EAR_POINTS
from neural_forge.services.signal_processing import time_weighted_fraction


@dataclass
class BlinkStatistics:
    """Blink metrics for one step instance."""

    blink_count: int
    durations: list[float]
    prolonged_closures: int
    perclos: float
    duration_seconds: float

    @property
    def blink_rate_per_minute(self) -> float:
        """Blinks per minute using the measured duration."""
        if self.duration_seconds <= 0:
            return float("nan")
        return self.blink_count / (self.duration_seconds / 60.0)


class BlinkAlertnessAgent(AreaAgent):
    """Detects blinks from the eye aspect ratio with hysteresis using real timestamps."""

    name = "blink_alertness"
    area = "eyes"
    required_trackers = frozenset({TrackerName.FACE})
    supported_events = frozenset({"blinks"})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._series: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record the mean eye aspect ratio of both eyes."""
        face = frame_context.face
        if face is None:
            return []
        right = eye_aspect_ratio(face.points(RIGHT_EYE_EAR_POINTS))
        left = eye_aspect_ratio(face.points(LEFT_EYE_EAR_POINTS))
        if not np.isfinite(right) or not np.isfinite(left):
            return []
        mean_ratio = (right + left) / 2.0
        self._series[frame_context.step_instance or ""].append((frame_context.timestamp_s, mean_ratio))
        return [self.make_reading(frame_context, "eye_aspect_ratio", mean_ratio, "ratio", "Mean eye opening (eye aspect ratio) of both eyes.")]

    def _statistics(self, series: list[tuple[float, float]]) -> BlinkStatistics | None:
        if len(series) < 10:
            return None
        timestamps = np.array([item[0] for item in series])
        ratios = np.array([item[1] for item in series])
        open_reference = float(np.percentile(ratios, self.threshold_float("open_reference_percentile")))
        if open_reference <= 0:
            return None
        close_level = self.threshold_float("close_ratio") * open_reference
        reopen_level = self.threshold_float("reopen_ratio") * open_reference
        minimum_blink = self.threshold_float("min_blink_seconds")
        maximum_blink = self.threshold_float("max_blink_seconds")
        durations: list[float] = []
        prolonged = 0
        closed_since: float | None = None
        for timestamp, ratio in zip(timestamps, ratios):
            if closed_since is None and ratio < close_level:
                closed_since = float(timestamp)
            elif closed_since is not None and ratio > reopen_level:
                duration = float(timestamp) - closed_since
                if minimum_blink <= duration <= maximum_blink:
                    durations.append(duration)
                elif duration > maximum_blink:
                    prolonged += 1
                closed_since = None
        closure = 1.0 - ratios / open_reference
        perclos = time_weighted_fraction(timestamps, closure >= self.threshold_float("perclos_closure_fraction"))
        return BlinkStatistics(
            blink_count=len(durations),
            durations=durations,
            prolonged_closures=prolonged,
            perclos=perclos,
            duration_seconds=float(timestamps[-1] - timestamps[0]),
        )

    def count_events(self, step_instance: str, event: str) -> int:
        """Number of blinks detected so far in the step instance."""
        if event != "blinks":
            return 0
        statistics = self._statistics(self._series.get(step_instance, []))
        return statistics.blink_count if statistics is not None else 0

    def finalize(self) -> AgentResult:
        """Compute blink rate, blink duration and PERCLOS, and screen for drowsiness."""
        analysis_steps = {str(step) for step in self.threshold_list("analysis_steps")}
        combined: list[tuple[float, float]] = []
        for step_instance, series in self._series.items():
            if step_instance.split("__")[0] in analysis_steps:
                combined.extend(series)
        statistics = self._statistics(combined)
        if statistics is None or statistics.duration_seconds < self.threshold_float("min_duration_seconds"):
            return self.insufficient_result("Not enough continuous eye tracking to measure blinking.")
        confidence = self.data_confidence(list(analysis_steps))
        steps = ",".join(sorted(analysis_steps))
        blink_rate = statistics.blink_rate_per_minute
        mean_duration = float(np.mean(statistics.durations)) if statistics.durations else float("nan")
        metrics = [
            self.summary_reading("blink_rate", blink_rate, "blinks/min", "Blinks per minute.", confidence, steps),
            self.summary_reading("mean_blink_duration", mean_duration, "s", "Average blink duration.", confidence, steps),
            self.summary_reading("perclos", statistics.perclos, "fraction", "Fraction of time the eyes were at least 80% closed (PERCLOS).", confidence, steps),
            self.summary_reading("prolonged_eye_closures", float(statistics.prolonged_closures), "count", "Eye closures longer than a normal blink.", confidence, steps),
        ]
        flags: list[ScreeningFlag] = []
        perclos_threshold = self.threshold_float("perclos_drowsy")
        long_blink = self.threshold_float("long_blink_seconds")
        drowsy_reasons: list[str] = []
        if statistics.perclos >= perclos_threshold:
            drowsy_reasons.append(f"PERCLOS {statistics.perclos:.2f}")
        if np.isfinite(mean_duration) and mean_duration >= long_blink:
            drowsy_reasons.append(f"mean blink duration {mean_duration:.2f} s")
        if statistics.prolonged_closures > 0:
            drowsy_reasons.append(f"{statistics.prolonged_closures} prolonged eye closure(s)")
        if drowsy_reasons:
            flags.append(
                self.make_flag(
                    "drowsiness_signs",
                    "Signs of drowsiness",
                    FlagSeverity.MODERATE,
                    max(
                        self.exceedance_confidence(statistics.perclos, perclos_threshold, confidence) if statistics.perclos >= perclos_threshold else 0.0,
                        confidence * 0.6,
                    ),
                    "Eye closure patterns suggested reduced alertness.",
                    sorted(analysis_steps),
                    evidence=drowsy_reasons,
                )
            )
        low_rate = self.threshold_float("blink_rate_low_per_minute")
        high_rate = self.threshold_float("blink_rate_high_per_minute")
        if np.isfinite(blink_rate) and blink_rate < low_rate:
            flags.append(
                self.make_flag(
                    "reduced_blink_rate",
                    "Reduced blink rate",
                    FlagSeverity.LOW,
                    self.shortfall_confidence(blink_rate, low_rate, confidence),
                    "Blinking was less frequent than usual, which can accompany reduced facial movement or intense focus.",
                    sorted(analysis_steps),
                    evidence=[f"{blink_rate:.1f} blinks/min"],
                )
            )
        elif np.isfinite(blink_rate) and blink_rate > high_rate:
            flags.append(
                self.make_flag(
                    "increased_blink_rate",
                    "Increased blink rate",
                    FlagSeverity.LOW,
                    self.exceedance_confidence(blink_rate, high_rate, confidence),
                    "Blinking was more frequent than usual, which can accompany eye irritation, dryness or fatigue.",
                    sorted(analysis_steps),
                    evidence=[f"{blink_rate:.1f} blinks/min"],
                )
            )
        return self.build_result(
            description=f"Blink rate {blink_rate:.1f} per minute, PERCLOS {statistics.perclos:.2f}.",
            metrics=metrics,
            flags=flags,
            confidence=confidence,
            value=blink_rate,
            unit="blinks/min",
        )

    def _reset_measurements(self) -> None:
        self._series.clear()
