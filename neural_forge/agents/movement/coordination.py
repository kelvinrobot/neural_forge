"""Coordination agent: finger-to-nose accuracy and terminal tremor, rapid hand flipping rhythm."""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.neural_forge_hands import HandLandmarks
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import asymmetry_ratio, euclidean_distance
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.signal_processing import coefficient_of_variation, find_peaks_irregular


def palm_orientation(hand: HandLandmarks) -> float:
    """Signed sine of the angle between wrist-to-index and wrist-to-little knuckle vectors; flips with pronation."""
    index_vector = hand.point(5) - hand.point(0)
    little_vector = hand.point(17) - hand.point(0)
    denominator = float(np.linalg.norm(index_vector) * np.linalg.norm(little_vector))
    if denominator < 1e-6:
        return float("nan")
    cross = float(index_vector[0] * little_vector[1] - index_vector[1] * little_vector[0])
    return cross / denominator


@dataclass
class NoseAnalysis:
    """Finger-to-nose results for one hand."""

    touches: int
    endpoint_error_cm: float
    terminal_tremor_cm: float


@dataclass
class FlipAnalysis:
    """Hand flipping results for one hand."""

    cycles: float
    rate_hz: float
    interval_cv: float
    amplitude: float


class CoordinationAgent(AreaAgent):
    """Measures cerebellar-style coordination features for each hand."""

    name = "coordination"
    area = "movement"
    required_trackers = frozenset({TrackerName.HANDS, TrackerName.POSE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._nose_series: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        self._flip_series: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def _mapping(self, key: str) -> dict[str, str]:
        return {str(step): str(side) for step, side in self.threshold(key).items()}

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record fingertip-to-nose distance or palm orientation depending on the step."""
        step_instance = frame_context.step_instance or ""
        step_id = base_step_id(step_instance)
        nose_steps = self._mapping("nose_steps")
        flip_steps = self._mapping("flipping_steps")
        if step_id in nose_steps:
            return self._measure_nose(frame_context, step_instance, nose_steps[step_id])
        if step_id in flip_steps:
            return self._measure_flip(frame_context, step_instance, flip_steps[step_id])
        return []

    def _measure_nose(self, frame_context: FrameContext, step_instance: str, side: str) -> list[Reading]:
        pose = frame_context.pose
        if pose is None:
            return []
        minimum_score = self.threshold_float("min_keypoint_score")
        if not pose.all_visible(("nose", "left_shoulder", "right_shoulder"), minimum_score):
            return []
        shoulder_width = euclidean_distance(pose.keypoint("left_shoulder"), pose.keypoint("right_shoulder"))
        if shoulder_width < 5.0:
            return []
        tip: np.ndarray | None = None
        hand = frame_context.hands.get(side) if frame_context.hands is not None else None  # type: ignore[arg-type]
        if hand is not None:
            tip = hand.point(8)
        elif pose.hand_keypoint_score(side, 8) >= minimum_score:
            tip = pose.hand_keypoint(side, 8)
        if tip is None:
            return []
        centimetres_per_pixel = self.threshold_float("shoulder_width_cm") / shoulder_width
        normalised_distance = euclidean_distance(tip, pose.keypoint("nose")) / shoulder_width
        self._nose_series[step_instance].append(
            (frame_context.timestamp_s, normalised_distance, float(tip[0] * centimetres_per_pixel), float(tip[1] * centimetres_per_pixel))
        )
        return [
            self.make_reading(frame_context, "fingertip_to_nose_distance", normalised_distance * self.threshold_float("shoulder_width_cm"), "cm", f"{side.capitalize()} fingertip distance from the nose.", side=side)  # type: ignore[arg-type]
        ]

    def _measure_flip(self, frame_context: FrameContext, step_instance: str, side: str) -> list[Reading]:
        hands = frame_context.hands
        if hands is None:
            return []
        hand = hands.get(side)  # type: ignore[arg-type]
        if hand is None:
            return []
        orientation = palm_orientation(hand)
        if not np.isfinite(orientation):
            return []
        self._flip_series[step_instance].append((frame_context.timestamp_s, orientation))
        return [self.make_reading(frame_context, "palm_orientation", orientation, "sine", f"{side.capitalize()} palm orientation signal.", side=side)]  # type: ignore[arg-type]

    def _analyse_nose(self, samples: list[tuple[float, float, float, float]]) -> NoseAnalysis | None:
        if len(samples) < 10:
            return None
        timestamps = np.array([sample[0] for sample in samples])
        distances = np.array([sample[1] for sample in samples])
        tips = np.array([[sample[2], sample[3]] for sample in samples])
        touch_limit = self.threshold_float("touch_distance_shoulder_fraction")
        minima = find_peaks_irregular(
            timestamps,
            -distances,
            self.threshold_float("min_touch_interval_seconds"),
            self.threshold_float("touch_prominence_shoulder_fraction"),
        )
        touches = [index for index in minima if distances[index] <= touch_limit]
        if not touches:
            return NoseAnalysis(touches=0, endpoint_error_cm=float("nan"), terminal_tremor_cm=float("nan"))
        shoulder_cm = self.threshold_float("shoulder_width_cm")
        errors = [float(distances[index]) * shoulder_cm for index in touches]
        window = self.threshold_float("terminal_window_seconds")
        terminal_amplitudes: list[float] = []
        for index in touches:
            mask = (timestamps >= timestamps[index] - window) & (timestamps <= timestamps[index])
            if mask.sum() < 5:
                continue
            window_times = timestamps[mask] - timestamps[mask][0]
            residual_power: list[float] = []
            for axis in range(2):
                coefficients = np.polyfit(window_times, tips[mask, axis], deg=2)
                residual = tips[mask, axis] - np.polyval(coefficients, window_times)
                residual_power.append(float(np.mean(residual ** 2)))
            terminal_amplitudes.append(math.sqrt(sum(residual_power)) * 2.0 * math.sqrt(2.0))
        return NoseAnalysis(
            touches=len(touches),
            endpoint_error_cm=float(np.median(errors)),
            terminal_tremor_cm=float(np.median(terminal_amplitudes)) if terminal_amplitudes else float("nan"),
        )

    def _analyse_flip(self, samples: list[tuple[float, float]]) -> FlipAnalysis | None:
        if len(samples) < 10:
            return None
        hysteresis = self.threshold_float("flip_hysteresis")
        values = np.array([sample[1] for sample in samples])
        center = float((np.percentile(values, 95) + np.percentile(values, 5)) / 2.0)
        state = 0
        transition_times: list[float] = []
        for timestamp, value in samples:
            relative = value - center
            if relative > hysteresis and state != 1:
                if state != 0:
                    transition_times.append(timestamp)
                state = 1
            elif relative < -hysteresis and state != -1:
                if state != 0:
                    transition_times.append(timestamp)
                state = -1
        if len(transition_times) < 2:
            return FlipAnalysis(cycles=len(transition_times) / 2.0, rate_hz=0.0, interval_cv=float("nan"), amplitude=float(np.ptp(values)))
        half_cycles = np.diff(transition_times)
        duration = transition_times[-1] - transition_times[0]
        cycles = len(half_cycles) / 2.0
        return FlipAnalysis(
            cycles=cycles,
            rate_hz=cycles / duration if duration > 0 else 0.0,
            interval_cv=coefficient_of_variation(half_cycles),
            amplitude=float(np.percentile(values, 95) - np.percentile(values, 5)),
        )

    def finalize(self) -> AgentResult:
        """Evaluate dysmetria, intention tremor and dysdiadochokinesia for each hand."""
        if not self.steps_with_data:
            return self.insufficient_result("Coordination tasks could not be tracked.")
        nose_steps = self._mapping("nose_steps")
        flip_steps = self._mapping("flipping_steps")
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        flip_results: dict[str, tuple[FlipAnalysis, str]] = {}
        nose_results: dict[str, tuple[NoseAnalysis, str]] = {}
        for step_instance, samples in self._nose_series.items():
            side = nose_steps.get(base_step_id(step_instance))
            analysis = self._analyse_nose(samples)
            if side is None or analysis is None:
                continue
            nose_results[side] = (analysis, step_instance)
        for step_instance, samples in self._flip_series.items():
            side = flip_steps.get(base_step_id(step_instance))
            analysis = self._analyse_flip(samples)
            if side is None or analysis is None:
                continue
            flip_results[side] = (analysis, step_instance)
        error_threshold = self.threshold_float("dysmetria_error_cm")
        tremor_threshold = self.threshold_float("intention_tremor_amplitude_cm")
        minimum_touches = self.threshold_int("min_touches")
        for side, (analysis, step_instance) in nose_results.items():
            confidence = self.data_confidence([step_instance])
            metrics.append(self.summary_reading("nose_touches", float(analysis.touches), "count", f"{side.capitalize()} finger-to-nose touches detected.", confidence, step_instance, side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("endpoint_error", analysis.endpoint_error_cm, "cm", f"{side.capitalize()} median distance from the nose at touch.", confidence, step_instance, side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("terminal_tremor", analysis.terminal_tremor_cm, "cm", f"{side.capitalize()} fingertip oscillation near the nose.", confidence, step_instance, side=side))  # type: ignore[arg-type]
            if analysis.touches < minimum_touches:
                limitations.append(f"Only {analysis.touches} finger-to-nose touches were detected for the {side} hand.")
                continue
            if np.isfinite(analysis.endpoint_error_cm) and analysis.endpoint_error_cm >= error_threshold:
                flags.append(
                    self.make_flag(
                        "dysmetria",
                        "Inaccurate finger-to-nose pointing",
                        FlagSeverity.MODERATE,
                        self.exceedance_confidence(analysis.endpoint_error_cm, error_threshold, confidence),
                        f"The {side} index finger often stopped away from the nose, which can indicate a coordination problem.",
                        [step_instance],
                        evidence=[f"median endpoint error {analysis.endpoint_error_cm:.1f} cm"],
                        side=side,  # type: ignore[arg-type]
                    )
                )
            if np.isfinite(analysis.terminal_tremor_cm) and analysis.terminal_tremor_cm >= tremor_threshold:
                flags.append(
                    self.make_flag(
                        "intention_tremor",
                        "Tremor when approaching the target",
                        FlagSeverity.MODERATE,
                        self.exceedance_confidence(analysis.terminal_tremor_cm, tremor_threshold, confidence),
                        f"The {side} fingertip oscillated as it approached the nose (intention tremor).",
                        [step_instance],
                        evidence=[f"terminal oscillation {analysis.terminal_tremor_cm:.2f} cm"],
                        side=side,  # type: ignore[arg-type]
                    )
                )
        slow_rate = self.threshold_float("slow_flip_rate_hz")
        cv_threshold = self.threshold_float("flip_interval_cv_max")
        minimum_cycles = self.threshold_float("min_flip_cycles")
        for side, (analysis, step_instance) in flip_results.items():
            confidence = self.data_confidence([step_instance])
            metrics.append(self.summary_reading("flip_rate", analysis.rate_hz, "Hz", f"{side.capitalize()} hand flipping rate.", confidence, step_instance, side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("flip_rhythm_variability", analysis.interval_cv, "CV", f"{side.capitalize()} hand flipping rhythm variability.", confidence, step_instance, side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("flip_amplitude", analysis.amplitude, "sine", f"{side.capitalize()} hand rotation range.", confidence, step_instance, side=side))  # type: ignore[arg-type]
            if analysis.cycles < minimum_cycles:
                limitations.append(f"Only {analysis.cycles:.1f} flipping cycles were detected for the {side} hand.")
                continue
            reasons: list[str] = []
            if analysis.rate_hz < slow_rate:
                reasons.append(f"slow rate {analysis.rate_hz:.2f} Hz")
            if np.isfinite(analysis.interval_cv) and analysis.interval_cv > cv_threshold:
                reasons.append(f"irregular rhythm (CV {analysis.interval_cv:.2f})")
            if reasons:
                flags.append(
                    self.make_flag(
                        "dysdiadochokinesia",
                        "Slow or irregular rapid hand flipping",
                        FlagSeverity.MODERATE,
                        confidence * min(1.0, 0.5 + 0.2 * len(reasons)),
                        f"Rapid palm-up, palm-down movements of the {side} hand were {' and '.join(reasons)}.",
                        [step_instance],
                        evidence=reasons,
                        side=side,  # type: ignore[arg-type]
                    )
                )
        if "right" in flip_results and "left" in flip_results:
            right_rate = flip_results["right"][0].rate_hz
            left_rate = flip_results["left"][0].rate_hz
            rate_asymmetry = asymmetry_ratio(right_rate, left_rate)
            confidence = min(self.data_confidence([flip_results["right"][1]]), self.data_confidence([flip_results["left"][1]]))
            metrics.append(self.summary_reading("flip_rate_asymmetry", rate_asymmetry, "ratio", "Relative left-right difference in flipping rate.", confidence, ",".join(self.steps_with_data)))
            asymmetry_threshold = self.threshold_float("asymmetry_ratio_flag")
            if rate_asymmetry >= asymmetry_threshold:
                slower = "right" if right_rate < left_rate else "left"
                flags.append(
                    self.make_flag(
                        "coordination_asymmetry",
                        "Left-right difference in rapid alternating movements",
                        FlagSeverity.LOW,
                        self.exceedance_confidence(rate_asymmetry, asymmetry_threshold, confidence),
                        f"The {slower} hand flipped more slowly than the other hand.",
                        [flip_results["right"][1], flip_results["left"][1]],
                        evidence=[f"flip rate asymmetry {rate_asymmetry:.2f}"],
                        side=slower,  # type: ignore[arg-type]
                    )
                )
        description = (
            "Coordination findings: " + "; ".join(flag.title for flag in flags) + "."
            if flags
            else "Finger-to-nose pointing and rapid hand flipping were within screening reference limits."
        )
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=self.data_confidence(self.steps_with_data),
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._nose_series.clear()
        self._flip_series.clear()
