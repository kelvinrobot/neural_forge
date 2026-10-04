"""Tremor agent: rest, postural and kinetic tremor frequency, amplitude, regularity and pattern."""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.signal_processing import (
    butter_bandpass,
    dominant_peak,
    estimate_sampling_rate,
    resample_uniform,
    root_mean_square,
    welch_spectrum,
)


@dataclass
class TremorAnalysis:
    """Spectral tremor analysis of one hand in one step instance."""

    condition: str
    side: str
    step_instance: str
    frequency_hz: float
    amplitude_cm: float
    regularity: float
    sampling_rate: float
    present: bool


class TremorAgent(AreaAgent):
    """Band-pass filters fingertip trajectories and estimates tremor with a Welch power spectrum."""

    name = "tremor"
    area = "movement"
    required_trackers = frozenset({TrackerName.HANDS})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._series: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)

    def _condition(self, step_id: str) -> str | None:
        if step_id in [str(step) for step in self.threshold_list("rest_steps")]:
            return "rest"
        if step_id in [str(step) for step in self.threshold_list("postural_steps")]:
            return "postural"
        if step_id in self.threshold("kinetic_steps"):
            return "kinetic"
        return None

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record fingertip positions in centimetres for each visible hand."""
        hands = frame_context.hands
        if hands is None or hands.count == 0:
            return []
        step_instance = frame_context.step_instance or ""
        step_id = base_step_id(step_instance)
        kinetic_sides: dict[str, str] = {str(key): str(value) for key, value in self.threshold("kinetic_steps").items()}
        palm_length_cm = self.threshold_float("palm_length_cm")
        live_window = self.threshold_float("live_window_seconds")
        readings: list[Reading] = []
        for hand in hands.hands:
            if step_id in kinetic_sides and hand.side != kinetic_sides[step_id]:
                continue
            palm_pixels = hand.palm_length()
            if palm_pixels < 5.0:
                continue
            centimetres_per_pixel = palm_length_cm / palm_pixels
            tip = hand.point(8)
            key = (step_instance, hand.side)
            self._series[key].append((frame_context.timestamp_s, float(tip[0] * centimetres_per_pixel), float(tip[1] * centimetres_per_pixel)))
            recent = [item for item in self._series[key] if item[0] >= frame_context.timestamp_s - live_window]
            if len(recent) >= 3:
                median_x = float(np.median([item[1] for item in recent]))
                median_y = float(np.median([item[2] for item in recent]))
                displacement = math.hypot(recent[-1][1] - median_x, recent[-1][2] - median_y)
            else:
                displacement = 0.0
            readings.append(
                self.make_reading(frame_context, "fingertip_displacement", displacement, "cm", f"{hand.side.capitalize()} index fingertip displacement from its recent average position.", side=hand.side)
            )
        return readings

    def _analyse(self, step_instance: str, side: str, samples: list[tuple[float, float, float]]) -> TremorAnalysis | None:
        condition = self._condition(base_step_id(step_instance))
        if condition is None or len(samples) < 16:
            return None
        timestamps = np.array([sample[0] for sample in samples])
        if timestamps[-1] - timestamps[0] < self.threshold_float("min_duration_seconds"):
            return None
        if estimate_sampling_rate(timestamps) < self.threshold_float("min_sampling_rate_hz"):
            return None
        positions = np.array([[sample[1], sample[2]] for sample in samples])
        uniform = resample_uniform(timestamps, positions)
        low, high = self.threshold_range("analysis_band_hz")
        filtered = np.stack(
            [butter_bandpass(uniform.values[:, axis], uniform.sampling_rate, low, high) for axis in range(2)], axis=1
        )
        centered = filtered - filtered.mean(axis=0)
        _, _, right_singular = np.linalg.svd(centered, full_matrices=False)
        principal = centered @ right_singular[0]
        frequencies, power = welch_spectrum(principal, uniform.sampling_rate)
        peak = dominant_peak(frequencies, power, low, min(high, uniform.sampling_rate / 2.0))
        if peak is None:
            return None
        amplitude_cm = root_mean_square(principal) * 2.0 * math.sqrt(2.0)
        present = amplitude_cm >= self.threshold_float("min_amplitude_cm") and peak.regularity >= self.threshold_float("min_regularity")
        return TremorAnalysis(
            condition=condition,
            side=side,
            step_instance=step_instance,
            frequency_hz=peak.frequency_hz,
            amplitude_cm=amplitude_cm,
            regularity=peak.regularity,
            sampling_rate=uniform.sampling_rate,
            present=present,
        )

    @staticmethod
    def _strongest(analyses: list[TremorAnalysis], condition: str) -> TremorAnalysis | None:
        candidates = [analysis for analysis in analyses if analysis.condition == condition]
        if not candidates:
            return None
        return max(candidates, key=lambda analysis: (analysis.present, analysis.amplitude_cm))

    def finalize(self) -> AgentResult:
        """Classify the tremor pattern using YAML frequency bands."""
        if not self.steps_with_data:
            return self.insufficient_result("Hands could not be tracked for the tremor assessment.")
        analyses: list[TremorAnalysis] = []
        limitations: list[str] = []
        for (step_instance, side), samples in self._series.items():
            analysis = self._analyse(step_instance, side, samples)
            if analysis is None:
                limitations.append(f"Tremor analysis was not possible for the {side} hand during {step_instance}.")
                continue
            analyses.append(analysis)
        metrics: list[Reading] = []
        band_high = self.threshold_range("analysis_band_hz")[1]
        for analysis in analyses:
            confidence = self.data_confidence([analysis.step_instance]) * min(1.0, analysis.sampling_rate / (2.0 * band_high))
            label = f"{analysis.condition} tremor ({analysis.side} hand, {analysis.step_instance})"
            metrics.append(self.summary_reading("tremor_frequency", analysis.frequency_hz, "Hz", f"Dominant frequency, {label}.", confidence, analysis.step_instance, side=analysis.side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("tremor_amplitude", analysis.amplitude_cm, "cm", f"Peak-to-peak amplitude, {label}.", confidence, analysis.step_instance, side=analysis.side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("tremor_regularity", analysis.regularity, "fraction", f"Spectral regularity, {label}.", confidence, analysis.step_instance, side=analysis.side))  # type: ignore[arg-type]
        rest = self._strongest(analyses, "rest")
        postural = self._strongest(analyses, "postural")
        kinetic = self._strongest(analyses, "kinetic")
        parkinsonian_low, parkinsonian_high = self.threshold_range("parkinsonian_band_hz")
        essential_low, essential_high = self.threshold_range("essential_band_hz")
        dominance_ratio = self.threshold_float("rest_dominance_ratio")
        flags: list[ScreeningFlag] = []
        rest_present = rest is not None and rest.present
        action_candidates = [analysis for analysis in (postural, kinetic) if analysis is not None and analysis.present]
        strongest_action = max(action_candidates, key=lambda analysis: analysis.amplitude_cm) if action_candidates else None
        pattern = "none"
        if rest_present and rest is not None and parkinsonian_low <= rest.frequency_hz <= parkinsonian_high and (
            strongest_action is None or rest.amplitude_cm >= dominance_ratio * strongest_action.amplitude_cm
        ):
            pattern = "parkinsonian_range"
            flags.append(
                self.make_flag(
                    "parkinsonian_range_tremor",
                    "Rest tremor in the Parkinsonian frequency range",
                    FlagSeverity.MODERATE,
                    self.data_confidence([rest.step_instance]) * rest.regularity * 1.3,
                    f"A regular tremor of about {rest.frequency_hz:.1f} Hz was seen with the hands at rest ({rest.side} hand), a pattern typical of Parkinsonian tremor.",
                    [rest.step_instance],
                    evidence=[f"rest {rest.frequency_hz:.1f} Hz, {rest.amplitude_cm:.2f} cm, regularity {rest.regularity:.2f}"],
                    side=rest.side,  # type: ignore[arg-type]
                )
            )
        elif strongest_action is not None and not rest_present and essential_low <= strongest_action.frequency_hz <= essential_high:
            pattern = "essential_range"
            flags.append(
                self.make_flag(
                    "essential_range_tremor",
                    "Action tremor in the essential-tremor frequency range",
                    FlagSeverity.MODERATE,
                    self.data_confidence([strongest_action.step_instance]) * strongest_action.regularity * 1.3,
                    f"A regular tremor of about {strongest_action.frequency_hz:.1f} Hz appeared with the arms held out or moving but not at rest, a pattern typical of essential tremor.",
                    [analysis.step_instance for analysis in action_candidates],
                    evidence=[f"{analysis.condition} {analysis.frequency_hz:.1f} Hz, {analysis.amplitude_cm:.2f} cm" for analysis in action_candidates],
                    side=strongest_action.side,  # type: ignore[arg-type]
                )
            )
        elif rest_present or strongest_action is not None:
            pattern = "unclassified"
            reference = rest if rest_present and rest is not None else strongest_action
            assert reference is not None
            flags.append(
                self.make_flag(
                    "tremor_unclassified",
                    "Tremor detected",
                    FlagSeverity.LOW,
                    self.data_confidence([reference.step_instance]) * reference.regularity,
                    f"A tremor of about {reference.frequency_hz:.1f} Hz was detected but did not fit a typical pattern.",
                    [reference.step_instance],
                    evidence=[f"{reference.condition} {reference.frequency_hz:.1f} Hz, {reference.amplitude_cm:.2f} cm"],
                    side=reference.side,  # type: ignore[arg-type]
                )
            )
        metrics.append(self.summary_reading("tremor_pattern_code", {"none": 0.0, "parkinsonian_range": 1.0, "essential_range": 2.0, "unclassified": 3.0}[pattern], "code", "Tremor pattern: 0 none, 1 Parkinsonian range, 2 essential range, 3 unclassified.", self.data_confidence(self.steps_with_data), ",".join(self.steps_with_data)))
        largest_amplitude = max((analysis.amplitude_cm for analysis in analyses if analysis.present), default=0.0)
        description = {
            "none": "No significant tremor was detected at rest, with arms out or during movement.",
            "parkinsonian_range": "A rest-dominant tremor in the Parkinsonian frequency range was detected.",
            "essential_range": "An action tremor in the essential-tremor frequency range was detected.",
            "unclassified": "A tremor was detected that did not fit a typical frequency pattern.",
        }[pattern]
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=self.data_confidence(self.steps_with_data),
            value=largest_amplitude,
            unit="cm",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._series.clear()
