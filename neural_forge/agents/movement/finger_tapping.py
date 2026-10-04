"""Finger tapping agent: tap count, rate, amplitude, decrement and hesitations, left versus right."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import asymmetry_ratio, euclidean_distance
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.signal_processing import find_peaks_irregular, linear_trend


@dataclass
class TappingAnalysis:
    """Tapping metrics for one hand."""

    side: str
    tap_count: int
    rate_hz: float
    mean_amplitude: float
    amplitude_decrement: float
    interval_increase: float
    amplitude_slope: float
    hesitations: int
    interval_cv: float


class FingerTappingAgent(AreaAgent):
    """Analyses the thumb-index distance signal with real timestamps."""

    name = "finger_tapping"
    area = "movement"
    required_trackers = frozenset({TrackerName.HANDS})
    supported_events = frozenset({"taps"})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._series: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def _side_for(self, step_instance: str) -> str | None:
        mapping = {str(key): str(value) for key, value in self.threshold("step_sides").items()}
        return mapping.get(base_step_id(step_instance))

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record the normalised thumb-index distance of the hand being tested."""
        hands = frame_context.hands
        step_instance = frame_context.step_instance or ""
        side = self._side_for(step_instance)
        if hands is None or side is None:
            return []
        hand = hands.get(side)  # type: ignore[arg-type]
        if hand is None:
            return []
        palm_length = hand.palm_length()
        if palm_length < 5.0:
            return []
        aperture = euclidean_distance(hand.point(4), hand.point(8)) / palm_length
        self._series[step_instance].append((frame_context.timestamp_s, aperture))
        return [
            self.make_reading(frame_context, "thumb_index_aperture", aperture * self.threshold_float("palm_length_cm"), "cm", f"{side.capitalize()} thumb-index distance.", side=side)  # type: ignore[arg-type]
        ]

    def _tap_peaks(self, samples: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(samples) < 5:
            return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
        timestamps = np.array([sample[0] for sample in samples])
        values = np.array([sample[1] for sample in samples])
        signal_range = float(np.percentile(values, 95) - np.percentile(values, 5))
        prominence = max(signal_range * self.threshold_float("peak_prominence_fraction"), self.threshold_float("min_amplitude_palm_fraction") * 0.5)
        peaks = find_peaks_irregular(timestamps, values, self.threshold_float("min_tap_interval_seconds"), prominence)
        return timestamps, values, peaks

    def count_events(self, step_instance: str, event: str) -> int:
        """Number of taps detected so far."""
        if event != "taps":
            return 0
        _, _, peaks = self._tap_peaks(self._series.get(step_instance, []))
        return int(len(peaks))

    def _analyse(self, side: str, samples: list[tuple[float, float]]) -> TappingAnalysis | None:
        timestamps, values, peaks = self._tap_peaks(samples)
        if len(peaks) < 3:
            return None
        palm_cm = self.threshold_float("palm_length_cm")
        amplitudes: list[float] = []
        previous_index = 0
        for peak_index in peaks:
            trough = float(values[previous_index : peak_index + 1].min())
            amplitudes.append((float(values[peak_index]) - trough) * palm_cm)
            previous_index = int(peak_index)
        amplitude_array = np.array(amplitudes)
        tap_times = timestamps[peaks]
        intervals = np.diff(tap_times)
        duration = float(tap_times[-1] - tap_times[0])
        rate = (len(peaks) - 1) / duration if duration > 0 else float("nan")
        third = max(1, len(amplitude_array) // 3)
        first_amplitude = float(amplitude_array[:third].mean())
        last_amplitude = float(amplitude_array[-third:].mean())
        decrement = 1.0 - last_amplitude / first_amplitude if first_amplitude > 0 else 0.0
        interval_third = max(1, len(intervals) // 3)
        first_interval = float(intervals[:interval_third].mean())
        last_interval = float(intervals[-interval_third:].mean())
        interval_increase = last_interval / first_interval - 1.0 if first_interval > 0 else 0.0
        median_interval = float(np.median(intervals))
        hesitations = int(np.sum(intervals > self.threshold_float("hesitation_interval_factor") * median_interval))
        slope = linear_trend(np.arange(len(amplitude_array), dtype=float), amplitude_array).slope
        interval_cv = float(np.std(intervals) / np.mean(intervals)) if len(intervals) > 1 and np.mean(intervals) > 0 else 0.0
        return TappingAnalysis(
            side=side,
            tap_count=int(len(peaks)),
            rate_hz=rate,
            mean_amplitude=float(amplitude_array.mean()),
            amplitude_decrement=decrement,
            interval_increase=interval_increase,
            amplitude_slope=slope,
            hesitations=hesitations,
            interval_cv=interval_cv,
        )

    def finalize(self) -> AgentResult:
        """Evaluate bradykinesia features and left-right asymmetry."""
        if not self.steps_with_data:
            return self.insufficient_result("Finger tapping could not be tracked.")
        analyses: dict[str, TappingAnalysis] = {}
        limitations: list[str] = []
        side_steps: dict[str, list[str]] = defaultdict(list)
        combined: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for step_instance, samples in self._series.items():
            side = self._side_for(step_instance)
            if side is None:
                continue
            side_steps[side].append(step_instance)
            if len(samples) > len(combined[side]):
                combined[side] = samples
        for side, samples in combined.items():
            analysis = self._analyse(side, samples)
            if analysis is None:
                limitations.append(f"Too few taps were detected for the {side} hand.")
                continue
            analyses[side] = analysis
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        slow_rate = self.threshold_float("slow_tap_rate_hz")
        decrement_threshold = self.threshold_float("amplitude_decrement_fraction")
        interval_threshold = self.threshold_float("interval_increase_fraction")
        hesitation_threshold = self.threshold_int("hesitation_count_flag")
        for side, analysis in analyses.items():
            confidence = self.data_confidence(side_steps[side])
            steps = ",".join(side_steps[side])
            metrics.extend(
                [
                    self.summary_reading("tap_count", float(analysis.tap_count), "taps", f"{side.capitalize()} hand tap count.", confidence, steps, side=side),  # type: ignore[arg-type]
                    self.summary_reading("tap_rate", analysis.rate_hz, "Hz", f"{side.capitalize()} hand tapping rate.", confidence, steps, side=side),  # type: ignore[arg-type]
                    self.summary_reading("tap_amplitude", analysis.mean_amplitude, "cm", f"{side.capitalize()} hand mean tap amplitude.", confidence, steps, side=side),  # type: ignore[arg-type]
                    self.summary_reading("amplitude_decrement", analysis.amplitude_decrement, "fraction", f"{side.capitalize()} hand amplitude reduction from first to last third.", confidence, steps, side=side),  # type: ignore[arg-type]
                    self.summary_reading("interval_increase", analysis.interval_increase, "fraction", f"{side.capitalize()} hand slowing from first to last third.", confidence, steps, side=side),  # type: ignore[arg-type]
                    self.summary_reading("hesitations", float(analysis.hesitations), "count", f"{side.capitalize()} hand pauses longer than usual.", confidence, steps, side=side),  # type: ignore[arg-type]
                    self.summary_reading("interval_variability", analysis.interval_cv, "CV", f"{side.capitalize()} hand rhythm variability.", confidence, steps, side=side),  # type: ignore[arg-type]
                ]
            )
            reasons: list[str] = []
            if np.isfinite(analysis.rate_hz) and analysis.rate_hz < slow_rate:
                reasons.append(f"slow rate {analysis.rate_hz:.1f} Hz")
            if analysis.amplitude_decrement >= decrement_threshold:
                reasons.append(f"amplitude decrement {analysis.amplitude_decrement:.0%}")
            if analysis.interval_increase >= interval_threshold:
                reasons.append(f"progressive slowing {analysis.interval_increase:.0%}")
            if analysis.hesitations >= hesitation_threshold:
                reasons.append(f"{analysis.hesitations} hesitations")
            if reasons:
                flags.append(
                    self.make_flag(
                        "finger_tapping_bradykinesia",
                        "Slowing or fading of finger tapping",
                        FlagSeverity.MODERATE,
                        confidence * min(1.0, 0.45 + 0.15 * len(reasons)),
                        f"{side.capitalize()} hand tapping showed features of bradykinesia: {', '.join(reasons)}.",
                        side_steps[side],
                        evidence=reasons,
                        side=side,  # type: ignore[arg-type]
                    )
                )
        if "right" in analyses and "left" in analyses:
            right = analyses["right"]
            left = analyses["left"]
            rate_asymmetry = asymmetry_ratio(right.rate_hz, left.rate_hz)
            amplitude_asymmetry = asymmetry_ratio(right.mean_amplitude, left.mean_amplitude)
            confidence = min(self.data_confidence(side_steps["right"]), self.data_confidence(side_steps["left"]))
            metrics.append(self.summary_reading("tap_rate_asymmetry", rate_asymmetry, "ratio", "Relative left-right difference in tapping rate.", confidence, ",".join(self.steps_with_data)))
            metrics.append(self.summary_reading("tap_amplitude_asymmetry", amplitude_asymmetry, "ratio", "Relative left-right difference in tap amplitude.", confidence, ",".join(self.steps_with_data)))
            asymmetry_threshold = self.threshold_float("asymmetry_ratio_flag")
            worst = max(rate_asymmetry, amplitude_asymmetry)
            if worst >= asymmetry_threshold:
                weaker = "right" if (right.rate_hz * right.mean_amplitude) < (left.rate_hz * left.mean_amplitude) else "left"
                flags.append(
                    self.make_flag(
                        "finger_tapping_asymmetry",
                        "Left-right difference in finger tapping",
                        FlagSeverity.LOW,
                        self.exceedance_confidence(worst, asymmetry_threshold, confidence),
                        f"The {weaker} hand tapped more slowly or with smaller movements than the other hand.",
                        self.steps_with_data,
                        evidence=[f"rate asymmetry {rate_asymmetry:.2f}", f"amplitude asymmetry {amplitude_asymmetry:.2f}"],
                        side=weaker,  # type: ignore[arg-type]
                    )
                )
        description = "; ".join(
            f"{side} hand: {analysis.tap_count} taps at {analysis.rate_hz:.1f} Hz, amplitude decrement {analysis.amplitude_decrement:.0%}"
            for side, analysis in analyses.items()
        ) or "Taps could not be counted reliably."
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=self.data_confidence(self.steps_with_data),
            value=float(np.mean([analysis.rate_hz for analysis in analyses.values()])) if analyses else None,
            unit="Hz",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._series.clear()
