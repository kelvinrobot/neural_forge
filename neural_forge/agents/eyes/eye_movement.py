"""Eye movement agent: iris position time series, nystagmus frequency analysis and gaze range."""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import euclidean_distance, eye_aspect_ratio, signed_distance_to_line, unit_vector
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import iris_geometry, landmark_index, landmark_indices
from neural_forge.services.signal_processing import (
    butter_bandpass,
    dominant_peak,
    estimate_sampling_rate,
    resample_uniform,
    root_mean_square,
    welch_spectrum,
)


@dataclass
class GazeSample:
    """Conjugate iris position in eye-width units (+x toward the patient's left, +y downward)."""

    timestamp: float
    horizontal: float
    vertical: float
    eye_width_mm: float


@dataclass
class NystagmusAnalysis:
    """Frequency analysis of one step."""

    present: bool
    frequency_hz: float
    amplitude_degrees: float
    regularity: float
    axis: str
    beat_direction: str
    sampling_rate: float
    mean_horizontal: float
    mean_vertical: float


class EyeMovementAgent(AreaAgent):
    """Analyses gaze holding in primary and eccentric gaze for nystagmus and limited range."""

    name = "eye_movement"
    area = "eyes"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._samples: dict[str, list[GazeSample]] = defaultdict(list)

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record the conjugate iris position relative to the eye corners."""
        face = frame_context.face
        if face is None:
            return []
        blink_threshold = self.threshold_float("blink_ear_threshold")
        horizontal_values: list[float] = []
        vertical_values: list[float] = []
        widths_mm: list[float] = []
        for side in ("right", "left"):
            if eye_aspect_ratio(face.points(landmark_indices(side, "ear_points"))) < blink_threshold:
                return []
            outer = face.point(landmark_index(side, "outer"))
            inner = face.point(landmark_index(side, "inner"))
            start, end = (outer, inner) if side == "right" else (inner, outer)
            width = euclidean_distance(start, end)
            center, radius = iris_geometry(face, side)
            if width < 1.0 or radius <= 0.5:
                return []
            axis_unit = unit_vector(end - start)
            eye_center = (start + end) / 2.0
            horizontal_values.append(float(np.dot(center - eye_center, axis_unit)) / width)
            vertical_values.append(signed_distance_to_line(center, start, end - start) / width)
            widths_mm.append(width * self.threshold_float("iris_diameter_mm") / (2.0 * radius))
        sample = GazeSample(
            timestamp=frame_context.timestamp_s,
            horizontal=float(np.mean(horizontal_values)),
            vertical=float(np.mean(vertical_values)),
            eye_width_mm=float(np.mean(widths_mm)),
        )
        self._samples[frame_context.step_instance or ""].append(sample)
        return [
            self.make_reading(frame_context, "horizontal_gaze_position", sample.horizontal, "eye widths", "Iris position along the eye (positive toward the patient's left)."),
            self.make_reading(frame_context, "vertical_gaze_position", sample.vertical, "eye widths", "Iris position across the eye (positive downward)."),
        ]

    def _analyse(self, samples: list[GazeSample]) -> NystagmusAnalysis | None:
        if len(samples) < 8:
            return None
        timestamps = np.array([sample.timestamp for sample in samples])
        if timestamps[-1] - timestamps[0] < self.threshold_float("min_duration_seconds"):
            return None
        measured_rate = estimate_sampling_rate(timestamps)
        if measured_rate < self.threshold_float("min_sampling_rate_hz"):
            return None
        positions = np.stack([[sample.horizontal, sample.vertical] for sample in samples])
        uniform = resample_uniform(timestamps, positions)
        low = self.threshold_float("band_low_hz")
        high = self.threshold_float("band_high_hz")
        eye_width_mm = float(np.median([sample.eye_width_mm for sample in samples]))
        radius = self.threshold_float("eyeball_radius_mm")
        best: tuple[float, str, float, float, np.ndarray] | None = None
        for axis_index, axis_name in ((0, "horizontal"), (1, "vertical")):
            filtered = butter_bandpass(uniform.values[:, axis_index], uniform.sampling_rate, low, high)
            frequencies, power = welch_spectrum(filtered, uniform.sampling_rate)
            peak = dominant_peak(frequencies, power, low, min(high, uniform.sampling_rate / 2.0))
            if peak is None:
                continue
            if best is None or peak.band_power > best[0]:
                best = (peak.band_power, axis_name, peak.frequency_hz, peak.regularity, filtered)
        if best is None:
            return None
        _, axis_name, frequency, regularity, filtered = best
        amplitude_mm = root_mean_square(filtered) * math.sqrt(2.0) * eye_width_mm
        amplitude_degrees = math.degrees(math.asin(min(1.0, amplitude_mm / radius)))
        velocity = np.gradient(filtered, 1.0 / uniform.sampling_rate)
        positive_peak = float(np.percentile(velocity, 95))
        negative_peak = float(abs(np.percentile(velocity, 5)))
        if axis_name == "horizontal":
            positive_label, negative_label = "toward the patient's left", "toward the patient's right"
        else:
            positive_label, negative_label = "downward", "upward"
        if positive_peak > 1.5 * negative_peak:
            beat_direction = positive_label
        elif negative_peak > 1.5 * positive_peak:
            beat_direction = negative_label
        else:
            beat_direction = "pendular"
        present = regularity >= self.threshold_float("min_regularity") and amplitude_degrees >= self.threshold_float("min_amplitude_degrees")
        return NystagmusAnalysis(
            present=present,
            frequency_hz=frequency,
            amplitude_degrees=amplitude_degrees,
            regularity=regularity,
            axis=axis_name,
            beat_direction=beat_direction,
            sampling_rate=uniform.sampling_rate,
            mean_horizontal=float(np.mean(positions[:, 0])),
            mean_vertical=float(np.mean(positions[:, 1])),
        )

    def finalize(self) -> AgentResult:
        """Run frequency analysis per gaze step and evaluate nystagmus and gaze range."""
        if not self.steps_with_data:
            return self.insufficient_result("Eye movements could not be tracked.")
        primary_steps = [str(step) for step in self.threshold_list("primary_steps")]
        gaze_steps: dict[str, str] = {str(key): str(value) for key, value in self.threshold("gaze_steps").items()}
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        analyses: dict[str, NystagmusAnalysis] = {}
        for step_instance, samples in self._samples.items():
            analysis = self._analyse(samples)
            if analysis is None:
                limitations.append(f"Eye movement frequency analysis was not possible for {step_instance} (too short or too few frames per second).")
                continue
            analyses[step_instance] = analysis
            confidence = self.data_confidence([step_instance]) * min(1.0, analysis.sampling_rate / (2.0 * self.threshold_float("band_high_hz")))
            metrics.append(self.summary_reading("oscillation_frequency", analysis.frequency_hz, "Hz", f"Dominant eye oscillation frequency during {step_instance}.", confidence, step_instance))
            metrics.append(self.summary_reading("oscillation_amplitude", analysis.amplitude_degrees, "degrees", f"Eye oscillation amplitude during {step_instance}.", confidence, step_instance))
            metrics.append(self.summary_reading("oscillation_regularity", analysis.regularity, "fraction", f"Spectral regularity of eye oscillation during {step_instance}.", confidence, step_instance))
        primary_present = any(analysis.present for instance, analysis in analyses.items() if base_step_id(instance) in primary_steps)
        positive_steps = [(instance, analysis) for instance, analysis in analyses.items() if analysis.present]
        if positive_steps:
            strongest_instance, strongest = max(positive_steps, key=lambda item: item[1].amplitude_degrees)
            gaze_only = not primary_present and all(base_step_id(instance) in gaze_steps for instance, _ in positive_steps)
            confidence = self.data_confidence([instance for instance, _ in positive_steps]) * strongest.regularity
            flags.append(
                self.make_flag(
                    "gaze_evoked_nystagmus" if gaze_only else "nystagmus",
                    "Gaze-evoked nystagmus" if gaze_only else "Rhythmic eye oscillation (nystagmus)",
                    FlagSeverity.MODERATE,
                    min(1.0, confidence * 1.2),
                    f"A rhythmic {strongest.axis} eye oscillation of about {strongest.frequency_hz:.1f} Hz, beating {strongest.beat_direction}, was detected"
                    + (" only when looking to the side." if gaze_only else "."),
                    [instance for instance, _ in positive_steps],
                    evidence=[
                        f"{instance}: {analysis.frequency_hz:.1f} Hz, {analysis.amplitude_degrees:.2f} deg, regularity {analysis.regularity:.2f}"
                        for instance, analysis in positive_steps
                    ],
                )
            )
        primary_analyses = [analysis for instance, analysis in analyses.items() if base_step_id(instance) in primary_steps]
        primary_horizontal = float(np.mean([analysis.mean_horizontal for analysis in primary_analyses])) if primary_analyses else 0.0
        primary_vertical = float(np.mean([analysis.mean_vertical for analysis in primary_analyses])) if primary_analyses else 0.0
        minimum_excursion = self.threshold_float("min_gaze_excursion_fraction")
        limited_directions: list[str] = []
        for step_id, direction in gaze_steps.items():
            step_samples = [sample for instance, samples in self._samples.items() if base_step_id(instance) == step_id for sample in samples]
            if len(step_samples) < 5:
                continue
            mean_horizontal = float(np.mean([sample.horizontal for sample in step_samples]))
            mean_vertical = float(np.mean([sample.vertical for sample in step_samples]))
            if direction == "left":
                excursion = mean_horizontal - primary_horizontal
            elif direction == "right":
                excursion = primary_horizontal - mean_horizontal
            elif direction == "up":
                excursion = primary_vertical - mean_vertical
            else:
                excursion = mean_vertical - primary_vertical
            metrics.append(self.summary_reading("gaze_excursion", excursion, "eye widths", f"Iris movement when looking {direction}.", self.data_confidence([step_id]), step_id))
            if excursion < minimum_excursion:
                limited_directions.append(direction)
        if limited_directions:
            flags.append(
                self.make_flag(
                    "limited_gaze_range",
                    "Limited eye movement range",
                    FlagSeverity.LOW,
                    self.data_confidence(list(gaze_steps.keys())) * 0.6,
                    f"The eyes moved less than expected when looking {', '.join(limited_directions)}.",
                    list(gaze_steps.keys()),
                    evidence=[f"limited: {', '.join(limited_directions)}"],
                )
            )
        confidence = self.data_confidence(self.steps_with_data)
        description = (
            "Rhythmic eye oscillation detected in: " + ", ".join(instance for instance, _ in positive_steps) + "."
            if positive_steps
            else "No rhythmic eye oscillation was detected during gaze holding."
        )
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=confidence,
            value=max((analysis.amplitude_degrees for _, analysis in positive_steps), default=0.0),
            unit="degrees",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._samples.clear()
