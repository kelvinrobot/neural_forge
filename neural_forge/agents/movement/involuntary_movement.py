"""Involuntary movement agent: chorea-like, tic-like, dystonic posture and seizure-like movement screening."""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import euclidean_distance, line_tilt_degrees
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.signal_processing import (
    butter_bandpass,
    butter_highpass,
    dominant_peak,
    estimate_sampling_rate,
    resample_uniform,
    root_mean_square,
    spectral_flatness,
    welch_spectrum,
)


@dataclass
class BodySample:
    """Shoulder-normalised keypoint positions and posture angles of one frame."""

    timestamp: float
    positions: np.ndarray
    head_tilt: float
    shoulder_elevation: float


@dataclass
class SeizureEvent:
    """A window with rhythmic high-amplitude movement of several limbs."""

    timestamp: float
    step_instance: str
    limbs: list[str]
    frequency_hz: float


class InvoluntaryMovementAgent(AreaAgent):
    """Observes movements the patient is not asked to make, mainly during quiet sitting."""

    name = "involuntary_movement"
    area = "movement"
    required_trackers = frozenset({TrackerName.POSE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._keypoint_names: list[str] = [str(name) for name in self.threshold_list("observed_keypoints")]
        self._samples: dict[str, list[BodySample]] = defaultdict(list)
        self._seizure_events: list[SeizureEvent] = []
        self._last_seizure_check: dict[str, float] = {}

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record normalised keypoints and run a rolling seizure-like movement check."""
        pose = frame_context.pose
        if pose is None:
            return []
        minimum_score = self.threshold_float("min_keypoint_score")
        if not pose.all_visible(("left_shoulder", "right_shoulder"), minimum_score):
            return []
        left_shoulder = pose.keypoint("left_shoulder")
        right_shoulder = pose.keypoint("right_shoulder")
        shoulder_width = euclidean_distance(left_shoulder, right_shoulder)
        if shoulder_width < 5.0:
            return []
        midpoint = (left_shoulder + right_shoulder) / 2.0
        positions = np.full((len(self._keypoint_names), 2), np.nan)
        for index, name in enumerate(self._keypoint_names):
            if pose.is_visible(name, minimum_score):
                positions[index] = (pose.keypoint(name) - midpoint) / shoulder_width
        shoulder_tilt = line_tilt_degrees(right_shoulder, left_shoulder)
        head_tilt = float("nan")
        if pose.all_visible(("left_ear", "right_ear"), minimum_score):
            head_tilt = line_tilt_degrees(pose.keypoint("right_ear"), pose.keypoint("left_ear")) - shoulder_tilt
        elif pose.all_visible(("left_eye", "right_eye"), minimum_score):
            head_tilt = line_tilt_degrees(pose.keypoint("right_eye"), pose.keypoint("left_eye")) - shoulder_tilt
        shoulder_elevation = float(left_shoulder[1] - right_shoulder[1]) / shoulder_width
        step_instance = frame_context.step_instance or ""
        self._samples[step_instance].append(BodySample(frame_context.timestamp_s, positions, head_tilt, shoulder_elevation))
        readings = [
            self.make_reading(frame_context, "head_tilt", head_tilt, "degrees", "Head tilt relative to the shoulder line."),
        ]
        alert = self._rolling_seizure_check(step_instance, frame_context.timestamp_s)
        if alert is not None:
            readings.append(
                self.make_reading(
                    frame_context,
                    "seizure_like_movement_alert",
                    1.0,
                    "flag",
                    f"Rhythmic jerking of several limbs at about {alert.frequency_hz:.1f} Hz was detected.",
                    details={"limbs": alert.limbs},
                )
            )
        return readings

    def _keypoint_series(self, samples: list[BodySample], keypoint_index: int) -> tuple[np.ndarray, np.ndarray]:
        timestamps = np.array([sample.timestamp for sample in samples])
        positions = np.array([sample.positions[keypoint_index] for sample in samples])
        valid = np.isfinite(positions).all(axis=1)
        return timestamps[valid], positions[valid]

    def _rhythmic_limbs(self, samples: list[BodySample]) -> tuple[list[str], float]:
        low, high = self.threshold_range("seizure_band_hz")
        regularity_minimum = self.threshold_float("seizure_regularity_min")
        amplitude_minimum = self.threshold_float("seizure_amplitude_shoulder_fraction")
        limbs: list[str] = []
        frequencies: list[float] = []
        for name in [str(limb) for limb in self.threshold_list("limb_keypoints")]:
            if name not in self._keypoint_names:
                continue
            timestamps, positions = self._keypoint_series(samples, self._keypoint_names.index(name))
            if len(timestamps) < 12 or estimate_sampling_rate(timestamps) < 2.5 * low:
                continue
            uniform = resample_uniform(timestamps, positions)
            filtered = np.stack([butter_bandpass(uniform.values[:, axis], uniform.sampling_rate, low, high) for axis in range(2)], axis=1)
            magnitude = filtered[:, 1] if np.std(filtered[:, 1]) >= np.std(filtered[:, 0]) else filtered[:, 0]
            frequencies_axis, power = welch_spectrum(magnitude, uniform.sampling_rate, segment_seconds=2.0)
            peak = dominant_peak(frequencies_axis, power, low, min(high, uniform.sampling_rate / 2.0))
            amplitude = root_mean_square(magnitude) * 2.0 * math.sqrt(2.0)
            if peak is not None and peak.regularity >= regularity_minimum and amplitude >= amplitude_minimum:
                limbs.append(name)
                frequencies.append(peak.frequency_hz)
        return limbs, float(np.median(frequencies)) if frequencies else 0.0

    def _rolling_seizure_check(self, step_instance: str, timestamp: float) -> SeizureEvent | None:
        excluded = {str(step) for step in self.threshold_list("seizure_excluded_steps")}
        if base_step_id(step_instance) in excluded:
            return None
        interval = self.threshold_float("seizure_check_interval_seconds")
        if timestamp - self._last_seizure_check.get(step_instance, -math.inf) < interval:
            return None
        self._last_seizure_check[step_instance] = timestamp
        window = self.threshold_float("seizure_window_seconds")
        recent = [sample for sample in self._samples[step_instance] if sample.timestamp >= timestamp - window]
        if len(recent) < 12 or recent[-1].timestamp - recent[0].timestamp < 0.75 * window:
            return None
        limbs, frequency = self._rhythmic_limbs(recent)
        if len(limbs) >= self.threshold_int("seizure_min_limbs"):
            event = SeizureEvent(timestamp=timestamp, step_instance=step_instance, limbs=limbs, frequency_hz=frequency)
            self._seizure_events.append(event)
            return event
        return None

    def _movement_character(self, samples: list[BodySample]) -> tuple[float, float, float]:
        highpass = self.threshold_float("highpass_hz")
        energies: list[float] = []
        flatness_values: list[float] = []
        regularities: list[float] = []
        for index in range(len(self._keypoint_names)):
            timestamps, positions = self._keypoint_series(samples, index)
            if len(timestamps) < 16:
                continue
            uniform = resample_uniform(timestamps, positions)
            filtered = np.stack([butter_highpass(uniform.values[:, axis], uniform.sampling_rate, highpass) for axis in range(2)], axis=1)
            speed = np.linalg.norm(np.gradient(filtered, 1.0 / uniform.sampling_rate, axis=0), axis=1)
            energies.append(float(np.mean(speed)))
            combined = np.linalg.norm(filtered, axis=1)
            frequencies, power = welch_spectrum(combined, uniform.sampling_rate)
            band = (frequencies >= 0.5) & (frequencies <= min(8.0, uniform.sampling_rate / 2.0))
            if band.sum() >= 3:
                flatness_values.append(spectral_flatness(power[band]))
                peak = dominant_peak(frequencies, power, 0.5, min(8.0, uniform.sampling_rate / 2.0))
                if peak is not None:
                    regularities.append(peak.regularity)
        if not energies:
            return float("nan"), float("nan"), float("nan")
        return (
            float(np.median(energies)),
            float(np.median(flatness_values)) if flatness_values else float("nan"),
            float(np.median(regularities)) if regularities else float("nan"),
        )

    def _tic_bursts(self) -> tuple[int, float]:
        speed_threshold = self.threshold_float("tic_speed_threshold")
        maximum_duration = self.threshold_float("tic_max_duration_seconds")
        quiet_steps = {str(step) for step in self.threshold_list("quiet_steps")}
        head_keypoints = {str(name) for name in self.threshold_list("head_keypoints")}
        best_count = 0
        best_similarity = 0.0
        for index, keypoint_name in enumerate(self._keypoint_names):
            displacements: list[np.ndarray] = []
            for step_instance, samples in self._samples.items():
                if base_step_id(step_instance) not in quiet_steps and keypoint_name not in head_keypoints:
                    continue
                timestamps, positions = self._keypoint_series(samples, index)
                if len(timestamps) < 5:
                    continue
                velocities = np.gradient(positions, timestamps, axis=0)
                speeds = np.linalg.norm(velocities, axis=1)
                fast = speeds > speed_threshold
                start: int | None = None
                for sample_index, is_fast in enumerate(fast):
                    if is_fast and start is None:
                        start = sample_index
                    elif not is_fast and start is not None:
                        duration = timestamps[sample_index] - timestamps[start]
                        if duration <= maximum_duration:
                            displacements.append(positions[min(sample_index, len(positions) - 1)] - positions[max(start - 1, 0)])
                        start = None
            if len(displacements) < 2:
                continue
            vectors = np.array(displacements)
            norms = np.linalg.norm(vectors, axis=1)
            vectors = vectors[norms > 1e-6] / norms[norms > 1e-6, None]
            if len(vectors) < 2:
                continue
            similarity_matrix = np.abs(vectors @ vectors.T)
            upper = similarity_matrix[np.triu_indices(len(vectors), k=1)]
            similarity = float(np.mean(upper)) if len(upper) else 0.0
            if len(vectors) > best_count or (len(vectors) == best_count and similarity > best_similarity):
                best_count = len(vectors)
                best_similarity = similarity
        return best_count, best_similarity

    def finalize(self) -> AgentResult:
        """Characterise involuntary movements and raise the seizure-like emergency flag when detected."""
        if not self.steps_with_data:
            return self.insufficient_result("Body movements could not be tracked.")
        quiet_steps = {str(step) for step in self.threshold_list("quiet_steps")}
        quiet_samples: list[BodySample] = []
        for step_instance, samples in self._samples.items():
            if base_step_id(step_instance) in quiet_steps:
                quiet_samples.extend(samples)
        quiet_confidence = self.data_confidence(list(quiet_steps))
        all_confidence = self.data_confidence(self.steps_with_data)
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        energy, flatness, regularity = self._movement_character(quiet_samples)
        quiet_label = ",".join(sorted(quiet_steps))
        metrics.append(self.summary_reading("quiet_movement_energy", energy, "shoulder widths/s", "Median involuntary movement speed while sitting quietly.", quiet_confidence, quiet_label))
        metrics.append(self.summary_reading("movement_spectral_flatness", flatness, "fraction", "Irregularity of movement (1 is noise-like, 0 is rhythmic).", quiet_confidence, quiet_label))
        metrics.append(self.summary_reading("movement_regularity", regularity, "fraction", "Rhythmicity of movement while sitting quietly.", quiet_confidence, quiet_label))
        if quiet_samples and quiet_samples[-1].timestamp - quiet_samples[0].timestamp < self.threshold_float("min_duration_seconds"):
            limitations.append("Quiet sitting observation was too short for a reliable involuntary movement assessment.")
        energy_threshold = self.threshold_float("chorea_energy_min")
        if (
            np.isfinite(energy)
            and energy >= energy_threshold
            and np.isfinite(flatness)
            and flatness >= self.threshold_float("chorea_flatness_min")
            and (not np.isfinite(regularity) or regularity <= self.threshold_float("chorea_regularity_max"))
        ):
            flags.append(
                self.make_flag(
                    "chorea_like_movements",
                    "Irregular involuntary movements",
                    FlagSeverity.MODERATE,
                    self.exceedance_confidence(energy, energy_threshold, quiet_confidence),
                    "Frequent irregular, non-rhythmic movements were seen while sitting still, a pattern resembling chorea.",
                    sorted(quiet_steps),
                    evidence=[f"movement energy {energy:.3f}", f"spectral flatness {flatness:.2f}"],
                )
            )
        tic_count, tic_similarity = self._tic_bursts()
        metrics.append(self.summary_reading("brief_movement_bursts", float(tic_count), "count", "Brief rapid movement bursts of one body part.", all_confidence, ",".join(self.steps_with_data)))
        metrics.append(self.summary_reading("burst_similarity", tic_similarity, "cosine", "How similar the brief bursts were to each other.", all_confidence, ",".join(self.steps_with_data)))
        if tic_count >= self.threshold_int("tic_min_count") and tic_similarity >= self.threshold_float("tic_similarity_min"):
            flags.append(
                self.make_flag(
                    "tic_like_movements",
                    "Brief repetitive stereotyped movements",
                    FlagSeverity.LOW,
                    all_confidence * tic_similarity,
                    "Brief, repeated movements with a similar direction each time were detected, resembling tics.",
                    self.steps_with_data,
                    evidence=[f"{tic_count} bursts", f"similarity {tic_similarity:.2f}"],
                )
            )
        tilts = np.array([sample.head_tilt for samples in self._samples.values() for sample in samples])
        elevations = np.array([sample.shoulder_elevation for samples in self._samples.values() for sample in samples])
        finite_tilts = tilts[np.isfinite(tilts)]
        tilt_threshold = self.threshold_float("dystonia_head_tilt_degrees")
        elevation_threshold = self.threshold_float("dystonia_shoulder_elevation_fraction")
        tilt_fraction = float((np.abs(finite_tilts) > tilt_threshold).mean()) if len(finite_tilts) else 0.0
        elevation_fraction = float((np.abs(elevations) > elevation_threshold).mean()) if len(elevations) else 0.0
        metrics.append(self.summary_reading("median_head_tilt", float(np.median(finite_tilts)) if len(finite_tilts) else float("nan"), "degrees", "Median head tilt relative to the shoulders.", all_confidence, ",".join(self.steps_with_data)))
        metrics.append(self.summary_reading("sustained_head_tilt_fraction", tilt_fraction, "fraction", "Fraction of time with marked head tilt.", all_confidence, ",".join(self.steps_with_data)))
        metrics.append(self.summary_reading("sustained_shoulder_elevation_fraction", elevation_fraction, "fraction", "Fraction of time with one shoulder held markedly higher.", all_confidence, ",".join(self.steps_with_data)))
        minimum_fraction = self.threshold_float("dystonia_min_fraction")
        if tilt_fraction >= minimum_fraction or elevation_fraction >= minimum_fraction:
            flags.append(
                self.make_flag(
                    "sustained_abnormal_posture",
                    "Sustained abnormal head or shoulder posture",
                    FlagSeverity.MODERATE,
                    all_confidence * max(tilt_fraction, elevation_fraction),
                    "The head or a shoulder was held in a tilted or raised position most of the time, which can occur with dystonia.",
                    self.steps_with_data,
                    evidence=[f"head tilt fraction {tilt_fraction:.2f}", f"shoulder elevation fraction {elevation_fraction:.2f}"],
                )
            )
        metrics.append(self.summary_reading("seizure_like_windows", float(len(self._seizure_events)), "count", "Windows with rhythmic jerking of several limbs.", all_confidence, ",".join(self.steps_with_data)))
        if self._seizure_events:
            limbs = sorted({limb for event in self._seizure_events for limb in event.limbs})
            flags.append(
                self.make_flag(
                    "seizure_like_movement",
                    "Rhythmic whole-limb jerking (seizure-like)",
                    FlagSeverity.HIGH,
                    min(1.0, all_confidence * (0.6 + 0.1 * len(self._seizure_events))),
                    "Rhythmic jerking of several limbs was detected outside voluntary tasks. If this was not intentional, seek emergency help.",
                    sorted({event.step_instance for event in self._seizure_events}),
                    evidence=[f"{len(self._seizure_events)} window(s)", f"limbs: {', '.join(limbs)}"],
                )
            )
        description = (
            "Involuntary movement findings: " + "; ".join(flag.title for flag in flags) + "."
            if flags
            else "No marked involuntary movements were detected."
        )
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=quiet_confidence if quiet_confidence > 0 else all_confidence,
            value=energy,
            unit="shoulder widths/s",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._samples.clear()
        self._seizure_events.clear()
        self._last_seizure_check.clear()
