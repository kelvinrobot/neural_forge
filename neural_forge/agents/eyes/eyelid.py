"""Eyelid agent: margin reflex distances, ptosis and sustained-upgaze eyelid fatigue."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.mediapipe_face import FaceLandmarks
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import eye_aspect_ratio, unit_vector
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import iris_geometry, landmark_index, landmark_indices
from neural_forge.services.signal_processing import linear_trend


@dataclass
class LidSample:
    """Eyelid distances of one eye in one frame."""

    timestamp: float
    step_elapsed: float
    mrd1_mm: float
    mrd2_mm: float
    fissure_mm: float


def eyelid_distances(face: FaceLandmarks, side: str, iris_diameter_mm: float) -> tuple[float, float, float]:
    """Return (MRD1, MRD2, palpebral fissure) in millimetres for one eye."""
    center, radius = iris_geometry(face, side)
    if radius <= 0.5:
        return float("nan"), float("nan"), float("nan")
    millimetres_per_pixel = iris_diameter_mm / (2.0 * radius)
    if side == "right":
        axis = face.point(landmark_index(side, "inner")) - face.point(landmark_index(side, "outer"))
    else:
        axis = face.point(landmark_index(side, "outer")) - face.point(landmark_index(side, "inner"))
    axis_unit = unit_vector(axis)
    downward_normal = np.array([-axis_unit[1], axis_unit[0]])
    if downward_normal[1] < 0:
        downward_normal = -downward_normal
    upper = face.point(landmark_index(side, "upper"))
    lower = face.point(landmark_index(side, "lower"))
    mrd1 = float(np.dot(center - upper, downward_normal)) * millimetres_per_pixel
    mrd2 = float(np.dot(lower - center, downward_normal)) * millimetres_per_pixel
    return mrd1, mrd2, mrd1 + mrd2


class EyelidAgent(AreaAgent):
    """Measures upper and lower eyelid position relative to the pupil and its change over sustained upgaze."""

    name = "eyelid"
    area = "eyes"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._samples: dict[str, dict[str, list[LidSample]]] = defaultdict(lambda: {"right": [], "left": []})

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure MRD1, MRD2 and fissure height for both eyes, skipping blinks."""
        face = frame_context.face
        if face is None:
            return []
        blink_threshold = self.threshold_float("blink_ear_threshold")
        iris_diameter = self.threshold_float("iris_diameter_mm")
        step_instance = frame_context.step_instance or ""
        elapsed = self.step_elapsed(step_instance, frame_context.timestamp_s)
        readings: list[Reading] = []
        for side in ("right", "left"):
            ear = eye_aspect_ratio(face.points(landmark_indices(side, "ear_points")))
            if not np.isfinite(ear) or ear < blink_threshold:
                continue
            mrd1, mrd2, fissure = eyelid_distances(face, side, iris_diameter)
            if not np.isfinite(mrd1):
                continue
            self._samples[step_instance][side].append(LidSample(frame_context.timestamp_s, elapsed, mrd1, mrd2, fissure))
            readings.append(
                self.make_reading(frame_context, "mrd1", mrd1, "mm", f"{side.capitalize()} upper eyelid to pupil centre distance.", side=side)  # type: ignore[arg-type]
            )
        return readings

    def _samples_for(self, step_ids: list[str], side: str) -> list[LidSample]:
        wanted = set(step_ids)
        collected: list[LidSample] = []
        for step_instance, sides in self._samples.items():
            if base_step_id(step_instance) in wanted:
                collected.extend(sides[side])
        return collected

    def _fatigue(self, samples: list[LidSample]) -> tuple[float, float]:
        window = self.threshold_float("fatigue_window_seconds")
        minimum_samples = self.threshold_int("min_window_samples")
        if len(samples) < 2 * minimum_samples:
            return float("nan"), float("nan")
        elapsed = np.array([sample.step_elapsed for sample in samples])
        values = np.array([sample.mrd1_mm for sample in samples])
        end_time = float(elapsed.max())
        early = values[elapsed <= elapsed.min() + window]
        late = values[elapsed >= end_time - window]
        if len(early) < minimum_samples or len(late) < minimum_samples:
            return float("nan"), float("nan")
        droop = float(np.median(early) - np.median(late))
        slope_per_minute = linear_trend(elapsed, values).slope * 60.0
        return droop, slope_per_minute

    def finalize(self) -> AgentResult:
        """Evaluate ptosis at baseline and eyelid fatigue during sustained upgaze."""
        if not self.steps_with_data:
            return self.insufficient_result("Eyelid position could not be measured.")
        baseline_steps = [str(step) for step in self.threshold_list("baseline_steps")]
        fatigue_steps = [str(step) for step in self.threshold_list("fatigue_steps")]
        baseline_confidence = self.data_confidence(baseline_steps)
        fatigue_confidence = self.data_confidence(fatigue_steps)
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        mrd1: dict[str, float] = {}
        for side in ("right", "left"):
            samples = self._samples_for(baseline_steps, side)
            mrd1[side] = robust_median([sample.mrd1_mm for sample in samples])
            metrics.append(self.summary_reading("mrd1", mrd1[side], "mm", f"{side.capitalize()} margin reflex distance 1 at rest.", baseline_confidence, ",".join(baseline_steps), side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("mrd2", robust_median([sample.mrd2_mm for sample in samples]), "mm", f"{side.capitalize()} margin reflex distance 2 at rest.", baseline_confidence, ",".join(baseline_steps), side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("palpebral_fissure", robust_median([sample.fissure_mm for sample in samples]), "mm", f"{side.capitalize()} palpebral fissure height.", baseline_confidence, ",".join(baseline_steps), side=side))  # type: ignore[arg-type]
        ptosis_threshold = self.threshold_float("ptosis_mrd1_mm")
        asymmetry_threshold = self.threshold_float("ptosis_asymmetry_mm")
        asymmetry = abs(mrd1["right"] - mrd1["left"]) if np.isfinite(mrd1["right"]) and np.isfinite(mrd1["left"]) else float("nan")
        metrics.append(self.summary_reading("mrd1_asymmetry", asymmetry, "mm", "Difference in upper eyelid height between eyes.", baseline_confidence, ",".join(baseline_steps)))
        low_sides = [side for side in ("right", "left") if np.isfinite(mrd1[side]) and mrd1[side] < ptosis_threshold]
        if low_sides or (np.isfinite(asymmetry) and asymmetry >= asymmetry_threshold):
            if low_sides:
                side_label = low_sides[0] if len(low_sides) == 1 else "both"
                confidence = max(self.shortfall_confidence(mrd1[side], ptosis_threshold, baseline_confidence) for side in low_sides)
            else:
                side_label = "right" if mrd1["right"] < mrd1["left"] else "left"
                confidence = self.exceedance_confidence(asymmetry, asymmetry_threshold, baseline_confidence)
            flags.append(
                self.make_flag(
                    "ptosis",
                    "Drooping upper eyelid",
                    FlagSeverity.MODERATE,
                    confidence,
                    f"The upper eyelid sat low over the pupil ({side_label} side).",
                    baseline_steps,
                    evidence=[f"right MRD1 {mrd1['right']:.2f} mm", f"left MRD1 {mrd1['left']:.2f} mm"],
                    side=side_label,  # type: ignore[arg-type]
                )
            )
        droop_threshold = self.threshold_float("fatigue_droop_mm")
        fatigued_sides: list[tuple[str, float]] = []
        for side in ("right", "left"):
            droop, slope = self._fatigue(self._samples_for(fatigue_steps, side))
            metrics.append(self.summary_reading("upgaze_lid_droop", droop, "mm", f"{side.capitalize()} eyelid drop from start to end of sustained upgaze.", fatigue_confidence, ",".join(fatigue_steps), side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("upgaze_lid_slope", slope, "mm/min", f"{side.capitalize()} eyelid height trend during sustained upgaze.", fatigue_confidence, ",".join(fatigue_steps), side=side))  # type: ignore[arg-type]
            if np.isfinite(droop) and droop >= droop_threshold:
                fatigued_sides.append((side, droop))
        if fatigued_sides:
            worst_side, worst_droop = max(fatigued_sides, key=lambda item: item[1])
            flags.append(
                self.make_flag(
                    "upgaze_eyelid_fatigue",
                    "Eyelid fatigue during sustained upward gaze",
                    FlagSeverity.MODERATE,
                    self.exceedance_confidence(worst_droop, droop_threshold, fatigue_confidence),
                    "The upper eyelid drooped progressively while looking up, a fatigable pattern that can occur in myasthenia gravis.",
                    fatigue_steps,
                    evidence=[f"{side} droop {droop:.2f} mm" for side, droop in fatigued_sides],
                    side=worst_side if len(fatigued_sides) == 1 else "both",  # type: ignore[arg-type]
                )
            )
        description = f"Upper eyelid to pupil distance: right {mrd1['right']:.1f} mm, left {mrd1['left']:.1f} mm."
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=max(baseline_confidence, fatigue_confidence) if baseline_confidence == 0 else baseline_confidence,
            value=float(np.nanmin([mrd1["right"], mrd1["left"]])) if np.isfinite(mrd1["right"]) or np.isfinite(mrd1["left"]) else None,
            unit="mm",
        )

    def _reset_measurements(self) -> None:
        self._samples.clear()
