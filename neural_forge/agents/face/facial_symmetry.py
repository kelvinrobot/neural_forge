"""Facial symmetry agent: stroke-pattern versus whole-side facial weakness screening."""

from collections import defaultdict

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import asymmetry_ratio, euclidean_distance, eye_aspect_ratio
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import (
    LEFT_ALAR_BASE,
    LEFT_BROW_MID,
    LEFT_EYE_EAR_POINTS,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_FACE_CONTOUR,
    LOWER_LIP_INNER,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
    RIGHT_ALAR_BASE,
    RIGHT_BROW_MID,
    RIGHT_EYE_EAR_POINTS,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_FACE_CONTOUR,
    UPPER_LIP_INNER,
    face_aligned_pixels,
    inter_ocular_distance,
)


class FacialSymmetryAgent(AreaAgent):
    """Compares left and right facial movement at rest, smiling, brow raise, eye closure and cheek puff."""

    name = "facial_symmetry"
    area = "face"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._series: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def _role_steps(self, key: str) -> list[str]:
        return [str(step_id) for step_id in self.threshold_list(key)]

    def _values(self, step_ids: list[str], key: str) -> list[float]:
        values: list[float] = []
        for step_id in step_ids:
            values.extend(self._series.get(step_id, {}).get(key, []))
        return values

    def _median(self, step_ids: list[str], key: str) -> float:
        return robust_median(self._values(step_ids, key))

    def _percentile(self, step_ids: list[str], key: str, percentile: float) -> float:
        values = [value for value in self._values(step_ids, key) if np.isfinite(value)]
        if not values:
            return float("nan")
        return float(np.percentile(values, percentile))

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure normalised left/right facial geometry for one frame."""
        face = frame_context.face
        if face is None:
            return []
        distance = inter_ocular_distance(face)
        if distance < 1.0:
            return []
        points = face_aligned_pixels(face) / distance
        right_eye_level = (points[RIGHT_EYE_OUTER, 1] + points[RIGHT_EYE_INNER, 1]) / 2.0
        left_eye_level = (points[LEFT_EYE_OUTER, 1] + points[LEFT_EYE_INNER, 1]) / 2.0
        measures = {
            "corner_height_difference": float(points[MOUTH_RIGHT_CORNER, 1] - points[MOUTH_LEFT_CORNER, 1]),
            "right_corner_x": float(points[MOUTH_RIGHT_CORNER, 0]),
            "right_corner_y": float(points[MOUTH_RIGHT_CORNER, 1]),
            "left_corner_x": float(points[MOUTH_LEFT_CORNER, 0]),
            "left_corner_y": float(points[MOUTH_LEFT_CORNER, 1]),
            "right_nasolabial": euclidean_distance(points[RIGHT_ALAR_BASE], points[MOUTH_RIGHT_CORNER]),
            "left_nasolabial": euclidean_distance(points[LEFT_ALAR_BASE], points[MOUTH_LEFT_CORNER]),
            "right_brow_height": float(right_eye_level - points[RIGHT_BROW_MID, 1]),
            "left_brow_height": float(left_eye_level - points[LEFT_BROW_MID, 1]),
            "right_ear": eye_aspect_ratio(face.points(RIGHT_EYE_EAR_POINTS)),
            "left_ear": eye_aspect_ratio(face.points(LEFT_EYE_EAR_POINTS)),
            "right_cheek_width": float(abs(points[RIGHT_FACE_CONTOUR, 0])),
            "left_cheek_width": float(abs(points[LEFT_FACE_CONTOUR, 0])),
            "lip_gap": euclidean_distance(points[UPPER_LIP_INNER], points[LOWER_LIP_INNER]),
        }
        step_id = base_step_id(frame_context.step_instance or "")
        for key, value in measures.items():
            self._series[step_id][key].append(value)
        readings = [
            self.make_reading(
                frame_context,
                "mouth_corner_height_difference",
                measures["corner_height_difference"],
                "IOD",
                "Vertical difference between mouth corners (positive means the right corner is lower).",
            )
        ]
        baseline_steps = self._role_steps("baseline_steps")
        if step_id in self._role_steps("smile_steps"):
            for side in ("right", "left"):
                baseline_x = self._median(baseline_steps, f"{side}_corner_x")
                baseline_y = self._median(baseline_steps, f"{side}_corner_y")
                if np.isfinite(baseline_x) and np.isfinite(baseline_y):
                    excursion = float(np.hypot(measures[f"{side}_corner_x"] - baseline_x, measures[f"{side}_corner_y"] - baseline_y))
                    readings.append(
                        self.make_reading(frame_context, "smile_excursion", excursion, "IOD", f"{side.capitalize()} mouth corner movement from rest.", side=side)  # type: ignore[arg-type]
                    )
        elif step_id in self._role_steps("brow_steps"):
            for side in ("right", "left"):
                baseline_height = self._median(baseline_steps, f"{side}_brow_height")
                if np.isfinite(baseline_height):
                    readings.append(
                        self.make_reading(
                            frame_context,
                            "brow_raise",
                            measures[f"{side}_brow_height"] - baseline_height,
                            "IOD",
                            f"{side.capitalize()} eyebrow elevation from rest.",
                            side=side,  # type: ignore[arg-type]
                        )
                    )
        elif step_id in self._role_steps("eye_closure_steps"):
            for side in ("right", "left"):
                readings.append(
                    self.make_reading(frame_context, "eye_aspect_ratio", measures[f"{side}_ear"], "ratio", f"{side.capitalize()} eye opening during tight closure.", side=side)  # type: ignore[arg-type]
                )
        elif step_id in self._role_steps("cheek_puff_steps"):
            for side in ("right", "left"):
                baseline_width = self._median(baseline_steps, f"{side}_cheek_width")
                if np.isfinite(baseline_width):
                    readings.append(
                        self.make_reading(
                            frame_context,
                            "cheek_expansion",
                            measures[f"{side}_cheek_width"] - baseline_width,
                            "IOD",
                            f"{side.capitalize()} cheek expansion from rest.",
                            side=side,  # type: ignore[arg-type]
                        )
                    )
        return readings

    def _corner_excursions(self, smile_steps: list[str], baseline_steps: list[str], percentile: float) -> tuple[float, float]:
        excursions: dict[str, float] = {}
        for side in ("right", "left"):
            baseline_x = self._median(baseline_steps, f"{side}_corner_x")
            baseline_y = self._median(baseline_steps, f"{side}_corner_y")
            x_values = self._values(smile_steps, f"{side}_corner_x")
            y_values = self._values(smile_steps, f"{side}_corner_y")
            if not x_values or not np.isfinite(baseline_x) or not np.isfinite(baseline_y):
                excursions[side] = float("nan")
                continue
            displacement = np.hypot(np.array(x_values) - baseline_x, np.array(y_values) - baseline_y)
            excursions[side] = float(np.percentile(displacement, percentile))
        return excursions["right"], excursions["left"]

    def finalize(self) -> AgentResult:
        """Combine rest, smile, brow, eye-closure and cheek findings into pattern flags."""
        if not self.steps_with_data:
            return self.insufficient_result("No usable face frames were captured for the facial symmetry assessment.")
        baseline_steps = self._role_steps("baseline_steps")
        smile_steps = self._role_steps("smile_steps")
        brow_steps = self._role_steps("brow_steps")
        eye_steps = self._role_steps("eye_closure_steps")
        cheek_steps = self._role_steps("cheek_puff_steps")
        percentile = self.threshold_float("excursion_percentile")
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []

        baseline_confidence = self.data_confidence(baseline_steps)
        resting_difference = self._median(baseline_steps, "corner_height_difference")
        right_nasolabial = self._median(baseline_steps, "right_nasolabial")
        left_nasolabial = self._median(baseline_steps, "left_nasolabial")
        nasolabial_asymmetry = asymmetry_ratio(right_nasolabial, left_nasolabial) if np.isfinite(right_nasolabial) and np.isfinite(left_nasolabial) else float("nan")
        metrics.append(self.summary_reading("resting_mouth_corner_difference", resting_difference, "IOD", "Median mouth corner height difference at rest (positive: right lower).", baseline_confidence, ",".join(baseline_steps)))
        metrics.append(self.summary_reading("nasolabial_asymmetry_ratio", nasolabial_asymmetry, "ratio", "Relative left/right difference in nose-to-mouth-corner distance at rest.", baseline_confidence, ",".join(baseline_steps)))

        smile_confidence = min(self.data_confidence(smile_steps), max(baseline_confidence, 0.0))
        right_excursion, left_excursion = self._corner_excursions(smile_steps, baseline_steps, percentile)
        smile_asymmetry = asymmetry_ratio(right_excursion, left_excursion) if np.isfinite(right_excursion) and np.isfinite(left_excursion) else float("nan")
        smile_effort = np.isfinite(smile_asymmetry) and max(right_excursion, left_excursion) >= self.threshold_float("min_smile_excursion_iod")
        metrics.append(self.summary_reading("smile_excursion", right_excursion, "IOD", "Right mouth corner smile excursion.", smile_confidence, ",".join(smile_steps), side="right"))
        metrics.append(self.summary_reading("smile_excursion", left_excursion, "IOD", "Left mouth corner smile excursion.", smile_confidence, ",".join(smile_steps), side="left"))
        metrics.append(self.summary_reading("smile_asymmetry_ratio", smile_asymmetry, "ratio", "Relative left/right difference in smile excursion.", smile_confidence, ",".join(smile_steps)))
        if np.isfinite(smile_asymmetry) and not smile_effort:
            limitations.append("The smile was too small to compare both sides reliably.")

        brow_confidence = min(self.data_confidence(brow_steps), max(baseline_confidence, 0.0))
        right_raise = self._percentile(brow_steps, "right_brow_height", percentile) - self._median(baseline_steps, "right_brow_height")
        left_raise = self._percentile(brow_steps, "left_brow_height", percentile) - self._median(baseline_steps, "left_brow_height")
        brow_available = np.isfinite(right_raise) and np.isfinite(left_raise)
        brow_effort = brow_available and max(right_raise, left_raise) >= self.threshold_float("min_brow_raise_iod")
        brow_asymmetry = asymmetry_ratio(max(right_raise, 0.0), max(left_raise, 0.0)) if brow_available else float("nan")
        metrics.append(self.summary_reading("brow_raise", right_raise, "IOD", "Right eyebrow elevation during brow raise.", brow_confidence, ",".join(brow_steps), side="right"))
        metrics.append(self.summary_reading("brow_raise", left_raise, "IOD", "Left eyebrow elevation during brow raise.", brow_confidence, ",".join(brow_steps), side="left"))
        metrics.append(self.summary_reading("brow_raise_asymmetry_ratio", brow_asymmetry, "ratio", "Relative left/right difference in forehead movement.", brow_confidence, ",".join(brow_steps)))
        if brow_available and not brow_effort:
            limitations.append("Eyebrow raising was too small to judge forehead movement.")

        eye_confidence = self.data_confidence(eye_steps)
        right_closed_ear = self._median(eye_steps, "right_ear")
        left_closed_ear = self._median(eye_steps, "left_ear")
        metrics.append(self.summary_reading("closed_eye_aspect_ratio", right_closed_ear, "ratio", "Right eye opening during tight closure.", eye_confidence, ",".join(eye_steps), side="right"))
        metrics.append(self.summary_reading("closed_eye_aspect_ratio", left_closed_ear, "ratio", "Left eye opening during tight closure.", eye_confidence, ",".join(eye_steps), side="left"))
        incomplete_threshold = self.threshold_float("eye_closure_incomplete_ear")
        incomplete_sides = [
            side
            for side, value in (("right", right_closed_ear), ("left", left_closed_ear))
            if np.isfinite(value) and value > incomplete_threshold
        ]
        eye_asymmetry = abs(right_closed_ear - left_closed_ear) if np.isfinite(right_closed_ear) and np.isfinite(left_closed_ear) else float("nan")

        cheek_confidence = min(self.data_confidence(cheek_steps), max(baseline_confidence, 0.0))
        right_expansion = self._percentile(cheek_steps, "right_cheek_width", percentile) - self._median(baseline_steps, "right_cheek_width")
        left_expansion = self._percentile(cheek_steps, "left_cheek_width", percentile) - self._median(baseline_steps, "left_cheek_width")
        lip_gap = self._median(cheek_steps, "lip_gap")
        cheek_available = np.isfinite(right_expansion) and np.isfinite(left_expansion)
        cheek_effort = cheek_available and max(right_expansion, left_expansion) >= self.threshold_float("min_cheek_expansion_iod")
        cheek_asymmetry = asymmetry_ratio(max(right_expansion, 0.0), max(left_expansion, 0.0)) if cheek_available else float("nan")
        metrics.append(self.summary_reading("cheek_puff_asymmetry_ratio", cheek_asymmetry, "ratio", "Relative left/right difference in cheek expansion.", cheek_confidence, ",".join(cheek_steps)))
        metrics.append(self.summary_reading("cheek_puff_lip_gap", lip_gap, "IOD", "Lip gap while puffing the cheeks (air leak indicator).", cheek_confidence, ",".join(cheek_steps)))

        resting_threshold = self.threshold_float("resting_corner_difference_iod")
        smile_threshold = self.threshold_float("smile_excursion_asymmetry_ratio")
        resting_flag = np.isfinite(resting_difference) and abs(resting_difference) >= resting_threshold
        smile_flag = bool(smile_effort) and smile_asymmetry >= smile_threshold
        lower_weak_side: str | None = None
        lower_confidence = 0.0
        if smile_flag:
            lower_weak_side = "right" if right_excursion < left_excursion else "left"
            lower_confidence = self.exceedance_confidence(smile_asymmetry, smile_threshold, smile_confidence)
        if resting_flag:
            resting_side = "right" if resting_difference > 0 else "left"
            resting_confidence = self.exceedance_confidence(resting_difference, resting_threshold, baseline_confidence)
            if lower_weak_side is None:
                lower_weak_side = resting_side
                lower_confidence = resting_confidence
            elif lower_weak_side == resting_side:
                lower_confidence = max(lower_confidence, resting_confidence)

        brow_threshold = self.threshold_float("brow_raise_asymmetry_ratio")
        brow_weak_side = "right" if brow_available and right_raise < left_raise else "left"
        brow_flag = bool(brow_effort) and brow_asymmetry >= brow_threshold
        upper_involved = lower_weak_side is not None and (
            (brow_flag and brow_weak_side == lower_weak_side) or lower_weak_side in incomplete_sides
        )

        if lower_weak_side is not None and upper_involved:
            upper_confidence = self.exceedance_confidence(brow_asymmetry, brow_threshold, brow_confidence) if brow_flag else eye_confidence * 0.6
            flags.append(
                self.make_flag(
                    "whole_side_facial_weakness",
                    "Whole-side facial weakness pattern",
                    FlagSeverity.HIGH,
                    (lower_confidence + upper_confidence) / 2.0,
                    f"Reduced movement of both the lower face and the forehead/eye on the {lower_weak_side} side, a pattern seen in peripheral (Bell's-palsy-like) facial weakness.",
                    smile_steps + brow_steps + eye_steps,
                    evidence=[f"smile asymmetry {smile_asymmetry:.2f}", f"brow asymmetry {brow_asymmetry:.2f}", f"incomplete closure sides: {incomplete_sides}"],
                    side=lower_weak_side,  # type: ignore[arg-type]
                )
            )
        elif lower_weak_side is not None and brow_effort:
            flags.append(
                self.make_flag(
                    "facial_asymmetry_forehead_sparing",
                    "Lower facial weakness with forehead sparing",
                    FlagSeverity.HIGH,
                    lower_confidence,
                    f"The lower face on the {lower_weak_side} side moved less while forehead movement was symmetric, a pattern that can occur with a stroke and needs urgent assessment.",
                    baseline_steps + smile_steps + brow_steps,
                    evidence=[f"smile asymmetry {smile_asymmetry:.2f}", f"resting corner difference {resting_difference:.3f} IOD", f"brow asymmetry {brow_asymmetry:.2f}"],
                    side=lower_weak_side,  # type: ignore[arg-type]
                )
            )
        elif lower_weak_side is not None:
            flags.append(
                self.make_flag(
                    "lower_facial_asymmetry",
                    "Lower facial asymmetry",
                    FlagSeverity.MODERATE,
                    lower_confidence,
                    f"The {lower_weak_side} side of the lower face moved less or sat lower; forehead movement could not be compared.",
                    baseline_steps + smile_steps,
                    evidence=[f"smile asymmetry {smile_asymmetry:.2f}", f"resting corner difference {resting_difference:.3f} IOD"],
                    side=lower_weak_side,  # type: ignore[arg-type]
                )
            )
            limitations.append("Forehead movement could not be assessed, so the weakness pattern could not be classified.")

        nasolabial_threshold = self.threshold_float("nasolabial_asymmetry_ratio")
        if lower_weak_side is None and np.isfinite(nasolabial_asymmetry) and nasolabial_asymmetry >= nasolabial_threshold:
            flags.append(
                self.make_flag(
                    "nasolabial_fold_asymmetry",
                    "Nose-to-mouth fold asymmetry",
                    FlagSeverity.LOW,
                    self.exceedance_confidence(nasolabial_asymmetry, nasolabial_threshold, baseline_confidence),
                    "The distance from the side of the nose to the mouth corner differed between sides at rest.",
                    baseline_steps,
                    evidence=[f"nasolabial asymmetry {nasolabial_asymmetry:.2f}"],
                )
            )

        eye_asymmetry_threshold = self.threshold_float("eye_closure_asymmetry_ear")
        if incomplete_sides and not upper_involved:
            flags.append(
                self.make_flag(
                    "incomplete_eye_closure",
                    "Incomplete eye closure",
                    FlagSeverity.MODERATE,
                    eye_confidence * 0.8,
                    f"The eye did not close fully on: {', '.join(incomplete_sides)} side during tight closure.",
                    eye_steps,
                    evidence=[f"right EAR {right_closed_ear:.3f}", f"left EAR {left_closed_ear:.3f}"],
                    side=incomplete_sides[0] if len(incomplete_sides) == 1 else "both",  # type: ignore[arg-type]
                )
            )
        elif np.isfinite(eye_asymmetry) and eye_asymmetry >= eye_asymmetry_threshold and not upper_involved:
            weaker_eye = "right" if right_closed_ear > left_closed_ear else "left"
            flags.append(
                self.make_flag(
                    "eye_closure_asymmetry",
                    "Eye closure asymmetry",
                    FlagSeverity.LOW,
                    self.exceedance_confidence(eye_asymmetry, eye_asymmetry_threshold, eye_confidence),
                    f"The {weaker_eye} eye closed less tightly than the other.",
                    eye_steps,
                    evidence=[f"EAR difference {eye_asymmetry:.3f}"],
                    side=weaker_eye,  # type: ignore[arg-type]
                )
            )

        cheek_threshold = self.threshold_float("cheek_puff_asymmetry_ratio")
        lip_seal_threshold = self.threshold_float("lip_seal_gap_iod")
        cheek_flag = bool(cheek_effort) and cheek_asymmetry >= cheek_threshold
        leak_flag = np.isfinite(lip_gap) and lip_gap >= lip_seal_threshold
        if cheek_flag or leak_flag:
            weaker_cheek = "right" if right_expansion < left_expansion else "left"
            confidence = max(
                self.exceedance_confidence(cheek_asymmetry, cheek_threshold, cheek_confidence) if cheek_flag else 0.0,
                self.exceedance_confidence(lip_gap, lip_seal_threshold, cheek_confidence) if leak_flag else 0.0,
            )
            flags.append(
                self.make_flag(
                    "cheek_puff_asymmetry",
                    "Cheek puff air-leak asymmetry",
                    FlagSeverity.MODERATE,
                    confidence,
                    f"Cheek inflation was weaker on the {weaker_cheek} side or the lips did not stay sealed.",
                    cheek_steps,
                    evidence=[f"cheek asymmetry {cheek_asymmetry:.2f}", f"lip gap {lip_gap:.3f} IOD"],
                    side=weaker_cheek,  # type: ignore[arg-type]
                )
            )

        overall_confidence = float(np.mean([value for value in (baseline_confidence, smile_confidence, brow_confidence, eye_confidence, cheek_confidence) if value > 0] or [0.0]))
        if flags:
            description = "Facial movement asymmetry was detected: " + "; ".join(flag.title for flag in flags) + "."
        else:
            description = "Facial movement looked symmetric at rest and during smile, brow raise, eye closure and cheek puff."
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=overall_confidence,
            value=smile_asymmetry,
            unit="ratio",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._series.clear()
