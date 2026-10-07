"""Eye alignment agent: iris position relative to the eye corners and strabismus screening."""

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.models.neural_forge_face import FaceLandmarks
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import euclidean_distance, eye_aspect_ratio, projection_parameter, signed_distance_to_line
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import iris_geometry, landmark_index, landmark_indices


def iris_position(face: FaceLandmarks, side: str) -> tuple[float, float, float]:
    """Return (nasal position 0-1 from outer to inner corner, vertical offset in eye widths, eye width px)."""
    outer = face.point(landmark_index(side, "outer"))
    inner = face.point(landmark_index(side, "inner"))
    center, _ = iris_geometry(face, side)
    width = euclidean_distance(outer, inner)
    nasal_position = projection_parameter(center, outer, inner)
    left_to_right_start, left_to_right_end = (outer, inner) if side == "right" else (inner, outer)
    vertical = signed_distance_to_line(center, left_to_right_start, left_to_right_end - left_to_right_start) / max(width, 1e-6)
    return nasal_position, vertical, width


class EyeAlignmentAgent(AreaAgent):
    """Detects consistent convergent, divergent or vertical misalignment in straight-ahead gaze."""

    name = "eye_alignment"
    area = "eyes"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._horizontal_mm: list[float] = []
        self._vertical_mm: list[float] = []

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure horizontal vergence and vertical offset between eyes for one frame."""
        face = frame_context.face
        if face is None:
            return []
        yaw, _, _ = face.head_pose_degrees()
        if abs(yaw) > self.threshold_float("max_head_yaw_degrees"):
            return []
        blink_threshold = self.threshold_float("blink_ear_threshold")
        for side in ("right", "left"):
            if eye_aspect_ratio(face.points(landmark_indices(side, "ear_points"))) < blink_threshold:
                return []
        right_nasal, right_vertical, right_width = iris_position(face, "right")
        left_nasal, left_vertical, left_width = iris_position(face, "left")
        _, right_radius = iris_geometry(face, "right")
        _, left_radius = iris_geometry(face, "left")
        mean_radius = (right_radius + left_radius) / 2.0
        if mean_radius <= 0.5:
            return []
        millimetres_per_pixel = self.threshold_float("iris_diameter_mm") / (2.0 * mean_radius)
        eye_width_mm = (right_width + left_width) / 2.0 * millimetres_per_pixel
        horizontal_mm = ((right_nasal - 0.5) + (left_nasal - 0.5)) * eye_width_mm
        vertical_mm = (right_vertical - left_vertical) * eye_width_mm
        self._horizontal_mm.append(horizontal_mm)
        self._vertical_mm.append(vertical_mm)
        return [
            self.make_reading(frame_context, "horizontal_deviation", horizontal_mm, "mm", "Combined nasal shift of both irises (positive is convergent, negative is divergent)."),
            self.make_reading(frame_context, "vertical_deviation", vertical_mm, "mm", "Vertical offset between the eyes (positive means the right iris sits lower)."),
        ]

    def finalize(self) -> AgentResult:
        """Flag consistent misalignment beyond screening thresholds."""
        if not self._horizontal_mm:
            return self.insufficient_result("Eye alignment could not be measured in straight-ahead gaze.")
        confidence = self.data_confidence(self.steps_with_data)
        steps = ",".join(self.steps_with_data)
        horizontal = np.array(self._horizontal_mm)
        vertical = np.array(self._vertical_mm)
        horizontal_median = robust_median(self._horizontal_mm)
        vertical_median = robust_median(self._vertical_mm)
        horizontal_threshold = self.threshold_float("horizontal_deviation_mm")
        vertical_threshold = self.threshold_float("vertical_deviation_mm")
        consistency = self.threshold_float("consistency_fraction")
        horizontal_fraction = float((np.abs(horizontal) > horizontal_threshold).mean())
        vertical_fraction = float((np.abs(vertical) > vertical_threshold).mean())
        metrics = [
            self.summary_reading("horizontal_deviation", horizontal_median, "mm", "Median horizontal misalignment (positive convergent).", confidence, steps),
            self.summary_reading("vertical_deviation", vertical_median, "mm", "Median vertical misalignment (positive right lower).", confidence, steps),
            self.summary_reading("horizontal_deviation_consistency", horizontal_fraction, "fraction", "Fraction of frames beyond the horizontal threshold.", confidence, steps),
            self.summary_reading("vertical_deviation_consistency", vertical_fraction, "fraction", "Fraction of frames beyond the vertical threshold.", confidence, steps),
        ]
        flags: list[ScreeningFlag] = []
        if abs(horizontal_median) > horizontal_threshold and horizontal_fraction >= consistency:
            pattern = "inward (convergent)" if horizontal_median > 0 else "outward (divergent)"
            flags.append(
                self.make_flag(
                    "strabismus_horizontal",
                    "Possible horizontal eye misalignment",
                    FlagSeverity.MODERATE,
                    self.exceedance_confidence(horizontal_median, horizontal_threshold, confidence) * horizontal_fraction,
                    f"The eyes did not point in the same direction; one eye appeared to turn {pattern}.",
                    self.steps_with_data,
                    evidence=[f"median deviation {horizontal_median:.2f} mm", f"consistency {horizontal_fraction:.2f}"],
                )
            )
        if abs(vertical_median) > vertical_threshold and vertical_fraction >= consistency:
            lower_eye = "right" if vertical_median > 0 else "left"
            flags.append(
                self.make_flag(
                    "strabismus_vertical",
                    "Possible vertical eye misalignment",
                    FlagSeverity.MODERATE,
                    self.exceedance_confidence(vertical_median, vertical_threshold, confidence) * vertical_fraction,
                    f"The {lower_eye} eye sat lower than the other in straight-ahead gaze.",
                    self.steps_with_data,
                    evidence=[f"median vertical deviation {vertical_median:.2f} mm", f"consistency {vertical_fraction:.2f}"],
                    side=lower_eye,  # type: ignore[arg-type]
                )
            )
        return self.build_result(
            description=f"Straight-ahead eye alignment: horizontal {horizontal_median:+.2f} mm, vertical {vertical_median:+.2f} mm.",
            metrics=metrics,
            flags=flags,
            confidence=confidence,
            value=horizontal_median,
            unit="mm",
        )

    def _reset_measurements(self) -> None:
        self._horizontal_mm.clear()
        self._vertical_mm.clear()
