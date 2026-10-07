"""Static posture agent: shoulder and hip levels, trunk lean, forward head, trunk flexion and knee alignment."""

import math
from collections import defaultdict

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.neural_forge_pose import PoseResult
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import angle_from_vertical_degrees, joint_angle_degrees, midpoint, unit_vector
from neural_forge.services.landmark_service import TrackerName


def level_difference_degrees(pose: PoseResult, left_name: str, right_name: str) -> float:
    """Angle of the line between paired keypoints; positive when the patient's left point is lower."""
    left_point = pose.keypoint(left_name)
    right_point = pose.keypoint(right_name)
    horizontal = abs(float(left_point[0] - right_point[0]))
    if horizontal < 1.0:
        return float("nan")
    return math.degrees(math.atan2(float(left_point[1] - right_point[1]), horizontal))


def knee_frontal_deviation(pose: PoseResult, side: str) -> float:
    """Frontal knee angle deviation in degrees; positive is valgus (knee toward midline), negative varus."""
    other = "left" if side == "right" else "right"
    hip = pose.keypoint(f"{side}_hip")
    knee = pose.keypoint(f"{side}_knee")
    ankle = pose.keypoint(f"{side}_ankle")
    deviation = 180.0 - joint_angle_degrees(hip, knee, ankle)
    if not np.isfinite(deviation):
        return float("nan")
    line_direction = unit_vector(ankle - hip)
    projection = hip + line_direction * float(np.dot(knee - hip, line_direction))
    medial_direction = unit_vector(pose.keypoint(f"{other}_hip") - hip)
    sign = 1.0 if float(np.dot(knee - projection, medial_direction)) >= 0 else -1.0
    return sign * deviation


class StaticPostureAgent(AreaAgent):
    """Measures standing posture from the front, side and back views."""

    name = "static_posture"
    area = "posture"
    required_trackers = frozenset({TrackerName.POSE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._values: dict[str, list[float]] = defaultdict(list)

    def _view(self, step_id: str) -> str | None:
        for view, key in (("front", "front_steps"), ("side", "side_steps"), ("back", "back_steps")):
            if step_id in [str(step) for step in self.threshold_list(key)]:
                return view
        return None

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure view-specific posture angles."""
        pose = frame_context.pose
        if pose is None:
            return []
        view = self._view(base_step_id(frame_context.step_instance or ""))
        if view is None:
            return []
        minimum_score = self.threshold_float("min_keypoint_score")
        readings: list[Reading] = []
        if view in ("front", "back"):
            required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
            if not pose.all_visible(required, minimum_score):
                return []
            shoulder_level = level_difference_degrees(pose, "left_shoulder", "right_shoulder")
            hip_level = level_difference_degrees(pose, "left_hip", "right_hip")
            lean_image = angle_from_vertical_degrees(
                midpoint(pose.keypoint("left_shoulder"), pose.keypoint("right_shoulder")),
                midpoint(pose.keypoint("left_hip"), pose.keypoint("right_hip")),
            )
            trunk_lean_toward_left = lean_image if view == "front" else -lean_image
            for key, value in (("shoulder_level", shoulder_level), ("hip_level", hip_level), ("trunk_lean", trunk_lean_toward_left)):
                if np.isfinite(value):
                    self._values[key].append(value)
            readings.append(self.make_reading(frame_context, "shoulder_level_difference", shoulder_level, "degrees", "Shoulder line tilt (positive: left shoulder lower)."))
            readings.append(self.make_reading(frame_context, "hip_level_difference", hip_level, "degrees", "Hip line tilt (positive: left hip lower)."))
            readings.append(self.make_reading(frame_context, "trunk_lateral_lean", trunk_lean_toward_left, "degrees", "Trunk lean (positive: toward the patient's left)."))
            for side in ("right", "left"):
                if pose.all_visible((f"{side}_hip", f"{side}_knee", f"{side}_ankle"), minimum_score):
                    deviation = knee_frontal_deviation(pose, side)
                    if np.isfinite(deviation):
                        self._values[f"{side}_knee_deviation"].append(deviation)
                        readings.append(self.make_reading(frame_context, "knee_frontal_deviation", deviation, "degrees", f"{side.capitalize()} knee alignment (positive valgus, negative varus).", side=side))  # type: ignore[arg-type]
        else:
            best_side: str | None = None
            best_score = -1.0
            for side in ("right", "left"):
                names = (f"{side}_ear", f"{side}_shoulder", f"{side}_hip")
                if pose.all_visible(names, minimum_score):
                    score = sum(pose.score(name) for name in names)
                    if score > best_score:
                        best_score = score
                        best_side = side
            if best_side is None:
                return []
            ear = pose.keypoint(f"{best_side}_ear")
            shoulder = pose.keypoint(f"{best_side}_shoulder")
            hip = pose.keypoint(f"{best_side}_hip")
            craniovertebral = math.degrees(math.atan2(float(shoulder[1] - ear[1]), max(abs(float(ear[0] - shoulder[0])), 1e-6)))
            trunk_flexion = abs(angle_from_vertical_degrees(shoulder, hip))
            self._values["craniovertebral_angle"].append(craniovertebral)
            self._values["trunk_flexion"].append(trunk_flexion)
            readings.append(self.make_reading(frame_context, "craniovertebral_angle", craniovertebral, "degrees", "Angle of the ear above the shoulder (smaller means a more forward head)."))
            readings.append(self.make_reading(frame_context, "trunk_flexion", trunk_flexion, "degrees", "Forward inclination of the trunk from vertical."))
        return readings

    def finalize(self) -> AgentResult:
        """Evaluate posture asymmetry and alignment thresholds."""
        if not self.steps_with_data:
            return self.insufficient_result("Standing posture could not be measured.")
        frontal_steps = [str(step) for step in self.threshold_list("front_steps")] + [str(step) for step in self.threshold_list("back_steps")]
        side_steps = [str(step) for step in self.threshold_list("side_steps")]
        frontal_confidence = self.data_confidence(frontal_steps)
        side_confidence = self.data_confidence(side_steps)
        medians = {key: robust_median(values) for key, values in self._values.items()}
        shoulder = medians.get("shoulder_level", float("nan"))
        hip = medians.get("hip_level", float("nan"))
        lean = medians.get("trunk_lean", float("nan"))
        craniovertebral = medians.get("craniovertebral_angle", float("nan"))
        flexion = medians.get("trunk_flexion", float("nan"))
        frontal_label = ",".join(frontal_steps)
        side_label = ",".join(side_steps)
        metrics = [
            self.summary_reading("shoulder_level_difference", shoulder, "degrees", "Median shoulder tilt (positive: left lower).", frontal_confidence, frontal_label),
            self.summary_reading("hip_level_difference", hip, "degrees", "Median hip tilt (positive: left lower).", frontal_confidence, frontal_label),
            self.summary_reading("trunk_lateral_lean", lean, "degrees", "Median trunk lean (positive: toward the left).", frontal_confidence, frontal_label),
            self.summary_reading("craniovertebral_angle", craniovertebral, "degrees", "Median craniovertebral angle.", side_confidence, side_label),
            self.summary_reading("trunk_flexion", flexion, "degrees", "Median trunk forward inclination.", side_confidence, side_label),
        ]
        flags: list[ScreeningFlag] = []
        checks = (
            ("shoulder_tilt_degrees", shoulder, "shoulder_height_asymmetry", "Uneven shoulder height", "One shoulder sat lower than the other ({side} lower)."),
            ("hip_tilt_degrees", hip, "pelvic_obliquity", "Uneven hip height", "One hip sat lower than the other ({side} lower)."),
            ("trunk_lean_degrees", lean, "trunk_lateral_lean", "Trunk leaning to one side", "The trunk leaned toward the {side} side."),
        )
        for threshold_key, value, code, title, template in checks:
            threshold = self.threshold_float(threshold_key)
            if np.isfinite(value) and abs(value) >= threshold:
                side = "left" if value > 0 else "right"
                flags.append(
                    self.make_flag(code, title, FlagSeverity.LOW, self.exceedance_confidence(value, threshold, frontal_confidence), template.format(side=side), frontal_steps, evidence=[f"{value:+.1f} degrees"], side=side)  # type: ignore[arg-type]
                )
        cva_threshold = self.threshold_float("craniovertebral_angle_min_degrees")
        if np.isfinite(craniovertebral) and craniovertebral < cva_threshold:
            flags.append(
                self.make_flag("forward_head_posture", "Forward head posture", FlagSeverity.LOW, self.shortfall_confidence(craniovertebral, cva_threshold, side_confidence), "The head was held forward of the shoulders.", side_steps, evidence=[f"craniovertebral angle {craniovertebral:.1f} degrees"])
            )
        flexion_threshold = self.threshold_float("trunk_flexion_degrees")
        if np.isfinite(flexion) and flexion >= flexion_threshold:
            flags.append(
                self.make_flag("stooped_posture", "Stooped or hunched posture", FlagSeverity.LOW, self.exceedance_confidence(flexion, flexion_threshold, side_confidence), "The trunk was bent forward while standing.", side_steps, evidence=[f"trunk flexion {flexion:.1f} degrees"])
            )
        knee_threshold = self.threshold_float("knee_frontal_deviation_degrees")
        for side in ("right", "left"):
            deviation = medians.get(f"{side}_knee_deviation", float("nan"))
            metrics.append(self.summary_reading("knee_frontal_deviation", deviation, "degrees", f"{side.capitalize()} knee alignment (positive valgus).", frontal_confidence, frontal_label, side=side))  # type: ignore[arg-type]
            if np.isfinite(deviation) and abs(deviation) >= knee_threshold:
                valgus = deviation > 0
                flags.append(
                    self.make_flag(
                        "knee_valgus" if valgus else "knee_varus",
                        "Knock-knee alignment" if valgus else "Bow-leg alignment",
                        FlagSeverity.LOW,
                        self.exceedance_confidence(deviation, knee_threshold, frontal_confidence),
                        f"The {side} knee angled {'inward' if valgus else 'outward'} relative to the hip and ankle.",
                        frontal_steps,
                        evidence=[f"{deviation:+.1f} degrees"],
                        side=side,  # type: ignore[arg-type]
                    )
                )
        confidence_values = [value for value in (frontal_confidence, side_confidence) if value > 0]
        return self.build_result(
            description=f"Shoulder tilt {shoulder:+.1f} deg, hip tilt {hip:+.1f} deg, craniovertebral angle {craniovertebral:.1f} deg.",
            metrics=metrics,
            flags=flags,
            confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
            value=shoulder,
            unit="degrees",
            limitations=["Posture angles assume a level camera; a tilted camera changes level measurements."],
        )

    def _reset_measurements(self) -> None:
        self._values.clear()
