"""Range of motion agent: shoulder, elbow, hip and knee angles, maximum range, symmetry and repetitions."""

from collections import defaultdict

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.rtmw_pose import PoseResult
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import joint_angle_degrees, vector_angle_degrees
from neural_forge.services.landmark_service import TrackerName

SeriesKey = tuple[str, str, str]


def count_repetitions(values: list[float], high: float, low: float) -> int:
    """Count excursions above a high threshold that return below a low threshold."""
    repetitions = 0
    raised = False
    for value in values:
        if not np.isfinite(value):
            continue
        if not raised and value >= high:
            raised = True
        elif raised and value <= low:
            raised = False
            repetitions += 1
    return repetitions


class RangeOfMotionAgent(AreaAgent):
    """Tracks joint angles across movement steps and counts repetitions."""

    name = "range_of_motion"
    area = "posture"
    required_trackers = frozenset({TrackerName.POSE})
    supported_events = frozenset({"repetitions"})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._series: dict[SeriesKey, list[float]] = defaultdict(list)

    def _role(self, step_id: str) -> str | None:
        for role, key in (("shoulder_flexion", "shoulder_flexion_steps"), ("shoulder_abduction", "shoulder_abduction_steps"), ("knee", "knee_steps")):
            if step_id in [str(step) for step in self.threshold_list(key)]:
                return role
        return None

    def _shoulder_elevation(self, pose: PoseResult, side: str, minimum_score: float) -> float:
        if not pose.all_visible((f"{side}_shoulder", f"{side}_hip"), minimum_score):
            return float("nan")
        distal = f"{side}_wrist" if pose.is_visible(f"{side}_wrist", minimum_score) else f"{side}_elbow"
        if not pose.is_visible(distal, minimum_score):
            return float("nan")
        shoulder = pose.keypoint(f"{side}_shoulder")
        return vector_angle_degrees(pose.keypoint(f"{side}_hip") - shoulder, pose.keypoint(distal) - shoulder)

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure the joint angles relevant to the current movement step."""
        pose = frame_context.pose
        if pose is None:
            return []
        step_instance = frame_context.step_instance or ""
        role = self._role(base_step_id(step_instance))
        if role is None:
            return []
        minimum_score = self.threshold_float("min_keypoint_score")
        readings: list[Reading] = []
        for side in ("right", "left"):
            if role in ("shoulder_flexion", "shoulder_abduction"):
                elevation = self._shoulder_elevation(pose, side, minimum_score)
                if np.isfinite(elevation):
                    self._series[(step_instance, side, "shoulder")].append(elevation)
                    readings.append(self.make_reading(frame_context, f"{role}_angle", elevation, "degrees", f"{side.capitalize()} arm elevation angle.", side=side))  # type: ignore[arg-type]
                if pose.all_visible((f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"), minimum_score):
                    elbow = joint_angle_degrees(pose.keypoint(f"{side}_shoulder"), pose.keypoint(f"{side}_elbow"), pose.keypoint(f"{side}_wrist"))
                    self._series[(step_instance, side, "elbow")].append(elbow)
            else:
                if pose.all_visible((f"{side}_hip", f"{side}_knee", f"{side}_ankle"), minimum_score):
                    knee_flexion = 180.0 - joint_angle_degrees(pose.keypoint(f"{side}_hip"), pose.keypoint(f"{side}_knee"), pose.keypoint(f"{side}_ankle"))
                    self._series[(step_instance, side, "knee")].append(knee_flexion)
                    readings.append(self.make_reading(frame_context, "knee_flexion_angle", knee_flexion, "degrees", f"{side.capitalize()} knee flexion.", side=side))  # type: ignore[arg-type]
                if pose.all_visible((f"{side}_shoulder", f"{side}_hip", f"{side}_knee"), minimum_score):
                    hip_flexion = 180.0 - joint_angle_degrees(pose.keypoint(f"{side}_shoulder"), pose.keypoint(f"{side}_hip"), pose.keypoint(f"{side}_knee"))
                    self._series[(step_instance, side, "hip")].append(hip_flexion)
        return readings

    def _repetitions(self, step_instance: str, side: str) -> int:
        role = self._role(base_step_id(step_instance))
        if role is None:
            return 0
        if role == "knee":
            return count_repetitions(
                self._series.get((step_instance, side, "knee"), []),
                self.threshold_float("knee_rep_high_flexion_degrees"),
                self.threshold_float("knee_rep_low_flexion_degrees"),
            )
        return count_repetitions(
            self._series.get((step_instance, side, "shoulder"), []),
            self.threshold_float("shoulder_rep_high_degrees"),
            self.threshold_float("shoulder_rep_low_degrees"),
        )

    def count_events(self, step_instance: str, event: str) -> int:
        """Repetitions completed by both sides so far."""
        if event != "repetitions":
            return 0
        return min(self._repetitions(step_instance, "right"), self._repetitions(step_instance, "left"))

    def _maximum(self, role_steps: list[str], side: str, joint: str) -> float:
        values = [
            value
            for (step_instance, series_side, series_joint), series in self._series.items()
            if base_step_id(step_instance) in role_steps and series_side == side and series_joint == joint
            for value in series
            if np.isfinite(value)
        ]
        if not values:
            return float("nan")
        return float(np.percentile(values, self.threshold_float("max_percentile")))

    def finalize(self) -> AgentResult:
        """Report maximum ranges, left-right differences and limited range flags."""
        if not self.steps_with_data:
            return self.insufficient_result("Joint movements could not be measured.")
        definitions = (
            ("shoulder_flexion_steps", "shoulder", "shoulder_flexion_min_degrees", "shoulder_flexion", "shoulder forward elevation"),
            ("shoulder_abduction_steps", "shoulder", "shoulder_abduction_min_degrees", "shoulder_abduction", "shoulder sideways elevation"),
            ("knee_steps", "knee", "knee_flexion_min_degrees", "knee_flexion", "knee bending"),
            ("knee_steps", "hip", "hip_flexion_min_degrees", "hip_flexion", "hip bending"),
            ("shoulder_flexion_steps", "elbow", "elbow_extension_min_degrees", "elbow_extension", "elbow straightening"),
        )
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        asymmetry_threshold = self.threshold_float("asymmetry_degrees")
        for steps_key, joint, threshold_key, label, description in definitions:
            role_steps = [str(step) for step in self.threshold_list(steps_key)]
            confidence = self.data_confidence(role_steps)
            if confidence == 0:
                continue
            threshold = self.threshold_float(threshold_key)
            maxima: dict[str, float] = {}
            for side in ("right", "left"):
                maximum = self._maximum(role_steps, side, joint)
                maxima[side] = maximum
                metrics.append(self.summary_reading(f"max_{label}", maximum, "degrees", f"{side.capitalize()} maximum {description}.", confidence, ",".join(role_steps), side=side))  # type: ignore[arg-type]
                if np.isfinite(maximum) and maximum < threshold:
                    flags.append(
                        self.make_flag(
                            f"limited_{label}",
                            f"Limited {description}",
                            FlagSeverity.LOW,
                            self.shortfall_confidence(maximum, threshold, confidence),
                            f"The {side} side reached {maximum:.0f} degrees of {description}, below the educational reference of {threshold:.0f} degrees.",
                            role_steps,
                            evidence=[f"{maximum:.0f} degrees"],
                            side=side,  # type: ignore[arg-type]
                        )
                    )
            if np.isfinite(maxima["right"]) and np.isfinite(maxima["left"]):
                difference = abs(maxima["right"] - maxima["left"])
                metrics.append(self.summary_reading(f"{label}_asymmetry", difference, "degrees", f"Left-right difference in maximum {description}.", confidence, ",".join(role_steps)))
                if difference >= asymmetry_threshold and joint != "elbow":
                    weaker = "right" if maxima["right"] < maxima["left"] else "left"
                    flags.append(
                        self.make_flag(
                            f"{label}_asymmetry",
                            f"Left-right difference in {description}",
                            FlagSeverity.LOW,
                            self.exceedance_confidence(difference, asymmetry_threshold, confidence),
                            f"The {weaker} side moved through a smaller range than the other side.",
                            role_steps,
                            evidence=[f"difference {difference:.0f} degrees"],
                            side=weaker,  # type: ignore[arg-type]
                        )
                    )
        for step_instance in self.steps_with_data:
            repetitions = self.count_events(step_instance, "repetitions")
            metrics.append(self.summary_reading("repetitions", float(repetitions), "count", f"Repetitions completed during {step_instance}.", self.data_confidence([step_instance]), step_instance))
        return self.build_result(
            description="Joint range of motion: " + (", ".join(flag.title for flag in flags) if flags else "within educational reference ranges") + ".",
            metrics=metrics,
            flags=flags,
            confidence=self.data_confidence(self.steps_with_data),
            limitations=["Angles are measured in the camera plane; movement toward or away from the camera is under-estimated."],
        )

    def _reset_measurements(self) -> None:
        self._series.clear()
