"""Balance agent: centre-of-mass proxy sway, Romberg comparison and single-leg stance duration."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.neural_forge_pose import PoseResult
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import confidence_ellipse_area, euclidean_distance, midpoint, path_length
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.signal_processing import butter_lowpass, contiguous_segments, resample_uniform


@dataclass
class SwayResult:
    """Sway metrics for one condition."""

    velocity: float
    area: float
    duration: float


class BalanceAgent(AreaAgent):
    """Quantifies standing sway from a centre-of-mass proxy using real timestamps."""

    name = "balance"
    area = "posture"
    required_trackers = frozenset({TrackerName.POSE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._sway: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        self._stance: dict[str, list[tuple[float, bool]]] = defaultdict(list)

    def _centre_of_mass(self, pose: PoseResult, minimum_score: float) -> tuple[np.ndarray, float] | None:
        required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose")
        if not pose.all_visible(required, minimum_score):
            return None
        ankles = [pose.keypoint(name) for name in ("left_ankle", "right_ankle") if pose.is_visible(name, minimum_score)]
        if not ankles:
            return None
        ankle_point = np.mean(ankles, axis=0)
        body_height = euclidean_distance(pose.keypoint("nose"), ankle_point)
        if body_height < 20.0:
            return None
        hips = midpoint(pose.keypoint("left_hip"), pose.keypoint("right_hip"))
        shoulders = midpoint(pose.keypoint("left_shoulder"), pose.keypoint("right_shoulder"))
        return 0.6 * hips + 0.4 * shoulders, body_height

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record the normalised centre-of-mass proxy and, on one-leg steps, whether the foot is lifted."""
        pose = frame_context.pose
        if pose is None:
            return []
        minimum_score = self.threshold_float("min_keypoint_score")
        result = self._centre_of_mass(pose, minimum_score)
        if result is None:
            return []
        centre, body_height = result
        step_instance = frame_context.step_instance or ""
        self._sway[step_instance].append((frame_context.timestamp_s, float(centre[0] / body_height), float(centre[1] / body_height)))
        readings = [self.make_reading(frame_context, "centre_of_mass_x", float(centre[0] / body_height), "body heights", "Horizontal centre-of-mass proxy position.")]
        single_leg_steps = {str(key): str(value) for key, value in self.threshold("single_leg_steps").items()}
        step_id = base_step_id(step_instance)
        if step_id in single_leg_steps:
            stance = single_leg_steps[step_id]
            lifted = "left" if stance == "right" else "right"
            if pose.all_visible((f"{stance}_ankle", f"{lifted}_ankle"), minimum_score):
                lift = float(pose.keypoint(f"{stance}_ankle")[1] - pose.keypoint(f"{lifted}_ankle")[1]) / body_height
                raised = lift >= self.threshold_float("foot_lift_body_fraction")
                self._stance[step_instance].append((frame_context.timestamp_s, raised))
                readings.append(self.make_reading(frame_context, "foot_lifted", 1.0 if raised else 0.0, "flag", f"Whether the {lifted} foot is off the floor.", side=stance))  # type: ignore[arg-type]
        return readings

    def _sway_for(self, step_ids: list[str]) -> SwayResult | None:
        best: SwayResult | None = None
        for step_instance, samples in self._sway.items():
            if base_step_id(step_instance) not in step_ids or len(samples) < 10:
                continue
            timestamps = np.array([sample[0] for sample in samples])
            if timestamps[-1] - timestamps[0] < self.threshold_float("min_duration_seconds"):
                continue
            positions = np.array([[sample[1], sample[2]] for sample in samples])
            uniform = resample_uniform(timestamps, positions)
            smoothed = np.stack([butter_lowpass(uniform.values[:, axis], uniform.sampling_rate, self.threshold_float("lowpass_hz")) for axis in range(2)], axis=1)
            duration = uniform.duration
            if duration <= 0:
                continue
            result = SwayResult(velocity=path_length(smoothed) / duration, area=confidence_ellipse_area(smoothed), duration=duration)
            if best is None or result.duration > best.duration:
                best = result
        return best

    def _longest_stance(self, step_id: str) -> float:
        longest = 0.0
        for step_instance, samples in self._stance.items():
            if base_step_id(step_instance) != step_id or len(samples) < 2:
                continue
            timestamps = np.array([sample[0] for sample in samples])
            raised = np.array([sample[1] for sample in samples])
            for start, end, _, _ in contiguous_segments(timestamps, raised):
                longest = max(longest, end - start)
        return longest

    def finalize(self) -> AgentResult:
        """Compare eyes-open and eyes-closed sway and evaluate single-leg stance time."""
        if not self.steps_with_data:
            return self.insufficient_result("Standing balance could not be measured.")
        open_steps = [str(step) for step in self.threshold_list("eyes_open_steps")]
        closed_steps = [str(step) for step in self.threshold_list("eyes_closed_steps")]
        single_leg_steps = {str(key): str(value) for key, value in self.threshold("single_leg_steps").items()}
        open_confidence = self.data_confidence(open_steps)
        closed_confidence = self.data_confidence(closed_steps)
        eyes_open = self._sway_for(open_steps)
        eyes_closed = self._sway_for(closed_steps)
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        if eyes_open is not None:
            metrics.append(self.summary_reading("sway_velocity_eyes_open", eyes_open.velocity, "body heights/s", "Centre-of-mass sway speed with eyes open.", open_confidence, ",".join(open_steps)))
            metrics.append(self.summary_reading("sway_area_eyes_open", eyes_open.area, "body heights^2", "95% sway ellipse area with eyes open.", open_confidence, ",".join(open_steps)))
            velocity_threshold = self.threshold_float("sway_velocity_flag")
            if eyes_open.velocity >= velocity_threshold:
                flags.append(self.make_flag("increased_postural_sway", "Increased standing sway", FlagSeverity.LOW, self.exceedance_confidence(eyes_open.velocity, velocity_threshold, open_confidence), "The body swayed more than expected while standing with feet together and eyes open.", open_steps, evidence=[f"sway velocity {eyes_open.velocity:.3f} body heights/s"]))
        if eyes_closed is not None:
            metrics.append(self.summary_reading("sway_velocity_eyes_closed", eyes_closed.velocity, "body heights/s", "Centre-of-mass sway speed with eyes closed.", closed_confidence, ",".join(closed_steps)))
            metrics.append(self.summary_reading("sway_area_eyes_closed", eyes_closed.area, "body heights^2", "95% sway ellipse area with eyes closed.", closed_confidence, ",".join(closed_steps)))
        if eyes_open is not None and eyes_closed is not None and eyes_open.velocity > 0:
            romberg_ratio = eyes_closed.velocity / eyes_open.velocity
            confidence = min(open_confidence, closed_confidence)
            metrics.append(self.summary_reading("romberg_ratio", romberg_ratio, "ratio", "Eyes-closed to eyes-open sway speed ratio.", confidence, ",".join(open_steps + closed_steps)))
            ratio_threshold = self.threshold_float("romberg_ratio_flag")
            if romberg_ratio >= ratio_threshold and eyes_closed.velocity >= self.threshold_float("min_closed_sway_velocity_for_romberg"):
                flags.append(self.make_flag("romberg_sign", "Much more sway with eyes closed", FlagSeverity.MODERATE, self.exceedance_confidence(romberg_ratio, ratio_threshold, confidence), "Balance worsened markedly when the eyes were closed, which can indicate a problem with position sense or the inner ear.", open_steps + closed_steps, evidence=[f"Romberg ratio {romberg_ratio:.2f}"]))
        minimum_stance = self.threshold_float("single_leg_min_seconds")
        for step_id, stance in single_leg_steps.items():
            if not any(base_step_id(instance) == step_id for instance in self._stance):
                continue
            longest = self._longest_stance(step_id)
            confidence = self.data_confidence([step_id])
            metrics.append(self.summary_reading("single_leg_stance_time", longest, "s", f"Longest time standing on the {stance} leg.", confidence, step_id, side=stance))  # type: ignore[arg-type]
            if longest < minimum_stance:
                flags.append(self.make_flag("reduced_single_leg_stance", "Short one-leg standing time", FlagSeverity.MODERATE, self.shortfall_confidence(longest, minimum_stance, confidence), f"Standing on the {stance} leg lasted {longest:.1f} seconds, below the screening reference of {minimum_stance:.0f} seconds; this is linked to higher fall risk.", [step_id], evidence=[f"{longest:.1f} s"], side=stance))  # type: ignore[arg-type]
        confidence_values = [value for value in (open_confidence, closed_confidence) if value > 0]
        return self.build_result(
            description="Standing balance: " + (", ".join(flag.title for flag in flags) if flags else "within screening reference limits") + ".",
            metrics=metrics,
            flags=flags,
            confidence=float(np.mean(confidence_values)) if confidence_values else self.data_confidence(self.steps_with_data),
            value=eyes_open.velocity if eyes_open is not None else None,
            unit="body heights/s",
        )

    def _reset_measurements(self) -> None:
        self._sway.clear()
        self._stance.clear()
