"""Gait agent: step detection, cadence, step-time asymmetry, stride proxy, arm swing and pelvic drop."""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import asymmetry_ratio, euclidean_distance
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.signal_processing import find_peaks_irregular


@dataclass
class WalkSample:
    """Lower and upper body positions of one walking frame, normalised by leg length."""

    timestamp: float
    ankle_separation: float
    right_arm: float
    left_arm: float


@dataclass
class GaitAnalysis:
    """Gait metrics from the walking step."""

    steps: int
    cadence: float
    step_time_asymmetry: float
    step_length: float
    right_arm_swing: float
    left_arm_swing: float


class GaitAgent(AreaAgent):
    """Detects steps from ankle trajectories using real timestamps and measures walking symmetry."""

    name = "gait"
    area = "posture"
    required_trackers = frozenset({TrackerName.POSE})
    supported_events = frozenset({"steps"})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._walk: dict[str, list[WalkSample]] = defaultdict(list)
        self._pelvic_drop: dict[str, list[float]] = defaultdict(list)

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record ankle separation and arm position while walking, or pelvic drop on one leg."""
        pose = frame_context.pose
        if pose is None:
            return []
        minimum_score = self.threshold_float("min_keypoint_score")
        step_instance = frame_context.step_instance or ""
        step_id = base_step_id(step_instance)
        legs = ("left_hip", "right_hip", "left_ankle", "right_ankle")
        if not pose.all_visible(legs, minimum_score):
            return []
        leg_length = (
            euclidean_distance(pose.keypoint("left_hip"), pose.keypoint("left_ankle"))
            + euclidean_distance(pose.keypoint("right_hip"), pose.keypoint("right_ankle"))
        ) / 2.0
        if leg_length < 10.0:
            return []
        if step_id in [str(step) for step in self.threshold_list("walking_steps")]:
            separation = float(pose.keypoint("left_ankle")[0] - pose.keypoint("right_ankle")[0]) / leg_length
            arms: dict[str, float] = {}
            for side in ("right", "left"):
                if pose.all_visible((f"{side}_wrist", f"{side}_shoulder"), minimum_score):
                    arms[side] = float(pose.keypoint(f"{side}_wrist")[0] - pose.keypoint(f"{side}_shoulder")[0]) / leg_length
                else:
                    arms[side] = float("nan")
            self._walk[step_instance].append(WalkSample(frame_context.timestamp_s, separation, arms["right"], arms["left"]))
            return [self.make_reading(frame_context, "ankle_separation", separation, "leg lengths", "Horizontal distance between the ankles.")]
        single_leg_steps = {str(key): str(value) for key, value in self.threshold("single_leg_steps").items()}
        if step_id in single_leg_steps:
            stance = single_leg_steps[step_id]
            lifted = "left" if stance == "right" else "right"
            lift = float(pose.keypoint(f"{stance}_ankle")[1] - pose.keypoint(f"{lifted}_ankle")[1]) / leg_length
            if lift < self.threshold_float("stance_lift_leg_fraction"):
                return []
            stance_hip = pose.keypoint(f"{stance}_hip")
            lifted_hip = pose.keypoint(f"{lifted}_hip")
            horizontal = max(abs(float(lifted_hip[0] - stance_hip[0])), 1e-6)
            drop = math.degrees(math.atan2(float(lifted_hip[1] - stance_hip[1]), horizontal))
            self._pelvic_drop[stance].append(drop)
            return [self.make_reading(frame_context, "pelvic_drop", drop, "degrees", f"Drop of the unsupported hip while standing on the {stance} leg.", side=stance)]  # type: ignore[arg-type]
        return []

    def _step_events(self, samples: list[WalkSample]) -> np.ndarray:
        if len(samples) < 5:
            return np.zeros(0)
        timestamps = np.array([sample.timestamp for sample in samples])
        separation = np.array([sample.ankle_separation for sample in samples])
        prominence = self.threshold_float("step_prominence_leg_fraction")
        interval = self.threshold_float("min_step_interval_seconds")
        maxima = find_peaks_irregular(timestamps, separation, interval, prominence)
        minima = find_peaks_irregular(timestamps, -separation, interval, prominence)
        return np.sort(np.concatenate([maxima, minima])).astype(int)

    def count_events(self, step_instance: str, event: str) -> int:
        """Number of steps detected so far."""
        if event != "steps":
            return 0
        return int(len(self._step_events(self._walk.get(step_instance, []))))

    def _analyse(self, samples: list[WalkSample]) -> GaitAnalysis | None:
        events = self._step_events(samples)
        if len(events) < self.threshold_int("min_steps_for_analysis"):
            return None
        timestamps = np.array([sample.timestamp for sample in samples])
        separation = np.array([sample.ankle_separation for sample in samples])
        step_times = np.diff(timestamps[events])
        median_step = float(np.median(step_times))
        steady = step_times[step_times <= 2.0 * median_step]
        if len(steady) < 2:
            return None
        cadence = 60.0 / float(np.median(steady))
        odd = steady[0::2]
        even = steady[1::2]
        asymmetry = abs(float(np.mean(odd)) - float(np.mean(even))) / float(np.mean(steady)) if len(even) else 0.0
        step_length = float(np.median(np.abs(separation[events])))
        swings: dict[str, float] = {}
        for side in ("right", "left"):
            values = np.array([sample.right_arm if side == "right" else sample.left_arm for sample in samples])
            values = values[np.isfinite(values)]
            swings[side] = float(np.percentile(values, 95) - np.percentile(values, 5)) if len(values) >= 10 else float("nan")
        return GaitAnalysis(
            steps=int(len(events)),
            cadence=cadence,
            step_time_asymmetry=asymmetry,
            step_length=step_length,
            right_arm_swing=swings["right"],
            left_arm_swing=swings["left"],
        )

    def finalize(self) -> AgentResult:
        """Evaluate limp, shuffling, reduced arm swing and Trendelenburg pattern thresholds."""
        if not self.steps_with_data:
            return self.insufficient_result("Walking could not be tracked.")
        walking_steps = [str(step) for step in self.threshold_list("walking_steps")]
        single_leg_steps = {str(key): str(value) for key, value in self.threshold("single_leg_steps").items()}
        walking_confidence = self.data_confidence(walking_steps)
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        limitations: list[str] = ["Step detection assumes walking across the camera view; walking toward the camera reduces accuracy."]
        samples = [sample for instance, items in self._walk.items() if base_step_id(instance) in walking_steps for sample in items]
        samples.sort(key=lambda sample: sample.timestamp)
        analysis = self._analyse(samples)
        walking_label = ",".join(walking_steps)
        if analysis is None:
            limitations.append("Too few steps were detected to analyse walking.")
        else:
            metrics.extend(
                [
                    self.summary_reading("steps_detected", float(analysis.steps), "steps", "Steps detected while walking.", walking_confidence, walking_label),
                    self.summary_reading("cadence", analysis.cadence, "steps/min", "Walking cadence.", walking_confidence, walking_label),
                    self.summary_reading("step_time_asymmetry", analysis.step_time_asymmetry, "fraction", "Difference between alternate step times.", walking_confidence, walking_label),
                    self.summary_reading("step_length_proxy", analysis.step_length, "leg lengths", "Ankle separation at each step.", walking_confidence, walking_label),
                    self.summary_reading("arm_swing", analysis.right_arm_swing, "leg lengths", "Right arm swing amplitude.", walking_confidence, walking_label, side="right"),
                    self.summary_reading("arm_swing", analysis.left_arm_swing, "leg lengths", "Left arm swing amplitude.", walking_confidence, walking_label, side="left"),
                ]
            )
            asymmetry_threshold = self.threshold_float("step_time_asymmetry_flag")
            if analysis.step_time_asymmetry >= asymmetry_threshold:
                flags.append(self.make_flag("gait_asymmetry_limp", "Uneven step timing (limp pattern)", FlagSeverity.MODERATE, self.exceedance_confidence(analysis.step_time_asymmetry, asymmetry_threshold, walking_confidence), "Alternate steps took different amounts of time, which is seen with a limp.", walking_steps, evidence=[f"step-time asymmetry {analysis.step_time_asymmetry:.2f}"]))
            shuffle_threshold = self.threshold_float("shuffle_step_length_leg_fraction")
            if analysis.step_length < shuffle_threshold:
                flags.append(self.make_flag("shuffling_gait", "Short, shuffling steps", FlagSeverity.MODERATE, self.shortfall_confidence(analysis.step_length, shuffle_threshold, walking_confidence), "Steps were short relative to leg length, which can occur with Parkinsonism or fear of falling.", walking_steps, evidence=[f"step length proxy {analysis.step_length:.2f} leg lengths"]))
            cadence_threshold = self.threshold_float("cadence_low_steps_per_minute")
            if analysis.cadence < cadence_threshold:
                flags.append(self.make_flag("slow_cadence", "Slow walking rhythm", FlagSeverity.LOW, self.shortfall_confidence(analysis.cadence, cadence_threshold, walking_confidence), "Fewer steps per minute than the educational reference.", walking_steps, evidence=[f"cadence {analysis.cadence:.0f} steps/min"]))
            if np.isfinite(analysis.right_arm_swing) and np.isfinite(analysis.left_arm_swing):
                swing_asymmetry = asymmetry_ratio(analysis.right_arm_swing, analysis.left_arm_swing)
                metrics.append(self.summary_reading("arm_swing_asymmetry", swing_asymmetry, "ratio", "Relative left-right difference in arm swing.", walking_confidence, walking_label))
                reduced_threshold = self.threshold_float("reduced_arm_swing_leg_fraction")
                reduced_sides = [side for side, value in (("right", analysis.right_arm_swing), ("left", analysis.left_arm_swing)) if value < reduced_threshold]
                if swing_asymmetry >= self.threshold_float("arm_swing_asymmetry_flag") or reduced_sides:
                    weaker = "right" if analysis.right_arm_swing < analysis.left_arm_swing else "left"
                    flags.append(self.make_flag("reduced_arm_swing", "Reduced arm swing", FlagSeverity.MODERATE, walking_confidence * max(min(1.0, swing_asymmetry), 0.5), f"The {weaker} arm swung less while walking, which can be an early sign of Parkinsonism or shoulder problems.", walking_steps, evidence=[f"arm swing asymmetry {swing_asymmetry:.2f}", f"reduced: {', '.join(reduced_sides) or 'none'}"], side=weaker))  # type: ignore[arg-type]
        drop_threshold = self.threshold_float("pelvic_drop_degrees_flag")
        for step_id, stance in single_leg_steps.items():
            drop = robust_median(self._pelvic_drop.get(stance, []))
            confidence = self.data_confidence([step_id])
            metrics.append(self.summary_reading("pelvic_drop", drop, "degrees", f"Unsupported hip drop while standing on the {stance} leg.", confidence, step_id, side=stance))  # type: ignore[arg-type]
            if np.isfinite(drop) and drop >= drop_threshold:
                flags.append(self.make_flag("trendelenburg_pattern", "Hip drop on one-leg standing", FlagSeverity.MODERATE, self.exceedance_confidence(drop, drop_threshold, confidence), f"While standing on the {stance} leg, the opposite hip dropped, suggesting weakness of the {stance} hip abductors.", [step_id], evidence=[f"pelvic drop {drop:.1f} degrees"], side=stance))  # type: ignore[arg-type]
        description = f"Walking: {analysis.steps} steps, cadence {analysis.cadence:.0f} steps/min." if analysis else "Walking could not be analysed in detail."
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=walking_confidence if walking_confidence > 0 else self.data_confidence(self.steps_with_data),
            value=analysis.cadence if analysis else None,
            unit="steps/min",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._walk.clear()
        self._pelvic_drop.clear()
