"""Functional test agent: automatic Timed Up and Go and five-times sit-to-stand timing from a pose state machine."""

import math
from dataclasses import dataclass, field

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.rtmw_pose import PoseResult
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import joint_angle_degrees, midpoint
from neural_forge.services.landmark_service import TrackerName

PHASE_CODES: dict[str, float] = {"sitting": 0.0, "rising": 1.0, "walking": 2.0, "returning": 3.0, "seated": 4.0, "standing": 5.0}


@dataclass
class FunctionalState:
    """State machine for one functional test step instance."""

    test: str
    posture: str = "unknown"
    candidate: str = "unknown"
    candidate_since: float = 0.0
    phase: str = "sitting"
    initial_seated: bool = False
    rise_started_at: float | None = None
    standing_since: float | None = None
    seated_again_at: float | None = None
    completed_cycles: int = 0
    cycle_stood: bool = False
    cycle_times: list[float] = field(default_factory=list)
    failed_rises: int = 0
    transition_peak_angle: float = 0.0
    start_position: np.ndarray | None = None
    start_height: float = 0.0
    max_displacement: float = 0.0
    turn_time: float | None = None


class FunctionalTestAgent(AreaAgent):
    """Times sit-stand-walk tasks from knee angles and body displacement."""

    name = "functional_test"
    area = "posture"
    required_trackers = frozenset({TrackerName.POSE})
    supported_events = frozenset({"repetitions"})
    supported_states = frozenset({"seated_after_walk"})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._states: dict[str, FunctionalState] = {}

    def _test_for(self, step_id: str) -> str | None:
        if step_id in [str(step) for step in self.threshold_list("tug_steps")]:
            return "tug"
        if step_id in [str(step) for step in self.threshold_list("sts_steps")]:
            return "sts"
        return None

    def on_step_start(self, step_instance: str, timestamp: float) -> None:
        """Reset the state machine at the start (or repeat) of a functional test step."""
        super().on_step_start(step_instance, timestamp)
        test = self._test_for(base_step_id(step_instance))
        if test is not None:
            self._states[step_instance] = FunctionalState(test=test)

    def _knee_angle(self, pose: PoseResult, minimum_score: float) -> float:
        angles = [
            joint_angle_degrees(pose.keypoint(f"{side}_hip"), pose.keypoint(f"{side}_knee"), pose.keypoint(f"{side}_ankle"))
            for side in ("right", "left")
            if pose.all_visible((f"{side}_hip", f"{side}_knee", f"{side}_ankle"), minimum_score)
        ]
        finite = [angle for angle in angles if np.isfinite(angle)]
        return float(np.mean(finite)) if finite else float("nan")

    def _classify(self, knee_angle: float) -> str:
        if knee_angle <= self.threshold_float("seated_knee_angle_max"):
            return "seated"
        if knee_angle >= self.threshold_float("standing_knee_angle_min"):
            return "standing"
        return "transition"

    def _update_posture(self, state: FunctionalState, observed: str, timestamp: float) -> bool:
        if observed != state.candidate:
            state.candidate = observed
            state.candidate_since = timestamp
            return False
        confirm = self.threshold_float("seated_confirm_seconds") if observed == "seated" else self.threshold_float("standing_confirm_seconds")
        if observed != state.posture and timestamp - state.candidate_since >= confirm:
            state.posture = observed
            return True
        return False

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Advance the state machine with the current posture."""
        pose = frame_context.pose
        step_instance = frame_context.step_instance or ""
        state = self._states.get(step_instance)
        if pose is None or state is None:
            return []
        minimum_score = self.threshold_float("min_keypoint_score")
        knee_angle = self._knee_angle(pose, minimum_score)
        if not np.isfinite(knee_angle):
            return []
        timestamp = frame_context.timestamp_s
        elapsed = self.step_elapsed(step_instance, timestamp)
        observed = self._classify(knee_angle)
        changed = self._update_posture(state, observed, timestamp)
        if state.posture == "transition" or observed == "transition":
            state.transition_peak_angle = max(state.transition_peak_angle, knee_angle)
        if pose.all_visible(("left_hip", "right_hip"), minimum_score):
            hips = midpoint(pose.keypoint("left_hip"), pose.keypoint("right_hip"))
            box = pose.body_bounding_box(minimum_score)
            height = (box[3] - box[1]) if box is not None else 0.0
            if state.start_position is None and height > 0:
                state.start_position = hips
                state.start_height = height
            elif state.start_position is not None and height > 0 and state.start_height > 0:
                displacement = abs(float(hips[0] - state.start_position[0])) / state.start_height + abs(math.log(height / state.start_height))
                if displacement > state.max_displacement:
                    state.max_displacement = displacement
                    state.turn_time = elapsed
        if changed:
            self._on_posture_change(state, elapsed)
        if state.test == "tug" and state.phase == "walking" and state.max_displacement >= self.threshold_float("walk_displacement_body_fraction"):
            if state.turn_time is not None and elapsed - state.turn_time > 0.5:
                state.phase = "returning"
        return [
            self.make_reading(frame_context, "knee_angle", knee_angle, "degrees", "Mean knee angle (180 is straight)."),
            self.make_reading(frame_context, "functional_phase", PHASE_CODES.get(state.phase, -1.0), "code", f"Current phase: {state.phase}."),
        ]

    def _on_posture_change(self, state: FunctionalState, elapsed: float) -> None:
        if state.posture == "seated":
            if not state.initial_seated and state.rise_started_at is None:
                state.initial_seated = True
                state.phase = "sitting"
            elif state.test == "sts" and state.cycle_stood:
                state.completed_cycles += 1
                state.cycle_times.append(elapsed)
                state.cycle_stood = False
                state.phase = "sitting"
            elif state.test == "tug" and state.standing_since is not None:
                if elapsed - state.standing_since >= self.threshold_float("tug_min_standing_seconds"):
                    state.seated_again_at = elapsed
                    state.phase = "seated"
            elif state.rise_started_at is not None and state.standing_since is None:
                if state.transition_peak_angle - self.threshold_float("seated_knee_angle_max") >= self.threshold_float("failed_rise_knee_angle_gain"):
                    state.failed_rises += 1
                state.rise_started_at = None
            state.transition_peak_angle = 0.0
        elif state.posture == "transition" and state.initial_seated and state.rise_started_at is None:
            state.rise_started_at = elapsed
            state.phase = "rising"
        elif state.posture == "standing" and state.initial_seated:
            if state.rise_started_at is None:
                state.rise_started_at = elapsed
            if state.test == "tug":
                if state.standing_since is None:
                    state.standing_since = elapsed
                state.phase = "walking"
            else:
                state.cycle_stood = True
                state.phase = "standing"

    def count_events(self, step_instance: str, event: str) -> int:
        """Completed sit-to-stand cycles."""
        if event != "repetitions":
            return 0
        state = self._states.get(step_instance)
        return state.completed_cycles if state is not None else 0

    def state_reached(self, step_instance: str, state_name: str) -> bool:
        """Whether the patient is seated again after walking (Timed Up and Go)."""
        if state_name != "seated_after_walk":
            return False
        state = self._states.get(step_instance)
        return state is not None and state.seated_again_at is not None

    def insufficiency_reason(self, step_instance: str) -> str | None:
        """Explain which part of the functional test was not detected."""
        state = self._states.get(step_instance)
        if state is not None and not state.initial_seated:
            return "A seated starting position was not detected; start fully seated with your knees and feet visible."
        return super().insufficiency_reason(step_instance)

    def finalize(self) -> AgentResult:
        """Report Timed Up and Go and five-times sit-to-stand results."""
        if not self._states:
            return self.insufficient_result("The functional tests were not performed.")
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        required_repetitions = self.threshold_int("sts_repetitions")
        for step_instance, state in self._states.items():
            confidence = self.data_confidence([step_instance])
            if state.test == "tug":
                if state.seated_again_at is None:
                    limitations.append(f"The Timed Up and Go return to sitting was not detected during {step_instance}.")
                    continue
                tug_seconds = state.seated_again_at
                rise_seconds = state.rise_started_at if state.rise_started_at is not None else float("nan")
                metrics.append(self.summary_reading("timed_up_and_go", tug_seconds, "s", "Time from 'go' until seated again.", confidence, step_instance))
                metrics.append(self.summary_reading("tug_rise_start", rise_seconds, "s", "Time from 'go' until starting to rise.", confidence, step_instance))
                metrics.append(self.summary_reading("tug_turn_time", state.turn_time, "s", "Time of the furthest point from the chair.", confidence * 0.7, step_instance))
                threshold = self.threshold_float("tug_flag_seconds")
                if tug_seconds > threshold:
                    flags.append(self.make_flag("slow_timed_up_and_go", "Slow Timed Up and Go", FlagSeverity.MODERATE, self.exceedance_confidence(tug_seconds, threshold, confidence), f"The Timed Up and Go took {tug_seconds:.1f} seconds, above the educational reference of {threshold:.0f} seconds, which is linked to higher fall risk.", [step_instance], evidence=[f"{tug_seconds:.1f} s"]))
            else:
                metrics.append(self.summary_reading("sit_to_stand_repetitions", float(state.completed_cycles), "count", "Completed sit-to-stand cycles.", confidence, step_instance))
                metrics.append(self.summary_reading("failed_rise_attempts", float(state.failed_rises), "count", "Attempts to stand that did not reach full standing.", confidence, step_instance))
                if state.completed_cycles >= required_repetitions:
                    sts_seconds = state.cycle_times[required_repetitions - 1]
                    metrics.append(self.summary_reading("five_times_sit_to_stand", sts_seconds, "s", "Time from 'go' to the end of the fifth sit-to-stand.", confidence, step_instance))
                    threshold = self.threshold_float("sts_flag_seconds")
                    if sts_seconds > threshold:
                        flags.append(self.make_flag("slow_sit_to_stand", "Slow five-times sit-to-stand", FlagSeverity.MODERATE, self.exceedance_confidence(sts_seconds, threshold, confidence), f"Five sit-to-stands took {sts_seconds:.1f} seconds, above the educational reference of {threshold:.0f} seconds, suggesting reduced leg strength.", [step_instance], evidence=[f"{sts_seconds:.1f} s"]))
                else:
                    limitations.append(f"Only {state.completed_cycles} of {required_repetitions} sit-to-stand cycles were detected during {step_instance}.")
                if state.failed_rises >= 1:
                    flags.append(self.make_flag("difficulty_rising", "Difficulty rising from the chair", FlagSeverity.LOW, confidence * 0.6, "One or more attempts to stand up did not reach full standing.", [step_instance], evidence=[f"{state.failed_rises} failed attempt(s)"]))
        return self.build_result(
            description="Functional tests: " + (", ".join(flag.title for flag in flags) if flags else "completed within educational reference times") + ".",
            metrics=metrics,
            flags=flags,
            confidence=self.data_confidence(self.steps_with_data),
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._states.clear()
