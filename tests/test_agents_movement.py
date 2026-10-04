"""Movement agent tests with synthetic landmark sequences and real-style timestamps."""

import math

import numpy as np

from neural_forge.agents.movement.arm_drift import ArmDriftAgent
from neural_forge.agents.movement.coordination import CoordinationAgent, palm_orientation
from neural_forge.agents.movement.finger_tapping import FingerTappingAgent
from neural_forge.agents.movement.involuntary_movement import InvoluntaryMovementAgent
from neural_forge.agents.movement.tremor import TremorAgent
from tests.synthetic import build_hand, build_pose, make_context, make_frame


def jittered_times(duration: float, rate: float, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    intervals = (1.0 / rate) * (1.0 + 0.2 * (generator.random(int(duration * rate)) - 0.5))
    return np.cumsum(intervals)


def test_rest_tremor_in_parkinsonian_band() -> None:
    agent = TremorAgent(make_context("movement"))
    agent.on_step_start("hands_at_rest", 0.0)
    for timestamp in jittered_times(10.0, 30.0):
        offset = (3.0 * math.sin(2.0 * math.pi * 5.0 * timestamp), 0.0)
        hand = build_hand("right", offset=offset)
        agent.process_frame(make_frame("hands_at_rest", float(timestamp), hands=(hand,)))
    result = agent.finalize()
    assert "parkinsonian_range_tremor" in {flag.code for flag in result.flags}
    frequency = next(metric for metric in result.metrics if metric.metric == "tremor_frequency")
    assert frequency.value is not None and abs(frequency.value - 5.0) < 0.6


def test_still_hands_have_no_tremor() -> None:
    agent = TremorAgent(make_context("movement"))
    agent.on_step_start("hands_at_rest", 0.0)
    for timestamp in jittered_times(10.0, 30.0):
        agent.process_frame(make_frame("hands_at_rest", float(timestamp), hands=(build_hand("right"),)))
    assert agent.finalize().flags == []


def test_finger_tapping_counts_taps_and_detects_decrement() -> None:
    agent = FingerTappingAgent(make_context("movement"))
    agent.on_step_start("finger_tapping_right", 0.0)
    for timestamp in jittered_times(10.0, 30.0):
        scale = 1.0 - 0.6 * timestamp / 10.0
        aperture = 0.05 + 0.5 * scale * (0.5 + 0.5 * math.sin(2.0 * math.pi * 3.0 * timestamp))
        agent.process_frame(make_frame("finger_tapping_right", float(timestamp), hands=(build_hand("right", aperture=aperture),)))
    assert agent.count_events("finger_tapping_right", "taps") >= 20
    result = agent.finalize()
    assert "finger_tapping_bradykinesia" in {flag.code for flag in result.flags}


def test_finger_tapping_ignores_other_hand() -> None:
    agent = FingerTappingAgent(make_context("movement"))
    agent.on_step_start("finger_tapping_right", 0.0)
    readings = agent.process_frame(make_frame("finger_tapping_right", 0.0, hands=(build_hand("left", aperture=0.3),)))
    assert readings == []


def test_slow_hand_flipping_is_flagged() -> None:
    agent = CoordinationAgent(make_context("movement"))
    agent.on_step_start("hand_flipping_right", 0.0)
    for timestamp in jittered_times(12.0, 30.0):
        mirrored = math.sin(2.0 * math.pi * 0.5 * timestamp) < 0
        agent.process_frame(make_frame("hand_flipping_right", float(timestamp), hands=(build_hand("right", mirrored=mirrored),)))
    result = agent.finalize()
    assert "dysdiadochokinesia" in {flag.code for flag in result.flags}


def test_palm_orientation_flips_sign() -> None:
    assert palm_orientation(build_hand("right")) > 0.3
    assert palm_orientation(build_hand("right", mirrored=True)) < -0.3


def test_one_sided_arm_drift() -> None:
    agent = ArmDriftAgent(make_context("movement"))
    agent.on_step_start("arm_drift", 0.0)
    for timestamp in np.arange(0.0, 20.0, 0.1):
        drift = 25.0 * timestamp / 20.0
        pose = build_pose({"right_wrist": (200.0, 140.0 + drift), "left_wrist": (440.0, 140.0)})
        agent.process_frame(make_frame("arm_drift", float(timestamp), pose=pose))
    flags = {flag.code: flag for flag in agent.finalize().flags}
    assert "arm_drift" in flags
    assert flags["arm_drift"].side == "right"


def test_rhythmic_limb_jerking_raises_seizure_flag() -> None:
    agent = InvoluntaryMovementAgent(make_context("movement"))
    agent.on_step_start("quiet_sitting", 0.0)
    alerts = 0
    for timestamp in jittered_times(8.0, 30.0):
        jerk = 40.0 * math.sin(2.0 * math.pi * 3.0 * timestamp)
        pose = build_pose(
            {
                "right_wrist": (255.0, 280.0 + jerk),
                "left_wrist": (385.0, 280.0 + jerk),
                "right_elbow": (260.0, 210.0 + 0.6 * jerk),
                "left_elbow": (380.0, 210.0 + 0.6 * jerk),
            }
        )
        readings = agent.process_frame(make_frame("quiet_sitting", float(timestamp), pose=pose))
        alerts += sum(1 for reading in readings if reading.metric == "seizure_like_movement_alert")
    assert alerts >= 1
    flags = {flag.code: flag for flag in agent.finalize().flags}
    assert "seizure_like_movement" in flags


def test_quiet_sitting_without_movement_has_no_flags() -> None:
    agent = InvoluntaryMovementAgent(make_context("movement"))
    agent.on_step_start("quiet_sitting", 0.0)
    for timestamp in jittered_times(8.0, 30.0):
        agent.process_frame(make_frame("quiet_sitting", float(timestamp), pose=build_pose()))
    assert agent.finalize().flags == []
