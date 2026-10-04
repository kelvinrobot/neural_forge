"""Posture, joint and walking agent tests with synthetic pose sequences."""

import math

import numpy as np

from neural_forge.agents.posture.balance import BalanceAgent
from neural_forge.agents.posture.functional_test import FunctionalTestAgent
from neural_forge.agents.posture.gait import GaitAgent
from neural_forge.agents.posture.range_of_motion import RangeOfMotionAgent, count_repetitions
from neural_forge.agents.posture.static_posture import StaticPostureAgent
from tests.synthetic import BASE_POSE, build_pose, make_context, make_frame


def test_uneven_shoulders_are_flagged() -> None:
    agent = StaticPostureAgent(make_context("posture"))
    agent.on_step_start("stand_front", 0.0)
    for index in range(40):
        pose = build_pose({"left_shoulder": (370.0, 150.0)})
        agent.process_frame(make_frame("stand_front", index * 0.1, pose=pose))
    flags = {flag.code: flag for flag in agent.finalize().flags}
    assert "shoulder_height_asymmetry" in flags
    assert flags["shoulder_height_asymmetry"].side == "left"


def test_count_repetitions_with_hysteresis() -> None:
    values = [10, 60, 130, 140, 40, 20, 125, 30, 125, 135, 45]
    assert count_repetitions(values, 120.0, 50.0) == 3


def arm_positions(angle_degrees: float) -> dict[str, tuple[float, float]]:
    radians = math.radians(angle_degrees)
    right_shoulder = BASE_POSE["right_shoulder"]
    left_shoulder = BASE_POSE["left_shoulder"]
    return {
        "right_wrist": (right_shoulder[0] - 140.0 * math.sin(radians), right_shoulder[1] + 140.0 * math.cos(radians)),
        "left_wrist": (left_shoulder[0] + 140.0 * math.sin(radians), left_shoulder[1] + 140.0 * math.cos(radians)),
        "right_elbow": (right_shoulder[0] - 70.0 * math.sin(radians), right_shoulder[1] + 70.0 * math.cos(radians)),
        "left_elbow": (left_shoulder[0] + 70.0 * math.sin(radians), left_shoulder[1] + 70.0 * math.cos(radians)),
    }


def test_arm_raises_are_counted_and_range_measured() -> None:
    agent = RangeOfMotionAgent(make_context("posture"))
    agent.on_step_start("arms_sideways_up", 0.0)
    timestamp = 0.0
    for _ in range(3):
        for angle in list(np.linspace(5.0, 170.0, 20)) + list(np.linspace(170.0, 5.0, 20)):
            agent.process_frame(make_frame("arms_sideways_up", timestamp, pose=build_pose(arm_positions(float(angle)))))
            timestamp += 0.1
    assert agent.count_events("arms_sideways_up", "repetitions") == 3
    result = agent.finalize()
    assert not any(flag.code.startswith("limited_shoulder") for flag in result.flags)


def test_walking_steps_and_cadence() -> None:
    agent = GaitAgent(make_context("posture"))
    agent.on_step_start("walking", 0.0)
    for timestamp in np.arange(0.0, 10.0, 1.0 / 30.0):
        swing = math.sin(2.0 * math.pi * 1.0 * timestamp)
        pose = build_pose(
            {
                "left_ankle": (320.0 + 60.0 * swing, 460.0),
                "right_ankle": (320.0 - 60.0 * swing, 460.0),
                "left_wrist": (385.0 - 30.0 * swing, 280.0),
                "right_wrist": (255.0 + 30.0 * swing, 280.0),
            }
        )
        agent.process_frame(make_frame("walking", float(timestamp), pose=pose))
    assert agent.count_events("walking", "steps") >= 12
    result = agent.finalize()
    cadence = next(metric for metric in result.metrics if metric.metric == "cadence")
    assert cadence.value is not None and abs(cadence.value - 120.0) < 15.0
    assert "shuffling_gait" not in {flag.code for flag in result.flags}


def sway_pose(offset: float) -> dict[str, tuple[float, float]]:
    moved = {}
    for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        x_coordinate, y_coordinate = BASE_POSE[name]
        moved[name] = (x_coordinate + offset, y_coordinate)
    moved["left_ankle"] = (320.0, 460.0)
    moved["right_ankle"] = (320.0, 460.0)
    return moved


def test_romberg_pattern() -> None:
    agent = BalanceAgent(make_context("posture"))
    timestamp = 0.0
    for step_id, amplitude in (("balance_eyes_open", 1.0), ("balance_eyes_closed", 8.0)):
        agent.on_step_start(step_id, timestamp)
        for index in range(150):
            offset = amplitude * math.sin(2.0 * math.pi * 0.5 * index / 15.0)
            agent.process_frame(make_frame(step_id, timestamp, pose=build_pose(sway_pose(offset))))
            timestamp += 1.0 / 15.0
    result = agent.finalize()
    assert "romberg_sign" in {flag.code for flag in result.flags}


def seated_pose() -> dict[str, tuple[float, float]]:
    return {
        "left_hip": (330.0, 300.0), "right_hip": (300.0, 300.0),
        "left_knee": (390.0, 300.0), "right_knee": (360.0, 300.0),
        "left_ankle": (390.0, 390.0), "right_ankle": (360.0, 390.0),
    }


def standing_pose() -> dict[str, tuple[float, float]]:
    return {
        "left_hip": (335.0, 200.0), "right_hip": (305.0, 200.0),
        "left_knee": (337.0, 300.0), "right_knee": (307.0, 300.0),
        "left_ankle": (337.0, 400.0), "right_ankle": (307.0, 400.0),
    }


def test_five_times_sit_to_stand_is_timed() -> None:
    agent = FunctionalTestAgent(make_context("posture"))
    agent.on_step_start("sit_to_stand", 0.0)
    timestamp = 0.0
    phases = [seated_pose()] + [standing_pose(), seated_pose()] * 5
    for phase in phases:
        for _ in range(15):
            agent.process_frame(make_frame("sit_to_stand", timestamp, pose=build_pose(phase)))
            timestamp += 1.0 / 15.0
    assert agent.count_events("sit_to_stand", "repetitions") == 5
    result = agent.finalize()
    timing = next(metric for metric in result.metrics if metric.metric == "five_times_sit_to_stand")
    assert timing.value is not None and 9.0 < timing.value < 12.5
    assert "slow_sit_to_stand" not in {flag.code for flag in result.flags}


def test_timed_up_and_go_detects_return_to_seat() -> None:
    agent = FunctionalTestAgent(make_context("posture"))
    agent.on_step_start("timed_up_and_go", 0.0)
    timestamp = 0.0
    for phase, seconds in ((seated_pose(), 1.0), (standing_pose(), 5.0), (seated_pose(), 1.5)):
        for _ in range(int(seconds * 15)):
            agent.process_frame(make_frame("timed_up_and_go", timestamp, pose=build_pose(phase)))
            timestamp += 1.0 / 15.0
    assert agent.state_reached("timed_up_and_go", "seated_after_walk")
    tug = next(metric for metric in agent.finalize().metrics if metric.metric == "timed_up_and_go")
    assert tug.value is not None and 6.0 < tug.value < 7.5
