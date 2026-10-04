"""Hand agent tests with synthetic landmarks and images."""

import cv2
import numpy as np

from neural_forge.agents.hands.capillary_refill import CapillaryRefillAgent
from neural_forge.agents.hands.finger_shape import FingerShapeAgent
from neural_forge.agents.hands.hand_colour import HandColourAgent
from neural_forge.agents.hands.joint_appearance import JointAppearanceAgent, signed_ulnar_deviation
from neural_forge.services import regions
from tests.synthetic import build_hand, make_context, make_frame, skin_image


def test_pale_palm_is_flagged_against_forearm() -> None:
    agent = HandColourAgent(make_context("hands"))
    hand = build_hand("right")
    image = skin_image()
    cv2.fillPoly(image, [np.round(hand.pixels[list(regions.PALM_POLYGON)]).astype(np.int32)], (200, 210, 225))
    agent.on_step_start("palms", 0.0)
    for index in range(30):
        agent.process_frame(make_frame("palms", index * 0.1, hands=(hand,), image=image))
    assert "palmar_pallor" in {flag.code for flag in agent.finalize().flags}


def test_capillary_refill_time_is_measured_and_flagged() -> None:
    agent = CapillaryRefillAgent(make_context("hands"))
    key = ("right", "index")
    agent._baseline[key] = [20.0] * 15
    release_series = []
    for step_index in range(60):
        elapsed = step_index * 0.1
        value = 8.0 if elapsed < 0.3 else min(20.0, 8.0 + 12.0 * (elapsed - 0.3) / 3.2)
        release_series.append((elapsed, value))
    agent.on_step_start("capillary_release", 0.0)
    agent._release["capillary_release"][key] = release_series
    assert agent.state_reached("capillary_release", "colour_returned_to_baseline")
    result = agent.finalize()
    refill = next(metric for metric in result.metrics if metric.metric == "capillary_refill_time")
    assert refill.value is not None and 2.5 < refill.value < 3.6
    assert "prolonged_capillary_refill" in {flag.code for flag in result.flags}


def test_capillary_refill_without_blanching_is_not_timed() -> None:
    agent = CapillaryRefillAgent(make_context("hands"))
    key = ("right", "index")
    agent._baseline[key] = [20.0] * 15
    agent.on_step_start("capillary_release", 0.0)
    agent._release["capillary_release"][key] = [(index * 0.1, 19.5) for index in range(40)]
    assert not agent.state_reached("capillary_release", "colour_returned_to_baseline")
    result = agent.finalize()
    assert result.flags == []
    assert result.limitations


def test_missing_nail_window_is_one_clubbing_sign() -> None:
    agent = FingerShapeAgent(make_context("hands"))
    agent.on_step_start("nail_to_nail", 0.0)
    for index in range(20):
        agent.process_frame(make_frame("nail_to_nail", index * 0.1, hands=(build_hand("right"), build_hand("left", wrist=(420.0, 400.0)))))
    result = agent.finalize()
    assert "possible_finger_clubbing" in {flag.code for flag in result.flags}


def test_signed_ulnar_deviation_direction() -> None:
    straight = build_hand("right")
    drifted = build_hand("right", finger_angles={"index": 20.0, "middle": 20.0, "ring": 20.0, "little": 20.0})
    straight_mean = np.mean([signed_ulnar_deviation(straight, finger) for finger in ("index", "middle", "ring", "little")])
    drifted_mean = np.mean([signed_ulnar_deviation(drifted, finger) for finger in ("index", "middle", "ring", "little")])
    assert abs(straight_mean) < 5.0
    assert drifted_mean > 15.0


def test_ulnar_drift_is_flagged() -> None:
    agent = JointAppearanceAgent(make_context("hands"))
    hand = build_hand("right", finger_angles={"index": 20.0, "middle": 20.0, "ring": 20.0, "little": 20.0})
    agent.on_step_start("palms", 0.0)
    for index in range(30):
        agent.process_frame(make_frame("palms", index * 0.1, hands=(hand,)))
    flags = {flag.code: flag for flag in agent.finalize().flags}
    assert "ulnar_deviation" in flags and flags["ulnar_deviation"].side == "right"
