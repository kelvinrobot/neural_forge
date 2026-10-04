"""Mouth agent tests with synthetic landmarks and images."""

import cv2
import numpy as np

from neural_forge.agents.mouth.mouth_movement import MouthMovementAgent, lateral_axis
from neural_forge.agents.mouth.oral_appearance import OralAppearanceAgent
from neural_forge.services import regions
from tests.synthetic import build_face, make_context, make_frame, skin_image


def test_lateral_axis_points_to_patient_left() -> None:
    face = build_face()
    _, downward, left_normal = lateral_axis(face)
    assert downward[1] > 0.9
    assert left_normal[0] > 0.9


def test_jaw_deviation_flags_uneven_opening() -> None:
    agent = MouthMovementAgent(make_context("mouth"))
    agent.on_step_start("open_wide", 0.0)
    face = build_face(mouth_open=12.0, chin_shift=12.0)
    for index in range(25):
        agent.process_frame(make_frame("open_wide", index * 0.1, face=face))
    flags = {flag.code: flag for flag in agent.finalize().flags}
    assert "mouth_opening_asymmetry" in flags
    assert flags["mouth_opening_asymmetry"].side == "left"


def test_symmetric_opening_is_not_flagged() -> None:
    agent = MouthMovementAgent(make_context("mouth"))
    agent.on_step_start("open_wide", 0.0)
    face = build_face(mouth_open=12.0)
    for index in range(25):
        agent.process_frame(make_frame("open_wide", index * 0.1, face=face))
    assert "mouth_opening_asymmetry" not in {flag.code for flag in agent.finalize().flags}


def test_white_patches_inside_mouth_are_flagged() -> None:
    agent = OralAppearanceAgent(make_context("mouth"))
    face = build_face(mouth_open=20.0)
    image = skin_image()
    cv2.fillPoly(image, [np.round(face.points(regions.LIPS_OUTER)).astype(np.int32)], (90, 80, 200))
    cv2.fillPoly(image, [np.round(face.points(regions.LIPS_INNER)).astype(np.int32)], (240, 240, 240))
    agent.on_step_start("inside_mouth", 0.0)
    for index in range(25):
        agent.process_frame(make_frame("inside_mouth", index * 0.1, face=face, image=image))
    result = agent.finalize()
    assert "oral_white_patches" in {flag.code for flag in result.flags}
    assert any("unavailable" in limitation for limitation in result.limitations)
