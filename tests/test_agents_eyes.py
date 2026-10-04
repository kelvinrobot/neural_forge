"""Eye agent tests with synthetic landmark sequences."""

import math

import cv2
import numpy as np

from neural_forge.agents.eyes.blink_alertness import BlinkAlertnessAgent
from neural_forge.agents.eyes.eye_alignment import EyeAlignmentAgent
from neural_forge.agents.eyes.eye_appearance import EyeAppearanceAgent
from neural_forge.agents.eyes.eye_movement import EyeMovementAgent
from neural_forge.agents.eyes.eyelid import EyelidAgent
from neural_forge.services import regions
from tests.synthetic import build_face, make_context, make_frame, skin_image


def test_eyelid_detects_ptosis() -> None:
    agent = EyelidAgent(make_context("eyes"))
    agent.on_step_start("look_straight", 0.0)
    for index in range(40):
        agent.process_frame(make_frame("look_straight", index * 0.1, face=build_face(right_upper_drop=6.0)))
    flags = {flag.code: flag for flag in agent.finalize().flags}
    assert "ptosis" in flags
    assert flags["ptosis"].side == "right"


def test_normal_eyelids_are_not_flagged() -> None:
    agent = EyelidAgent(make_context("eyes"))
    agent.on_step_start("look_straight", 0.0)
    for index in range(40):
        agent.process_frame(make_frame("look_straight", index * 0.1, face=build_face()))
    assert agent.finalize().flags == []


def test_eyelid_fatigue_during_sustained_upgaze() -> None:
    agent = EyelidAgent(make_context("eyes"))
    agent.on_step_start("sustained_upgaze", 0.0)
    for index in range(160):
        timestamp = index * 0.25
        drop = 4.0 * timestamp / 40.0
        agent.process_frame(make_frame("sustained_upgaze", timestamp, face=build_face(right_upper_drop=drop, left_upper_drop=drop)))
    assert "upgaze_eyelid_fatigue" in {flag.code for flag in agent.finalize().flags}


def test_convergent_misalignment_is_flagged() -> None:
    agent = EyeAlignmentAgent(make_context("eyes"))
    agent.on_step_start("look_straight", 0.0)
    for index in range(40):
        face = build_face(right_iris_offset=(2.0, 0.0), left_iris_offset=(-2.0, 0.0))
        agent.process_frame(make_frame("look_straight", index * 0.1, face=face))
    result = agent.finalize()
    assert "strabismus_horizontal" in {flag.code for flag in result.flags}
    assert result.value is not None and result.value > 1.0


def test_conjugate_gaze_is_not_misalignment() -> None:
    agent = EyeAlignmentAgent(make_context("eyes"))
    agent.on_step_start("look_straight", 0.0)
    for index in range(40):
        face = build_face(right_iris_offset=(3.0, 0.0), left_iris_offset=(3.0, 0.0))
        agent.process_frame(make_frame("look_straight", index * 0.1, face=face))
    assert agent.finalize().flags == []


def test_gaze_evoked_nystagmus_is_detected() -> None:
    agent = EyeMovementAgent(make_context("eyes"))
    agent.on_step_start("look_left", 0.0)
    for index in range(150):
        timestamp = index / 30.0
        offset = 3.0 + 1.5 * math.sin(2.0 * math.pi * 4.0 * timestamp)
        face = build_face(right_iris_offset=(offset, 0.0), left_iris_offset=(offset, 0.0))
        agent.process_frame(make_frame("look_left", timestamp, face=face))
    result = agent.finalize()
    codes = {flag.code for flag in result.flags}
    assert "gaze_evoked_nystagmus" in codes
    frequency = next(metric for metric in result.metrics if metric.metric == "oscillation_frequency")
    assert frequency.value is not None and abs(frequency.value - 4.0) < 0.6


def test_steady_gaze_has_no_nystagmus() -> None:
    agent = EyeMovementAgent(make_context("eyes"))
    agent.on_step_start("look_straight", 0.0)
    for index in range(150):
        agent.process_frame(make_frame("look_straight", index / 30.0, face=build_face()))
    codes = {flag.code for flag in agent.finalize().flags}
    assert "nystagmus" not in codes and "gaze_evoked_nystagmus" not in codes


def test_blink_rate_and_drowsiness() -> None:
    agent = BlinkAlertnessAgent(make_context("eyes"))
    agent.on_step_start("look_straight", 0.0)
    for index in range(600):
        timestamp = index / 30.0
        phase = timestamp % 3.0
        closed = phase < 0.15 or (timestamp > 10.0 and phase < 1.5 and int(timestamp // 3.0) % 2 == 0)
        opening = 1.0 if closed else 8.0
        agent.process_frame(make_frame("look_straight", timestamp, face=build_face(right_opening=opening, left_opening=opening)))
    assert agent.count_events("look_straight", "blinks") >= 3
    result = agent.finalize()
    assert result.status == "ok"
    assert "drowsiness_signs" in {flag.code for flag in result.flags}


def test_yellow_sclera_flags_icterus() -> None:
    agent = EyeAppearanceAgent(make_context("eyes"))
    face = build_face(right_opening=12.0, left_opening=12.0)
    image = skin_image()
    for side in ("right", "left"):
        cv2.fillPoly(image, [np.round(regions.eye_polygon(face, side)).astype(np.int32)], (150, 230, 240))
        center, radius = regions.iris_geometry(face, side)
        cv2.circle(image, (int(round(center[0])), int(round(center[1]))), int(round(radius)), (60, 60, 60), -1)
    agent.on_step_start("eyes_close_up", 0.0)
    for index in range(30):
        agent.process_frame(make_frame("eyes_close_up", index * 0.1, face=face, image=image))
    result = agent.finalize()
    assert "scleral_icterus" in {flag.code for flag in result.flags}
    assert any("unavailable" in limitation for limitation in result.limitations)
