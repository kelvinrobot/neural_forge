"""Face agent tests with synthetic landmark sequences."""

import cv2
import numpy as np

from neural_forge.agents.base import AreaAgent
from neural_forge.agents.face.face_colour import FaceColourAgent
from neural_forge.agents.face.facial_expression import FacialExpressionAgent
from neural_forge.agents.face.facial_symmetry import FacialSymmetryAgent
from neural_forge.agents.face.pain import PainAgent
from neural_forge.models.neural_forge_face import FaceLandmarks
from neural_forge.services import regions
from tests.synthetic import build_face, make_context, make_frame, skin_image


def run_sequence(agent: AreaAgent, steps: list[tuple[str, list[FaceLandmarks]]], interval: float = 0.1, image: np.ndarray | None = None) -> None:
    timestamp = 0.0
    for step_id, faces in steps:
        agent.on_step_start(step_id, timestamp)
        for face in faces:
            agent.process_frame(make_frame(step_id, timestamp, face=face, image=image))
            timestamp += interval


def symmetry_sequence(smile_right: tuple[float, float], smile_left: tuple[float, float], right_brow: float, left_brow: float) -> list[tuple[str, list[FaceLandmarks]]]:
    return [
        ("relaxed_face", [build_face()] * 12),
        ("big_smile", [build_face(right_corner=smile_right, left_corner=smile_left)] * 12),
        ("relax", [build_face()] * 6),
        ("raise_eyebrows", [build_face(right_brow=right_brow, left_brow=left_brow)] * 12),
        ("close_eyes_tightly", [build_face(right_opening=1.0, left_opening=1.0)] * 12),
        ("puff_cheeks", [build_face(right_cheek=5.0, left_cheek=5.0)] * 12),
    ]


def test_symmetric_face_has_no_flags() -> None:
    agent = FacialSymmetryAgent(make_context("face"))
    run_sequence(agent, symmetry_sequence((-8.0, -10.0), (8.0, -10.0), 6.0, 6.0))
    result = agent.finalize()
    assert result.status == "ok"
    assert result.flags == []


def test_lower_face_weakness_with_forehead_sparing() -> None:
    agent = FacialSymmetryAgent(make_context("face"))
    run_sequence(agent, symmetry_sequence((-2.0, -3.0), (8.0, -10.0), 6.0, 6.0))
    codes = {flag.code: flag for flag in agent.finalize().flags}
    assert "facial_asymmetry_forehead_sparing" in codes
    assert codes["facial_asymmetry_forehead_sparing"].side == "right"


def test_whole_side_weakness_pattern() -> None:
    agent = FacialSymmetryAgent(make_context("face"))
    run_sequence(agent, symmetry_sequence((-2.0, -3.0), (8.0, -10.0), 1.0, 6.0))
    codes = {flag.code for flag in agent.finalize().flags}
    assert "whole_side_facial_weakness" in codes
    assert "facial_asymmetry_forehead_sparing" not in codes


def test_masked_face_is_flagged() -> None:
    agent = FacialExpressionAgent(make_context("face"))
    flat = {name: 0.05 for name in agent.threshold_list("key_blendshapes")}
    face = build_face(blendshapes=flat)
    run_sequence(
        agent,
        [
            ("relaxed_face", [face] * 20),
            ("big_smile", [face] * 10),
            ("facial_expressions", [face] * 60),
            ("natural_observation", [face] * 40),
        ],
        interval=0.25,
    )
    assert "reduced_facial_expression" in {flag.code for flag in agent.finalize().flags}


def test_expressive_face_is_not_flagged() -> None:
    agent = FacialExpressionAgent(make_context("face"))
    neutral = build_face(blendshapes={"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0})
    happy = build_face(blendshapes={"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9, "cheekSquintLeft": 0.6, "cheekSquintRight": 0.6})
    surprised = build_face(blendshapes={"browInnerUp": 0.8, "browOuterUpLeft": 0.8, "browOuterUpRight": 0.8, "eyeWideLeft": 0.7, "eyeWideRight": 0.7, "jawOpen": 0.7})
    sad = build_face(blendshapes={"mouthFrownLeft": 0.7, "mouthFrownRight": 0.7, "browInnerUp": 0.6})
    blink = build_face(blendshapes={"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9})
    observation = ([neutral] * 7 + [blink]) * 6
    run_sequence(
        agent,
        [("facial_expressions", [happy] * 20 + [surprised] * 20 + [sad] * 20), ("natural_observation", observation)],
        interval=0.25,
    )
    result = agent.finalize()
    assert "reduced_facial_expression" not in {flag.code for flag in result.flags}
    achieved = next(metric for metric in result.metrics if metric.metric == "expressions_achieved")
    assert achieved.value == 3.0


def test_pain_expression_is_flagged() -> None:
    agent = PainAgent(make_context("face"))
    grimace = build_face(blendshapes={"browDownLeft": 0.8, "browDownRight": 0.8, "eyeSquintLeft": 0.8, "eyeSquintRight": 0.8, "noseSneerLeft": 0.7, "noseSneerRight": 0.7})
    run_sequence(agent, [("relaxed_face", [grimace] * 20), ("natural_observation", [grimace] * 20)])
    result = agent.finalize()
    assert result.value is not None and result.value > 8.0
    assert "pain_expression" in {flag.code for flag in result.flags}


def face_image_with_lips(face: FaceLandmarks, lip_colour: tuple[int, int, int]) -> np.ndarray:
    image = skin_image()
    cv2.fillPoly(image, [np.round(face.points(regions.LIPS_OUTER)).astype(np.int32)], lip_colour)
    return image


def test_bluish_lips_flag_cyanosis() -> None:
    agent = FaceColourAgent(make_context("face"))
    face = build_face()
    run_sequence(agent, [("relaxed_face", [face] * 15)], image=face_image_with_lips(face, (170, 120, 140)))
    assert "central_cyanosis_lips" in {flag.code for flag in agent.finalize().flags}


def test_red_lips_have_no_colour_flag() -> None:
    agent = FaceColourAgent(make_context("face"))
    face = build_face()
    run_sequence(agent, [("relaxed_face", [face] * 15)], image=face_image_with_lips(face, (90, 80, 200)))
    assert agent.finalize().flags == []
