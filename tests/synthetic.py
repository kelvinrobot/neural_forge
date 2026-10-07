"""Synthetic landmark, pose, frame and agent-context builders for agent tests."""

import math
from pathlib import Path
from typing import Any

import numpy as np

from neural_forge.agents.base import AgentContext
from neural_forge.core.exceptions import ModelNotAvailableError
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import ConfigRepository
from neural_forge.core.settings import Settings
from neural_forge.models.neural_forge_face import FaceLandmarks
from neural_forge.models.neural_forge_hands import HandLandmarks, HandsResult
from neural_forge.models.neural_forge_pose import BODY_KEYPOINT_INDEX, PoseResult
from neural_forge.schemas.readings import QualityAssessment, QualityFlag
from neural_forge.services import regions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIDTH = 640
HEIGHT = 480
SKIN_BGR = (120, 150, 200)


class FakeRegistry:
    """Model registry stand-in whose models are unavailable, exercising fallback paths."""

    def classifier(self) -> Any:
        """Classification is unavailable in unit tests."""
        raise ModelNotAvailableError("classifier not available in unit tests")

    def lesion_detector(self) -> Any:
        """Detection is unavailable in unit tests."""
        raise ModelNotAvailableError("detector not available in unit tests")


def config_repository() -> ConfigRepository:
    """Configuration repository reading the project's config directory."""
    return ConfigRepository(PROJECT_ROOT / "config")


def make_context(area: str, registry: Any | None = None) -> AgentContext:
    """Agent context for an area using real protocols and thresholds."""
    config = config_repository()
    return AgentContext(
        area=area,
        protocol=config.protocol(area),
        thresholds=config.thresholds(area),
        zero_shot_prompts=config.zero_shot_prompts(),
        registry=registry or FakeRegistry(),  # type: ignore[arg-type]
        settings=Settings(),
    )


def _place_ellipse(points: np.ndarray, indices: tuple[int, ...], center: tuple[float, float], radius_x: float, radius_y: float) -> None:
    count = len(indices)
    for position, index in enumerate(indices):
        theta = math.pi + 2.0 * math.pi * position / count
        points[index] = (center[0] + radius_x * math.cos(theta), center[1] - radius_y * math.sin(theta))


def _place_eye(
    points: np.ndarray,
    side: str,
    center: tuple[float, float],
    half_width: float,
    opening: float,
    upper_drop: float,
    iris_offset: tuple[float, float],
) -> None:
    direction = -1.0 if side == "right" else 1.0
    for contour, sign in ((regions.landmark_indices(side, "upper_lid"), -1.0), (regions.landmark_indices(side, "lower_lid"), 1.0)):
        count = len(contour)
        for position, index in enumerate(contour):
            fraction = position / (count - 1)
            x_coordinate = center[0] + direction * half_width * (1.0 - 2.0 * fraction)
            bulge = math.sin(math.pi * fraction)
            y_coordinate = center[1] + sign * opening * bulge
            if sign < 0:
                y_coordinate += upper_drop * bulge
            points[index] = (x_coordinate, y_coordinate)
    iris_center = (center[0] + iris_offset[0], center[1] + iris_offset[1])
    radius = 0.39 * half_width
    ring = regions.landmark_indices(side, "iris_ring")
    points[regions.landmark_index(side, "iris_center")] = iris_center
    points[ring[0]] = (iris_center[0] + radius, iris_center[1])
    points[ring[1]] = (iris_center[0], iris_center[1] - radius)
    points[ring[2]] = (iris_center[0] - radius, iris_center[1])
    points[ring[3]] = (iris_center[0], iris_center[1] + radius)


def build_face(
    right_corner: tuple[float, float] = (0.0, 0.0),
    left_corner: tuple[float, float] = (0.0, 0.0),
    right_brow: float = 0.0,
    left_brow: float = 0.0,
    right_opening: float = 8.0,
    left_opening: float = 8.0,
    right_upper_drop: float = 0.0,
    left_upper_drop: float = 0.0,
    right_iris_offset: tuple[float, float] = (0.0, 0.0),
    left_iris_offset: tuple[float, float] = (0.0, 0.0),
    right_cheek: float = 0.0,
    left_cheek: float = 0.0,
    mouth_open: float = 0.0,
    chin_shift: float = 0.0,
    blendshapes: dict[str, float] | None = None,
) -> FaceLandmarks:
    """Frontal synthetic face; the patient's right side is on the image left."""
    points = np.tile(np.array([[320.0, 240.0]]), (478, 1))
    _place_ellipse(points, regions.FACE_OVAL, (320.0, 240.0), 110.0, 130.0)
    _place_eye(points, "right", (280.0, 200.0), 20.0, right_opening, right_upper_drop, right_iris_offset)
    _place_eye(points, "left", (360.0, 200.0), 20.0, left_opening, left_upper_drop, left_iris_offset)
    for position, index in enumerate(regions.RIGHT_BROW):
        points[index] = (255.0 + 12.5 * position, 175.0 - right_brow)
    for position, index in enumerate(regions.LEFT_BROW):
        points[index] = (335.0 + 12.5 * position, 175.0 - left_brow)
    midline_y = (110.0, 130.0, 160.0, 180.0, 200.0, 212.0, 224.0, 236.0, 246.0, 254.0, 262.0)
    for index, y_coordinate in zip(regions.FACIAL_MIDLINE_UPPER, midline_y):
        points[index] = (320.0, y_coordinate)
    points[regions.RIGHT_ALAR_BASE] = (305.0, 262.0)
    points[regions.LEFT_ALAR_BASE] = (335.0, 262.0)
    _place_ellipse(points, regions.LIPS_OUTER, (320.0, 300.0), 35.0, 12.0 + mouth_open)
    _place_ellipse(points, regions.LIPS_INNER, (320.0, 300.0), 25.0, 1.5 + mouth_open)
    for index in (regions.MOUTH_RIGHT_CORNER, 78):
        points[index] = points[index] + np.array(right_corner)
    for index in (regions.MOUTH_LEFT_CORNER, 308):
        points[index] = points[index] + np.array(left_corner)
    points[regions.FACE_CHIN] = (320.0 + chin_shift, 370.0 + mouth_open)
    points[regions.RIGHT_FACE_CONTOUR] = (210.0 - right_cheek, 260.0)
    points[regions.LEFT_FACE_CONTOUR] = (430.0 + left_cheek, 260.0)
    for indices, center in ((regions.RIGHT_CHEEK_REGION, (262.0, 262.0)), (regions.LEFT_CHEEK_REGION, (378.0, 262.0))):
        for position, index in enumerate(indices):
            angle = 2.0 * math.pi * position / len(indices)
            points[index] = (center[0] + 16.0 * math.cos(angle), center[1] + 16.0 * math.sin(angle))
    normalized = np.column_stack([points[:, 0] / WIDTH, points[:, 1] / HEIGHT, np.zeros(478)])
    return FaceLandmarks(
        normalized=normalized,
        pixels=points.copy(),
        blendshapes=dict(blendshapes or {}),
        transformation_matrix=None,
        image_width=WIDTH,
        image_height=HEIGHT,
    )


def build_hand(
    side: str = "right",
    wrist: tuple[float, float] = (320.0, 400.0),
    mirrored: bool = False,
    aperture: float | None = None,
    finger_angles: dict[str, float] | None = None,
    offset: tuple[float, float] = (0.0, 0.0),
) -> HandLandmarks:
    """Synthetic upright hand with optional mirroring, thumb-index aperture and finger angles (degrees, + toward image right)."""
    angles = finger_angles or {}
    sign = -1.0 if mirrored else 1.0
    base = np.array(wrist, dtype=np.float64) + np.array(offset)
    points = np.zeros((21, 2))
    points[0] = base
    knuckles = {"index": (-30.0, -90.0), "middle": (-10.0, -100.0), "ring": (10.0, -95.0), "little": (28.0, -85.0)}
    chains = regions.FINGER_CHAINS
    for finger, (dx, dy) in knuckles.items():
        chain = chains[finger]
        knuckle = base + np.array([sign * dx, dy])
        points[chain[0]] = knuckle
        angle = math.radians(angles.get(finger, 0.0)) * sign
        direction = np.array([math.sin(angle), -math.cos(angle)])
        position = knuckle.copy()
        for joint_index, length in zip(chain[1:], (35.0, 25.0, 20.0)):
            position = position + direction * length
            points[joint_index] = position
    thumb_offsets = ((-30.0, -30.0), (-50.0, -50.0), (-62.0, -70.0), (-70.0, -88.0))
    for joint_index, (dx, dy) in zip(chains["thumb"], thumb_offsets):
        points[joint_index] = base + np.array([sign * dx, dy])
    if aperture is not None:
        palm_length = float(np.linalg.norm(points[9] - points[0]))
        points[4] = points[8] + np.array([sign * aperture * palm_length, 0.0])
    normalized = np.column_stack([points[:, 0] / WIDTH, points[:, 1] / HEIGHT, np.zeros(21)])
    return HandLandmarks(side=side, handedness_score=0.95, normalized=normalized, pixels=points, world=None)  # type: ignore[arg-type]


BASE_POSE: dict[str, tuple[float, float]] = {
    "nose": (320.0, 80.0),
    "right_eye": (310.0, 72.0),
    "left_eye": (330.0, 72.0),
    "right_ear": (298.0, 80.0),
    "left_ear": (342.0, 80.0),
    "right_shoulder": (270.0, 140.0),
    "left_shoulder": (370.0, 140.0),
    "right_elbow": (260.0, 210.0),
    "left_elbow": (380.0, 210.0),
    "right_wrist": (255.0, 280.0),
    "left_wrist": (385.0, 280.0),
    "right_hip": (290.0, 280.0),
    "left_hip": (350.0, 280.0),
    "right_knee": (288.0, 370.0),
    "left_knee": (352.0, 370.0),
    "right_ankle": (286.0, 460.0),
    "left_ankle": (354.0, 460.0),
}


def build_pose(overrides: dict[str, tuple[float, float]] | None = None, score: float = 0.9) -> PoseResult:
    """Synthetic standing frontal whole-body pose."""
    keypoints = np.zeros((133, 2))
    scores = np.zeros(133)
    merged = dict(BASE_POSE)
    merged.update(overrides or {})
    for name, point in merged.items():
        keypoints[BODY_KEYPOINT_INDEX[name]] = point
        scores[BODY_KEYPOINT_INDEX[name]] = score
    return PoseResult(keypoints=keypoints, scores=scores, image_width=WIDTH, image_height=HEIGHT, person_score=score)


def make_frame(
    step_id: str,
    timestamp: float,
    face: FaceLandmarks | None = None,
    hands: tuple[HandLandmarks, ...] | None = None,
    pose: PoseResult | None = None,
    image: np.ndarray | None = None,
    step_instance: str | None = None,
    frame_index: int = 0,
) -> FrameContext:
    """Usable frame context with optional tracker outputs."""
    frame = FrameContext(
        session_id="test",
        frame_index=frame_index,
        timestamp_s=timestamp,
        client_timestamp_ms=timestamp * 1000.0,
        server_receive_time=timestamp,
        image_bgr=image if image is not None else np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8),
        protocol_step=step_id,
        step_instance=step_instance or step_id,
    )
    frame.quality = QualityAssessment(usable=True, quality_flag=QualityFlag.GOOD, score=0.9, brightness=128.0, blur_variance=200.0, subject_present=True)
    frame.face = face
    frame.hands = HandsResult(hands=hands) if hands is not None else None
    frame.pose = pose
    return frame


def skin_image(colour: tuple[int, int, int] = SKIN_BGR) -> np.ndarray:
    """Uniform skin-coloured image."""
    return np.full((HEIGHT, WIDTH, 3), colour, dtype=np.uint8)
