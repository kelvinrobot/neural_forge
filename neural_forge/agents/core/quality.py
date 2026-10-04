"""Quality agent: exposure, sharpness, motion, subject presence, framing and distance checks."""

import math
from typing import Any

import cv2
import numpy as np

from neural_forge.core.exceptions import ConfigurationError
from neural_forge.core.frame_context import FrameContext
from neural_forge.schemas.protocols import CameraSetup, localize
from neural_forge.schemas.readings import QualityAssessment, QualityFlag
from neural_forge.services.geometry import euclidean_distance
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import (
    LEFT_EYE_OUTER,
    LEFT_IRIS_CENTER,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
    RIGHT_EYE_OUTER,
    RIGHT_IRIS_CENTER,
)

SUBJECT_TRACKERS: dict[str, frozenset[TrackerName]] = {
    "face": frozenset({TrackerName.FACE}),
    "hands": frozenset({TrackerName.HANDS}),
    "pose": frozenset({TrackerName.POSE}),
    "none": frozenset(),
}


class QualityAgent:
    """Assesses each frame against the camera setup required by the current step."""

    def __init__(self, quality_config: dict[str, Any], language: str) -> None:
        self._global: dict[str, Any] = quality_config["global"]
        self._setups: dict[str, dict[str, Any]] = quality_config["camera_setups"]
        self._guidance: dict[str, dict[str, str]] = quality_config["guidance"]
        self._language = language
        self._previous_small_grey: np.ndarray | None = None
        for setup in CameraSetup:
            if setup.value not in self._setups:
                raise ConfigurationError(f"quality.yaml is missing camera setup '{setup.value}'")
        self._non_blocking = set(self._global["non_blocking_issues"])

    @property
    def confirmation_frames(self) -> int:
        """Consecutive acceptable frames required before a step may start."""
        return int(self._global["confirmation_frames"])

    def trackers_for_setup(self, camera_setup: CameraSetup) -> frozenset[TrackerName]:
        """Trackers required to check the subject of a camera setup."""
        subject = str(self._setups[camera_setup.value]["subject"])
        if subject not in SUBJECT_TRACKERS:
            raise ConfigurationError(f"Unknown subject '{subject}' for camera setup '{camera_setup.value}'")
        return SUBJECT_TRACKERS[subject]

    def guidance_for(self, issues: list[str]) -> list[str]:
        """Localised guidance messages for issue codes."""
        messages: list[str] = []
        for issue in issues:
            text = localize(self._guidance.get(issue), self._language)
            if text and text not in messages:
                messages.append(text)
        return messages

    def rejected_assessment(self, reason: str) -> QualityAssessment:
        """Assessment for a frame that could not be decoded."""
        return QualityAssessment(
            usable=False,
            quality_flag=QualityFlag.UNUSABLE,
            score=0.0,
            issues=["frame_rejected"],
            guidance=self.guidance_for(["frame_rejected"]) or [reason],
        )

    def assess(self, frame: FrameContext, camera_setup: CameraSetup, allow_motion: bool) -> QualityAssessment:
        """Assess exposure, sharpness, motion and subject framing for one frame."""
        setup = self._setups[camera_setup.value]
        image = frame.image_bgr
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        analysis_width = int(self._global["analysis_width"])
        scale = analysis_width / float(grey.shape[1])
        analysis_grey = cv2.resize(grey, (analysis_width, max(1, int(round(grey.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        brightness = float(analysis_grey.mean())
        contrast = float(analysis_grey.std())
        blur_variance = float(cv2.Laplacian(analysis_grey, cv2.CV_64F).var())
        overexposed_fraction = float((analysis_grey >= 250).mean())
        small_grey = cv2.resize(analysis_grey, (160, max(1, int(round(analysis_grey.shape[0] * 160 / analysis_width)))))
        motion = 0.0
        if self._previous_small_grey is not None and self._previous_small_grey.shape == small_grey.shape:
            motion = float(np.abs(small_grey.astype(np.float32) - self._previous_small_grey.astype(np.float32)).mean())
        self._previous_small_grey = small_grey

        issues: list[str] = []
        if brightness < float(self._global["min_brightness"]):
            issues.append("too_dark")
        elif brightness > float(self._global["max_brightness"]):
            issues.append("too_bright")
        if overexposed_fraction > float(self._global["overexposed_fraction_max"]):
            issues.append("overexposed")
        if contrast < float(self._global["min_contrast"]):
            issues.append("low_contrast")
        minimum_blur = float(setup.get("min_blur_variance", self._global["min_blur_variance"]))
        if allow_motion:
            minimum_blur *= float(self._global["motion_step_blur_factor"])
        if blur_variance < minimum_blur:
            issues.append("blurry")
        if not allow_motion and motion > float(self._global["max_motion"]):
            issues.append("hold_still")

        subject = str(setup["subject"])
        subject_present = True
        subject_fraction = 0.0
        estimated_distance: float | None = None
        if subject == "face":
            subject_present, subject_fraction, estimated_distance = self._assess_face(frame, setup, issues, analysis_grey, scale)
        elif subject == "hands":
            subject_present, subject_fraction = self._assess_hands(frame, setup, issues)
        elif subject == "pose":
            subject_present, subject_fraction, estimated_distance = self._assess_pose(frame, setup, issues)

        blocking_issues = [issue for issue in issues if issue not in self._non_blocking]
        non_blocking_issues = [issue for issue in issues if issue in self._non_blocking]
        usable = not blocking_issues
        sharpness_factor = 0.6 + 0.4 * min(1.0, blur_variance / max(1.0, 2.0 * minimum_blur))
        score = max(0.0, 1.0 - 0.25 * len(blocking_issues) - 0.08 * len(non_blocking_issues)) * sharpness_factor
        if usable and not non_blocking_issues:
            quality_flag = QualityFlag.GOOD
        elif usable:
            quality_flag = QualityFlag.FAIR
        else:
            quality_flag = QualityFlag.POOR
        return QualityAssessment(
            usable=usable,
            quality_flag=quality_flag,
            score=float(min(1.0, max(0.0, score))),
            brightness=brightness,
            contrast=contrast,
            blur_variance=blur_variance,
            motion=motion,
            subject=subject,
            subject_present=subject_present,
            subject_fraction=subject_fraction,
            estimated_distance_m=estimated_distance,
            issues=issues,
            guidance=self.guidance_for(issues),
        )

    def _focal_length_pixels(self, image_width: int) -> float:
        field_of_view = math.radians(float(self._global["camera_horizontal_fov_degrees"]))
        return (image_width / 2.0) / math.tan(field_of_view / 2.0)

    def _check_size(self, size: float, setup: dict[str, Any], issues: list[str]) -> None:
        if size < float(setup["min_size"]):
            issues.append("move_closer")
        elif size > float(setup["max_size"]):
            issues.append("move_back")

    def _check_centre(self, center_x: float, center_y: float, width: int, height: int, issues: list[str]) -> None:
        offset_x = abs(center_x / width - 0.5)
        offset_y = abs(center_y / height - 0.5)
        if max(offset_x, offset_y) > float(self._global["max_center_offset"]):
            issues.append("center_yourself")

    def _assess_face(
        self,
        frame: FrameContext,
        setup: dict[str, Any],
        issues: list[str],
        analysis_grey: np.ndarray,
        scale: float,
    ) -> tuple[bool, float, float | None]:
        face = frame.face
        if face is None:
            issues.append("face_not_found")
            return False, 0.0, None
        x_min, y_min, x_max, y_max = face.bounding_box()
        measure = str(setup["size_measure"])
        if measure == "eye_span":
            size = euclidean_distance(face.point(RIGHT_EYE_OUTER), face.point(LEFT_EYE_OUTER)) / frame.width
        elif measure == "mouth_width":
            size = euclidean_distance(face.point(MOUTH_RIGHT_CORNER), face.point(MOUTH_LEFT_CORNER)) / frame.width
        else:
            size = (x_max - x_min) / frame.width
        self._check_size(size, setup, issues)
        self._check_centre((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, frame.width, frame.height, issues)
        face_region = analysis_grey[
            max(0, int(y_min * scale)) : max(1, int(y_max * scale)),
            max(0, int(x_min * scale)) : max(1, int(x_max * scale)),
        ]
        if face_region.size > 0:
            if float(analysis_grey.mean()) - float(face_region.mean()) > float(self._global["backlit_margin"]):
                issues.append("backlit")
        pupil_distance = euclidean_distance(face.point(RIGHT_IRIS_CENTER), face.point(LEFT_IRIS_CENTER))
        distance: float | None = None
        if pupil_distance > 1.0:
            distance = float(self._global["ipd_mm"]) * self._focal_length_pixels(frame.width) / pupil_distance / 1000.0
        return True, float(size), distance

    def _assess_hands(self, frame: FrameContext, setup: dict[str, Any], issues: list[str]) -> tuple[bool, float]:
        hands = frame.hands
        required_count = int(setup["min_count"])
        if hands is None or hands.count == 0:
            issues.append("hands_not_found")
            return False, 0.0
        if hands.count < required_count:
            issues.append("show_both_hands")
        spans: list[float] = []
        centers_x: list[float] = []
        centers_y: list[float] = []
        for hand in hands.hands:
            x_min, y_min, x_max, y_max = hand.bounding_box()
            spans.append(max((x_max - x_min) / frame.width, (y_max - y_min) / frame.height))
            centers_x.append((x_min + x_max) / 2.0)
            centers_y.append((y_min + y_max) / 2.0)
        size = min(spans) if required_count >= 2 else max(spans)
        self._check_size(size, setup, issues)
        self._check_centre(float(np.mean(centers_x)), float(np.mean(centers_y)), frame.width, frame.height, issues)
        return hands.count >= required_count, float(size)

    def _assess_pose(self, frame: FrameContext, setup: dict[str, Any], issues: list[str]) -> tuple[bool, float, float | None]:
        pose = frame.pose
        if pose is None:
            issues.append("body_not_found")
            return False, 0.0, None
        min_score = float(self._global["min_keypoint_score"])
        missing = False
        for requirement in setup["required_keypoints"]:
            alternatives = str(requirement).split("|")
            if not any(pose.is_visible(name, min_score) for name in alternatives):
                missing = True
                break
        if missing:
            issues.append(str(setup.get("missing_issue", "show_full_body")))
        box = pose.body_bounding_box(min_score)
        if box is None:
            return False, 0.0, None
        x_min, y_min, x_max, y_max = box
        measure = str(setup["size_measure"])
        if measure == "shoulder_width" and pose.is_visible("left_shoulder", min_score) and pose.is_visible("right_shoulder", min_score):
            size = euclidean_distance(pose.keypoint("left_shoulder"), pose.keypoint("right_shoulder")) / frame.width
        else:
            size = (y_max - y_min) / frame.height
        self._check_size(size, setup, issues)
        self._check_centre((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, frame.width, frame.height, issues)
        distance: float | None = None
        body_pixels = y_max - y_min
        if measure == "body_height" and body_pixels > 1.0 and not missing:
            distance = float(self._global["assumed_body_height_m"]) * self._focal_length_pixels(frame.width) / body_pixels
        return not missing, float(size), distance
