"""RTMW whole-body pose estimation through rtmlib and onnxruntime."""

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_forge.core.exceptions import ModelInferenceError, ModelNotAvailableError

BODY_KEYPOINT_INDEX: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "left_big_toe": 17,
    "left_small_toe": 18,
    "left_heel": 19,
    "right_big_toe": 20,
    "right_small_toe": 21,
    "right_heel": 22,
}

LEFT_HAND_OFFSET = 91
RIGHT_HAND_OFFSET = 112
WHOLEBODY_KEYPOINT_COUNT = 133


@dataclass(frozen=True)
class PoseResult:
    """133 COCO-WholeBody keypoints of the main person in a frame."""

    keypoints: np.ndarray
    scores: np.ndarray
    image_width: int
    image_height: int
    person_score: float

    def keypoint(self, name: str) -> np.ndarray:
        """Return pixel coordinates of a named body keypoint."""
        return self.keypoints[BODY_KEYPOINT_INDEX[name]]

    def score(self, name: str) -> float:
        """Return the confidence of a named body keypoint."""
        return float(self.scores[BODY_KEYPOINT_INDEX[name]])

    def is_visible(self, name: str, min_score: float) -> bool:
        """Return whether a named keypoint is confidently detected."""
        return self.score(name) >= min_score

    def all_visible(self, names: list[str] | tuple[str, ...], min_score: float) -> bool:
        """Return whether all named keypoints are confidently detected."""
        return all(self.is_visible(name, min_score) for name in names)

    def hand_keypoint(self, side: str, index: int) -> np.ndarray:
        """Return a hand keypoint (MediaPipe-style index 0-20) for the given side."""
        offset = LEFT_HAND_OFFSET if side == "left" else RIGHT_HAND_OFFSET
        return self.keypoints[offset + index]

    def hand_keypoint_score(self, side: str, index: int) -> float:
        """Return the confidence of a hand keypoint for the given side."""
        offset = LEFT_HAND_OFFSET if side == "left" else RIGHT_HAND_OFFSET
        return float(self.scores[offset + index])

    def body_bounding_box(self, min_score: float) -> tuple[float, float, float, float] | None:
        """Return the bounding box of confident body keypoints."""
        body_indices = list(BODY_KEYPOINT_INDEX.values())
        visible = [index for index in body_indices if self.scores[index] >= min_score]
        if len(visible) < 2:
            return None
        points = self.keypoints[visible]
        return (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )


class RTMWPoseEstimator:
    """Lazily loaded, thread-safe RTMW whole-body estimator shared across sessions."""

    def __init__(
        self,
        mode: str,
        backend: str,
        device: str,
        min_person_score: float,
        detector_model_path: Path | None = None,
        pose_model_path: Path | None = None,
    ) -> None:
        self._mode = mode
        self._backend = backend
        self._device = device
        self._min_person_score = min_person_score
        self._detector_model_path = detector_model_path
        self._pose_model_path = pose_model_path
        self._model: object | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from rtmlib import Wholebody
            except ImportError as error:
                raise ModelNotAvailableError(f"rtmlib is not installed: {error}") from error
            keyword_arguments: dict[str, object] = {
                "mode": self._mode,
                "backend": self._backend,
                "device": self._device,
                "to_openpose": False,
            }
            if self._detector_model_path is not None and self._pose_model_path is not None:
                if not self._detector_model_path.exists() or not self._pose_model_path.exists():
                    raise ModelNotAvailableError(
                        f"RTMW ONNX files are missing at {self._detector_model_path} or {self._pose_model_path}"
                    )
                keyword_arguments["det"] = str(self._detector_model_path)
                keyword_arguments["pose"] = str(self._pose_model_path)
            try:
                self._model = Wholebody(**keyword_arguments)
            except Exception as error:
                raise ModelNotAvailableError(
                    f"RTMW whole-body model could not be loaded ({error}). "
                    "Run scripts/download_models.py inside the container."
                ) from error
            return self._model

    def estimate(self, image_bgr: np.ndarray) -> PoseResult | None:
        """Return keypoints of the most prominent person, or None when nobody is found."""
        model = self._ensure_loaded()
        with self._lock:
            try:
                keypoints, scores = model(image_bgr)  # type: ignore[operator]
            except Exception as error:
                raise ModelInferenceError(f"RTMW inference failed: {error}") from error
        keypoints_array = np.asarray(keypoints, dtype=np.float64)
        scores_array = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
        if keypoints_array.ndim != 3 or keypoints_array.shape[0] == 0:
            return None
        if keypoints_array.shape[1] < WHOLEBODY_KEYPOINT_COUNT:
            return None
        best_index = -1
        best_rank = -1.0
        for person_index in range(keypoints_array.shape[0]):
            body_scores = scores_array[person_index, :23]
            mean_score = float(body_scores.mean())
            if mean_score < self._min_person_score:
                continue
            body_points = keypoints_array[person_index, :23]
            visible_points = body_points[body_scores >= self._min_person_score]
            if len(visible_points) < 2:
                continue
            extent = float(np.ptp(visible_points[:, 0]) * np.ptp(visible_points[:, 1]))
            rank = extent * mean_score
            if rank > best_rank:
                best_rank = rank
                best_index = person_index
        if best_index < 0:
            return None
        height, width = image_bgr.shape[:2]
        return PoseResult(
            keypoints=keypoints_array[best_index],
            scores=scores_array[best_index],
            image_width=width,
            image_height=height,
            person_score=float(scores_array[best_index, :23].mean()),
        )
