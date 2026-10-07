"""MediaPipe Face Landmarker wrapper running in VIDEO mode."""

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from neural_forge.core.exceptions import ModelInferenceError, ModelNotAvailableError


@dataclass(frozen=True)
class FaceLandmarks:
    """478 face landmarks with iris points, 52 blendshapes and the head transform."""

    normalized: np.ndarray
    pixels: np.ndarray
    blendshapes: dict[str, float]
    transformation_matrix: np.ndarray | None
    image_width: int
    image_height: int
    extra: dict[str, Any] = field(default_factory=dict)

    def point(self, index: int) -> np.ndarray:
        """Return the pixel coordinates of one landmark."""
        return self.pixels[index]

    def points(self, indices: tuple[int, ...] | list[int]) -> np.ndarray:
        """Return the pixel coordinates of several landmarks."""
        return self.pixels[list(indices)]

    def blendshape(self, name: str) -> float:
        """Return a blendshape score, or 0 when absent."""
        return float(self.blendshapes.get(name, 0.0))

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) of all landmarks in pixels."""
        x_values = self.pixels[:, 0]
        y_values = self.pixels[:, 1]
        return float(x_values.min()), float(y_values.min()), float(x_values.max()), float(y_values.max())

    def head_pose_degrees(self) -> tuple[float, float, float]:
        """Return (yaw, pitch, roll) in degrees from the facial transformation matrix."""
        if self.transformation_matrix is None:
            return 0.0, 0.0, 0.0
        rotation = self.transformation_matrix[:3, :3]
        pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
        yaw = math.degrees(math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2])))
        roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        return yaw, pitch, roll


class FaceTracker:
    """A per-session VIDEO-mode face landmarker with monotonic timestamps."""

    def __init__(
        self,
        model_bytes: bytes,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, image_bgr: np.ndarray, timestamp_ms: int) -> FaceLandmarks | None:
        """Track the face in one frame; return None when no face is present."""
        safe_timestamp = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = safe_timestamp
        rgb_image = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_image)
        try:
            result = self._landmarker.detect_for_video(mp_image, safe_timestamp)
        except Exception as error:
            raise ModelInferenceError(f"Face landmarker failed: {error}") from error
        if not result.face_landmarks:
            return None
        height, width = image_bgr.shape[:2]
        landmark_list = result.face_landmarks[0]
        normalized = np.array([[point.x, point.y, point.z] for point in landmark_list], dtype=np.float64)
        pixels = np.stack([normalized[:, 0] * width, normalized[:, 1] * height], axis=1)
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            blendshapes = {category.category_name: float(category.score) for category in result.face_blendshapes[0]}
        matrix: np.ndarray | None = None
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
        return FaceLandmarks(
            normalized=normalized,
            pixels=pixels,
            blendshapes=blendshapes,
            transformation_matrix=matrix,
            image_width=width,
            image_height=height,
        )

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._landmarker.close()


class MediaPipeFaceModel:
    """Shared face landmarker model asset; creates per-session VIDEO-mode trackers."""

    def __init__(
        self,
        task_path: Path,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self._task_path = task_path
        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_bytes: bytes | None = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """Return whether the task file exists, without loading it."""
        return self._task_path.exists()

    def _load_model_bytes(self) -> bytes:
        with self._lock:
            if self._model_bytes is None:
                if not self._task_path.exists():
                    raise ModelNotAvailableError(
                        f"Face landmarker task file is missing at {self._task_path}. "
                        "Run scripts/download_models.py inside the container."
                    )
                self._model_bytes = self._task_path.read_bytes()
            return self._model_bytes

    def create_tracker(self) -> FaceTracker:
        """Create a new VIDEO-mode tracker for one session."""
        return FaceTracker(
            model_bytes=self._load_model_bytes(),
            min_detection_confidence=self._min_detection_confidence,
            min_presence_confidence=self._min_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
