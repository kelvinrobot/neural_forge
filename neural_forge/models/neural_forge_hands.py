"""MediaPipe Hand Landmarker wrapper running in VIDEO mode."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from neural_forge.core.exceptions import ModelInferenceError, ModelNotAvailableError

HandSide = Literal["left", "right"]


@dataclass(frozen=True)
class HandLandmarks:
    """21 landmarks of one hand, labelled with the patient's anatomical side."""

    side: HandSide
    handedness_score: float
    normalized: np.ndarray
    pixels: np.ndarray
    world: np.ndarray | None

    def point(self, index: int) -> np.ndarray:
        """Return the pixel coordinates of one landmark."""
        return self.pixels[index]

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) of the hand in pixels."""
        return (
            float(self.pixels[:, 0].min()),
            float(self.pixels[:, 1].min()),
            float(self.pixels[:, 0].max()),
            float(self.pixels[:, 1].max()),
        )

    def palm_length(self) -> float:
        """Distance in pixels from the wrist to the middle finger knuckle."""
        return float(np.linalg.norm(self.pixels[9] - self.pixels[0]))


@dataclass(frozen=True)
class HandsResult:
    """All hands detected in a frame."""

    hands: tuple[HandLandmarks, ...]

    def get(self, side: HandSide) -> HandLandmarks | None:
        """Return the most confident hand for the given side."""
        candidates = [hand for hand in self.hands if hand.side == side]
        if not candidates:
            return None
        return max(candidates, key=lambda hand: hand.handedness_score)

    def largest(self) -> HandLandmarks | None:
        """Return the hand with the largest bounding box."""
        if not self.hands:
            return None

        def box_area(hand: HandLandmarks) -> float:
            x_min, y_min, x_max, y_max = hand.bounding_box()
            return (x_max - x_min) * (y_max - y_min)

        return max(self.hands, key=box_area)

    @property
    def count(self) -> int:
        """Number of detected hands."""
        return len(self.hands)


class HandTracker:
    """A per-session VIDEO-mode hand landmarker with monotonic timestamps."""

    def __init__(
        self,
        model_bytes: bytes,
        max_hands: int,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, image_bgr: np.ndarray, timestamp_ms: int) -> HandsResult:
        """Track hands in one unmirrored frame and label them with the patient's side."""
        safe_timestamp = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = safe_timestamp
        rgb_image = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_image)
        try:
            result = self._landmarker.detect_for_video(mp_image, safe_timestamp)
        except Exception as error:
            raise ModelInferenceError(f"Hand landmarker failed: {error}") from error
        height, width = image_bgr.shape[:2]
        detected: list[HandLandmarks] = []
        for hand_index, landmark_list in enumerate(result.hand_landmarks or []):
            normalized = np.array([[point.x, point.y, point.z] for point in landmark_list], dtype=np.float64)
            pixels = np.stack([normalized[:, 0] * width, normalized[:, 1] * height], axis=1)
            label = "Right"
            score = 0.0
            if result.handedness and hand_index < len(result.handedness) and result.handedness[hand_index]:
                category = result.handedness[hand_index][0]
                label = category.category_name
                score = float(category.score)
            side: HandSide = "left" if label.lower() == "right" else "right"
            world: np.ndarray | None = None
            if result.hand_world_landmarks and hand_index < len(result.hand_world_landmarks):
                world = np.array(
                    [[point.x, point.y, point.z] for point in result.hand_world_landmarks[hand_index]],
                    dtype=np.float64,
                )
            detected.append(
                HandLandmarks(side=side, handedness_score=score, normalized=normalized, pixels=pixels, world=world)
            )
        return HandsResult(hands=tuple(detected))

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._landmarker.close()


class MediaPipeHandModel:
    """Shared hand landmarker model asset; creates per-session VIDEO-mode trackers."""

    def __init__(
        self,
        task_path: Path,
        max_hands: int,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self._task_path = task_path
        self._max_hands = max_hands
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
                        f"Hand landmarker task file is missing at {self._task_path}. "
                        "Run scripts/download_models.py inside the container."
                    )
                self._model_bytes = self._task_path.read_bytes()
            return self._model_bytes

    def create_tracker(self) -> HandTracker:
        """Create a new VIDEO-mode tracker for one session."""
        return HandTracker(
            model_bytes=self._load_model_bytes(),
            max_hands=self._max_hands,
            min_detection_confidence=self._min_detection_confidence,
            min_presence_confidence=self._min_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
