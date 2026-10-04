"""Ultralytics YOLOE open-vocabulary detector used as a text-prompted skin lesion finder."""

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_forge.core.exceptions import ModelInferenceError, ModelNotAvailableError


@dataclass(frozen=True)
class Detection:
    """One open-vocabulary detection."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        x_min, y_min, x_max, y_max = self.bbox
        return max(0, x_max - x_min) * max(0, y_max - y_min)


class YOLOELesionDetector:
    """Lazily loaded, thread-safe YOLOE detector with fixed text prompts."""

    def __init__(self, weights_path: Path, class_prompts: list[str], confidence: float, device: str) -> None:
        self._weights_path = weights_path
        self._class_prompts = list(class_prompts)
        self._confidence = confidence
        self._device = device
        self._model: object | None = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """Return whether the weights file exists, without loading it."""
        return self._weights_path.exists()

    def _ensure_loaded(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if not self._weights_path.exists():
                raise ModelNotAvailableError(
                    f"YOLOE weights are missing at {self._weights_path}. "
                    "Run scripts/download_models.py inside the container."
                )
            if not self._class_prompts:
                raise ModelNotAvailableError("YOLOE requires at least one text prompt")
            try:
                from ultralytics import YOLOE
            except ImportError as error:
                raise ModelNotAvailableError(f"ultralytics is not installed: {error}") from error
            try:
                model = YOLOE(str(self._weights_path))
                try:
                    text_embeddings = model.get_text_pe(self._class_prompts)
                    model.set_classes(self._class_prompts, text_embeddings)
                except TypeError:
                    model.set_classes(self._class_prompts)
            except Exception as error:
                raise ModelNotAvailableError(f"YOLOE model could not be initialised: {error}") from error
            self._model = model
            return self._model

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Run text-prompted detection on one BGR image."""
        model = self._ensure_loaded()
        with self._lock:
            try:
                results = model.predict(  # type: ignore[attr-defined]
                    image_bgr,
                    conf=self._confidence,
                    device=self._device,
                    verbose=False,
                )
            except Exception as error:
                raise ModelInferenceError(f"YOLOE inference failed: {error}") from error
        detections: list[Detection] = []
        height, width = image_bgr.shape[:2]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            coordinates = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for box, confidence, class_index in zip(coordinates, confidences, classes):
                x_min = int(max(0, min(width - 1, box[0])))
                y_min = int(max(0, min(height - 1, box[1])))
                x_max = int(max(0, min(width, box[2])))
                y_max = int(max(0, min(height, box[3])))
                if x_max <= x_min or y_max <= y_min:
                    continue
                label = (
                    self._class_prompts[class_index]
                    if 0 <= class_index < len(self._class_prompts)
                    else "skin finding"
                )
                detections.append(
                    Detection(label=label, confidence=float(confidence), bbox=(x_min, y_min, x_max, y_max))
                )
        return detections
