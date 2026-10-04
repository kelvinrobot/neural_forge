"""Per-frame context object shared by the pipeline, the landmark service and all agents."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neural_forge.models.mediapipe_face import FaceLandmarks
from neural_forge.models.mediapipe_hands import HandsResult
from neural_forge.models.rtmw_pose import PoseResult
from neural_forge.schemas.readings import CalibrationInfo, QualityAssessment
from neural_forge.services.colour_utils import bgr_to_lab


@dataclass
class FrameContext:
    """A decoded frame with timestamps, quality, calibration and cached tracker outputs."""

    session_id: str
    frame_index: int
    timestamp_s: float
    client_timestamp_ms: float | None
    server_receive_time: float
    image_bgr: np.ndarray
    protocol_step: str | None = None
    step_instance: str | None = None
    allow_motion: bool = False
    calibrated_bgr: np.ndarray | None = None
    lab_image: np.ndarray | None = None
    calibration: CalibrationInfo | None = None
    quality: QualityAssessment | None = None
    face: FaceLandmarks | None = None
    hands: HandsResult | None = None
    pose: PoseResult | None = None
    trackers_run: set[str] = field(default_factory=set)
    annotations: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.image_bgr.shape[1])

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.image_bgr.shape[0])

    @property
    def is_usable(self) -> bool:
        """Whether the quality agent accepted the frame for measurement."""
        return self.quality is not None and self.quality.usable

    @property
    def quality_score(self) -> float:
        """Quality score in [0, 1], or 0 when not assessed."""
        return self.quality.score if self.quality is not None else 0.0

    @property
    def calibration_confidence(self) -> float:
        """Colour calibration confidence, or a neutral 0.5 when no calibration was applied."""
        return self.calibration.confidence if self.calibration is not None else 0.5

    def colour_image(self) -> np.ndarray:
        """Return the colour-calibrated image when available, otherwise the raw image."""
        return self.calibrated_bgr if self.calibrated_bgr is not None else self.image_bgr

    def lab(self) -> np.ndarray:
        """Return the CIELAB image of the calibrated frame, computing it once."""
        if self.lab_image is None:
            self.lab_image = bgr_to_lab(self.colour_image())
        return self.lab_image
