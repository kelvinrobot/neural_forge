"""Shared landmark service: runs each tracker at most once per frame and caches results on the frame."""

from enum import Enum
from typing import Iterable

from neural_forge.core.frame_context import FrameContext
from neural_forge.models.mediapipe_face import FaceTracker
from neural_forge.models.mediapipe_hands import HandTracker
from neural_forge.models.registry import ModelRegistry


class TrackerName(str, Enum):
    """Landmark trackers available to agents."""

    FACE = "face"
    HANDS = "hands"
    POSE = "pose"


class LandmarkService:
    """Per-session owner of VIDEO-mode trackers; shared models come from the registry."""

    def __init__(self, registry: ModelRegistry, enabled_trackers: frozenset[TrackerName]) -> None:
        self._registry = registry
        self._enabled_trackers = enabled_trackers
        self._face_tracker: FaceTracker | None = None
        self._hand_tracker: HandTracker | None = None

    @property
    def enabled_trackers(self) -> frozenset[TrackerName]:
        """Trackers this session is allowed to run."""
        return self._enabled_trackers

    def run(self, frame: FrameContext, trackers: Iterable[TrackerName]) -> None:
        """Run the requested trackers that are enabled and have not yet run on this frame."""
        timestamp_ms = int(round(frame.timestamp_s * 1000.0))
        for tracker in trackers:
            if tracker not in self._enabled_trackers or tracker.value in frame.trackers_run:
                continue
            if tracker == TrackerName.FACE:
                if self._face_tracker is None:
                    self._face_tracker = self._registry.face_model().create_tracker()
                frame.face = self._face_tracker.detect(frame.image_bgr, timestamp_ms)
            elif tracker == TrackerName.HANDS:
                if self._hand_tracker is None:
                    self._hand_tracker = self._registry.hand_model().create_tracker()
                frame.hands = self._hand_tracker.detect(frame.image_bgr, timestamp_ms)
            elif tracker == TrackerName.POSE:
                frame.pose = self._registry.pose_estimator().estimate(frame.image_bgr)
            frame.trackers_run.add(tracker.value)

    def close(self) -> None:
        """Release per-session tracker graphs."""
        if self._face_tracker is not None:
            self._face_tracker.close()
            self._face_tracker = None
        if self._hand_tracker is not None:
            self._hand_tracker.close()
            self._hand_tracker = None
