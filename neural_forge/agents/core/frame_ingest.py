"""Frame ingest agent: validation, timestamping, backpressure and decoding of incoming frames."""

import asyncio
import base64
import binascii
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from neural_forge.core.exceptions import FrameRejectedError
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.settings import Settings
from neural_forge.schemas.messages import FrameMessage


@dataclass(frozen=True)
class RawFrame:
    """An undecoded frame waiting in the ingest queue."""

    session_id: str
    sequence: int
    client_timestamp_ms: float
    timestamp_s: float
    image_base64: str
    received_monotonic: float


@dataclass(frozen=True)
class IngestStatistics:
    """Counters describing ingest behaviour for the quality summary."""

    received: int
    dropped: int
    rejected: int
    decoded: int
    decode_errors: int
    measured_fps: float


class FrameIngestAgent:
    """Accepts frames from the WebSocket, applies backpressure and decodes them in the worker pool."""

    def __init__(self, session_id: str, settings: Settings) -> None:
        self._session_id = session_id
        self._settings = settings
        self._queue: asyncio.Queue[RawFrame] = asyncio.Queue(maxsize=max(1, settings.frame_queue_size))
        self._recent_client_timestamps: deque[float] = deque(maxlen=60)
        self._sequence = 0
        self._received = 0
        self._dropped = 0
        self._rejected = 0
        self._decoded = 0
        self._decode_errors = 0
        self._last_client_timestamp_ms: float | None = None
        self._last_timestamp_s: float = 0.0
        self._last_received_monotonic: float | None = None

    @property
    def latest_timestamp_s(self) -> float:
        """Timeline timestamp of the most recently submitted frame."""
        return self._last_timestamp_s

    def submit(self, message: FrameMessage) -> None:
        """Validate and enqueue a frame, dropping the oldest queued frame when the pipeline is behind."""
        self._received += 1
        if message.session_id != self._session_id:
            self._rejected += 1
            raise FrameRejectedError("Frame session_id does not match this connection.")
        estimated_bytes = (len(message.image_base64) * 3) // 4
        if estimated_bytes > self._settings.max_frame_bytes:
            self._rejected += 1
            raise FrameRejectedError(
                f"Frame of about {estimated_bytes} bytes exceeds the limit of {self._settings.max_frame_bytes} bytes."
            )
        received_monotonic = time.monotonic()
        timestamp_s = self._timeline_timestamp(message.timestamp_ms, received_monotonic)
        self._sequence += 1
        raw_frame = RawFrame(
            session_id=message.session_id,
            sequence=self._sequence,
            client_timestamp_ms=message.timestamp_ms,
            timestamp_s=timestamp_s,
            image_base64=message.image_base64,
            received_monotonic=received_monotonic,
        )
        while self._queue.full():
            self._queue.get_nowait()
            self._dropped += 1
        self._queue.put_nowait(raw_frame)

    def _timeline_timestamp(self, client_timestamp_ms: float, received_monotonic: float) -> float:
        if self._last_client_timestamp_ms is None or client_timestamp_ms > self._last_client_timestamp_ms:
            timestamp_s = client_timestamp_ms / 1000.0
            self._last_client_timestamp_ms = client_timestamp_ms
            self._recent_client_timestamps.append(timestamp_s)
        else:
            elapsed = 0.001
            if self._last_received_monotonic is not None:
                elapsed = max(0.001, received_monotonic - self._last_received_monotonic)
            timestamp_s = self._last_timestamp_s + elapsed
        if timestamp_s <= self._last_timestamp_s:
            timestamp_s = self._last_timestamp_s + 0.001
        self._last_timestamp_s = timestamp_s
        self._last_received_monotonic = received_monotonic
        return timestamp_s

    async def next_frame(self, timeout: float) -> RawFrame | None:
        """Wait for the next frame up to a timeout."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def drain(self) -> int:
        """Discard all queued frames and return how many were discarded."""
        discarded = 0
        while True:
            try:
                self._queue.get_nowait()
                discarded += 1
            except asyncio.QueueEmpty:
                return discarded

    def measured_fps(self) -> float:
        """Real frame rate measured from client timestamps."""
        if len(self._recent_client_timestamps) < 2:
            return 0.0
        span = self._recent_client_timestamps[-1] - self._recent_client_timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._recent_client_timestamps) - 1) / span

    def decode(self, raw_frame: RawFrame, frame_index: int) -> FrameContext:
        """Decode a base64 JPEG into a frame context (runs in the worker pool)."""
        payload = raw_frame.image_base64
        if payload.startswith("data:"):
            separator_index = payload.find(",")
            payload = payload[separator_index + 1 :] if separator_index >= 0 else payload
        try:
            image_bytes = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as error:
            self._decode_errors += 1
            raise FrameRejectedError(f"Frame is not valid base64: {error}") from error
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            self._decode_errors += 1
            raise FrameRejectedError("Frame could not be decoded as an image.")
        height, width = image.shape[:2]
        if max(height, width) > self._settings.max_frame_dimension:
            self._decode_errors += 1
            raise FrameRejectedError(
                f"Frame of {width}x{height} exceeds the maximum dimension {self._settings.max_frame_dimension}."
            )
        if self._settings.frames_are_mirrored:
            image = cv2.flip(image, 1)
        longest_side = max(height, width)
        if longest_side > self._settings.processing_max_dimension:
            scale = self._settings.processing_max_dimension / float(longest_side)
            image = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
        self._decoded += 1
        return FrameContext(
            session_id=raw_frame.session_id,
            frame_index=frame_index,
            timestamp_s=raw_frame.timestamp_s,
            client_timestamp_ms=raw_frame.client_timestamp_ms,
            server_receive_time=raw_frame.received_monotonic,
            image_bgr=image,
        )

    def statistics(self) -> IngestStatistics:
        """Return ingest counters."""
        return IngestStatistics(
            received=self._received,
            dropped=self._dropped,
            rejected=self._rejected,
            decoded=self._decoded,
            decode_errors=self._decode_errors,
            measured_fps=self.measured_fps(),
        )
