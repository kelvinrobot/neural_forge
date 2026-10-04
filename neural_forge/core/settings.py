"""Application settings loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the Neural Forge backend."""

    model_config = SettingsConfigDict(
        env_prefix="NF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "Neural Forge - AI Visual Diagnostic"
    app_version: str = "1.0.0"
    environment: Literal["development", "production", "test"] = "production"
    log_level: str = "INFO"
    log_json: bool = True

    host: str = "0.0.0.0"
    port: int = 7860
    forwarded_allow_ips: str = "*"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    config_dir: Path = PROJECT_ROOT / "config"
    prompts_dir: Path = PROJECT_ROOT / "neural_forge" / "prompts"
    model_cache_dir: Path = PROJECT_ROOT / "model_store"
    trained_heads_dir: Path = PROJECT_ROOT / "model_store" / "trained_heads"
    crops_dir: Path = PROJECT_ROOT / "recordings" / "crops"
    save_report_crops: bool = False

    face_landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    hand_landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    face_min_detection_confidence: float = 0.5
    face_min_presence_confidence: float = 0.5
    face_min_tracking_confidence: float = 0.5
    hand_min_detection_confidence: float = 0.5
    hand_min_presence_confidence: float = 0.5
    hand_min_tracking_confidence: float = 0.5
    max_hands: int = 2

    rtmw_mode: Literal["performance", "balanced", "lightweight"] = "balanced"
    rtmw_backend: Literal["onnxruntime", "opencv", "openvino"] = "onnxruntime"
    rtmw_device: str = "cuda"
    rtmw_min_person_score: float = 0.3

    yoloe_weights_name: str = "yoloe-11s-seg.pt"
    yoloe_confidence: float = 0.25
    yoloe_device: str = "cuda"

    medsiglip_model_id: str = "google/medsiglip-448"
    medsiglip_device: str = "cuda"
    classifier_strategy: Literal["zero_shot", "trained_head", "auto"] = "auto"

    reasoning_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    reasoning_model_name: str = "qwen3-vl-4b"
    reasoning_base_url: str = "http://127.0.0.1:8001"
    consultant_model_id: str = "google/medgemma-1.5-4b-it"
    consultant_model_name: str = "medgemma-1.5-4b"
    consultant_base_url: str = "http://127.0.0.1:8002"
    llm_request_timeout_seconds: float = 120.0
    llm_health_timeout_seconds: float = 3.0
    reasoning_max_tokens: int = 1800
    reasoning_temperature: float = 0.2
    consultant_max_tokens: int = 400
    consultant_temperature: float = 0.1
    json_repair_attempts: int = 2
    use_guided_json: bool = True
    max_consultations: int = 3
    reasoning_attach_image: bool = True

    executor_max_workers: int = 4
    max_frame_bytes: int = 1_500_000
    max_frame_dimension: int = 1920
    processing_max_dimension: int = 960
    frame_queue_size: int = 2
    frames_are_mirrored: bool = False
    report_crop_jpeg_quality: int = 90

    ws_ping_interval_seconds: float = 20.0
    ws_ping_timeout_seconds: float = 20.0
    ws_keepalive_interval_seconds: float = 15.0
    ws_idle_timeout_seconds: float = 120.0
    start_timeout_seconds: float = 600.0
    ready_timeout_seconds: float = 300.0
    quality_wait_timeout_seconds: float = 90.0
    frame_starvation_seconds: float = 10.0
    live_reading_interval_seconds: float = 1.0
    progress_interval_seconds: float = 1.0
    quality_feedback_interval_seconds: float = 1.0
    repeatable_offer_timeout_seconds: float = 20.0

    session_ttl_seconds: int = 3600
    max_sessions: int = 200
    session_purge_interval_seconds: float = 300.0
    max_question_length: int = 1000

    @property
    def face_landmarker_path(self) -> Path:
        """Location of the MediaPipe face landmarker task file."""
        return self.model_cache_dir / "mediapipe" / "face_landmarker.task"

    @property
    def hand_landmarker_path(self) -> Path:
        """Location of the MediaPipe hand landmarker task file."""
        return self.model_cache_dir / "mediapipe" / "hand_landmarker.task"

    @property
    def yoloe_weights_path(self) -> Path:
        """Location of the YOLOE detector weights."""
        return self.model_cache_dir / "yoloe" / self.yoloe_weights_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
