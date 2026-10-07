"""Process-wide registry holding one lazily created instance of every model wrapper."""

import threading
from functools import lru_cache

from neural_forge.core.protocol_loader import ConfigRepository, get_config_repository
from neural_forge.core.settings import Settings, get_settings
from neural_forge.models.classifiers import (
    ClassifierRouter,
    ImageClassifier,
    TrainedHeadClassifier,
    ZeroShotClassifier,
)
from neural_forge.models.llm_clients import VLLMChatClient
from neural_forge.models.medsiglip_neural_forge import MedSigLIPEncoder
from neural_forge.models.neural_forge_face import MediaPipeFaceModel
from neural_forge.models.neural_forge_hands import MediaPipeHandModel
from neural_forge.models.neural_forge_pose import RTMWPoseEstimator
from neural_forge.models.neural_forge_detector import YOLOELesionDetector


class ModelRegistry:
    """Creates model wrappers on first use; wrappers load weights lazily on first inference."""

    def __init__(self, settings: Settings, config: ConfigRepository) -> None:
        self._settings = settings
        self._config = config
        self._lock = threading.Lock()
        self._face_model: MediaPipeFaceModel | None = None
        self._hand_model: MediaPipeHandModel | None = None
        self._pose_estimator: RTMWPoseEstimator | None = None
        self._lesion_detector: YOLOELesionDetector | None = None
        self._medsiglip: MedSigLIPEncoder | None = None
        self._classifier: ImageClassifier | None = None
        self._reasoning_client: VLLMChatClient | None = None
        self._consultant_client: VLLMChatClient | None = None

    def face_model(self) -> MediaPipeFaceModel:
        """Return the shared face landmarker model."""
        with self._lock:
            if self._face_model is None:
                self._face_model = MediaPipeFaceModel(
                    task_path=self._settings.face_landmarker_path,
                    min_detection_confidence=self._settings.face_min_detection_confidence,
                    min_presence_confidence=self._settings.face_min_presence_confidence,
                    min_tracking_confidence=self._settings.face_min_tracking_confidence,
                )
            return self._face_model

    def hand_model(self) -> MediaPipeHandModel:
        """Return the shared hand landmarker model."""
        with self._lock:
            if self._hand_model is None:
                self._hand_model = MediaPipeHandModel(
                    task_path=self._settings.hand_landmarker_path,
                    max_hands=self._settings.max_hands,
                    min_detection_confidence=self._settings.hand_min_detection_confidence,
                    min_presence_confidence=self._settings.hand_min_presence_confidence,
                    min_tracking_confidence=self._settings.hand_min_tracking_confidence,
                )
            return self._hand_model

    def pose_estimator(self) -> RTMWPoseEstimator:
        """Return the shared RTMW whole-body estimator."""
        with self._lock:
            if self._pose_estimator is None:
                self._pose_estimator = RTMWPoseEstimator(
                    mode=self._settings.rtmw_mode,
                    backend=self._settings.rtmw_backend,
                    device=self._settings.rtmw_device,
                    min_person_score=self._settings.rtmw_min_person_score,
                )
            return self._pose_estimator

    def lesion_detector(self) -> YOLOELesionDetector:
        """Return the shared YOLOE lesion detector."""
        with self._lock:
            if self._lesion_detector is None:
                prompts = self._config.zero_shot_prompts().get("lesion_detection_prompts", [])
                self._lesion_detector = YOLOELesionDetector(
                    weights_path=self._settings.yoloe_weights_path,
                    class_prompts=list(prompts),
                    confidence=self._settings.yoloe_confidence,
                    device=self._settings.yoloe_device,
                )
            return self._lesion_detector

    def medsiglip(self) -> MedSigLIPEncoder:
        """Return the shared MedSigLIP encoder."""
        with self._lock:
            if self._medsiglip is None:
                self._medsiglip = MedSigLIPEncoder(
                    model_id=self._settings.medsiglip_model_id,
                    device_preference=self._settings.medsiglip_device,
                )
            return self._medsiglip

    def classifier(self) -> ImageClassifier:
        """Return the shared classifier router (zero-shot or trained heads)."""
        encoder = self.medsiglip()
        with self._lock:
            if self._classifier is None:
                zero_shot = ZeroShotClassifier(encoder, self._config.zero_shot_prompts())
                trained = TrainedHeadClassifier(encoder, self._settings.trained_heads_dir)
                self._classifier = ClassifierRouter(zero_shot, trained, self._settings.classifier_strategy)
            return self._classifier

    def reasoning_client(self) -> VLLMChatClient:
        """Return the HTTP client for the Qwen3-VL reasoning server."""
        with self._lock:
            if self._reasoning_client is None:
                self._reasoning_client = VLLMChatClient(
                    role="reasoning",
                    base_url=self._settings.reasoning_base_url,
                    served_model_name=self._settings.reasoning_model_name,
                    timeout_seconds=self._settings.llm_request_timeout_seconds,
                    health_timeout_seconds=self._settings.llm_health_timeout_seconds,
                    default_max_tokens=self._settings.reasoning_max_tokens,
                    default_temperature=self._settings.reasoning_temperature,
                )
            return self._reasoning_client

    def consultant_client(self) -> VLLMChatClient:
        """Return the HTTP client for the MedGemma consultant server."""
        with self._lock:
            if self._consultant_client is None:
                self._consultant_client = VLLMChatClient(
                    role="consultant",
                    base_url=self._settings.consultant_base_url,
                    served_model_name=self._settings.consultant_model_name,
                    timeout_seconds=self._settings.llm_request_timeout_seconds,
                    health_timeout_seconds=self._settings.llm_health_timeout_seconds,
                    default_max_tokens=self._settings.consultant_max_tokens,
                    default_temperature=self._settings.consultant_temperature,
                )
            return self._consultant_client

    def configured_models(self) -> dict[str, str]:
        """Return the identifiers of all configured models without loading them."""
        return {
            "reasoning": self._settings.reasoning_model_id,
            "medical_consultant": self._settings.consultant_model_id,
            "image_text_encoder": self._settings.medsiglip_model_id,
            "face_landmarker": str(self._settings.face_landmarker_path.name),
            "hand_landmarker": str(self._settings.hand_landmarker_path.name),
            "whole_body_pose": f"rtmw ({self._settings.rtmw_mode})",
            "lesion_detector": self._settings.yoloe_weights_name,
        }

    async def aclose(self) -> None:
        """Close HTTP clients held by the registry."""
        if self._reasoning_client is not None:
            await self._reasoning_client.aclose()
        if self._consultant_client is not None:
            await self._consultant_client.aclose()


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """Return the process-wide model registry."""
    return ModelRegistry(get_settings(), get_config_repository())
