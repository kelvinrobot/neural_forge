"""MedSigLIP-448 image and text encoder."""

import os
import threading

import cv2
import numpy as np

from neural_forge.core.exceptions import ModelInferenceError, ModelNotAvailableError


class MedSigLIPEncoder:
    """Lazily loaded MedSigLIP encoder producing L2-normalised embeddings."""

    def __init__(self, model_id: str, device_preference: str, batch_size: int = 16) -> None:
        self._model_id = model_id
        self._device_preference = device_preference
        self._batch_size = batch_size
        self._model: object | None = None
        self._processor: object | None = None
        self._device: str = "cpu"
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoProcessor
            except ImportError as error:
                raise ModelNotAvailableError(f"torch/transformers are not installed: {error}") from error
            use_cuda = self._device_preference.startswith("cuda") and torch.cuda.is_available()
            self._device = self._device_preference if use_cuda else "cpu"
            dtype = torch.float16 if use_cuda else torch.float32
            token = os.environ.get("HF_TOKEN")
            try:
                model = AutoModel.from_pretrained(self._model_id, torch_dtype=dtype, token=token)
                processor = AutoProcessor.from_pretrained(self._model_id, token=token)
            except OSError as error:
                raise ModelNotAvailableError(
                    f"MedSigLIP weights for '{self._model_id}' are not available ({error}). "
                    "Accept the model terms on Hugging Face and run scripts/download_models.py."
                ) from error
            model.to(self._device)
            model.eval()
            self._processor = processor
            self._model = model

    @staticmethod
    def _to_pil_images(images_bgr: list[np.ndarray]) -> list[object]:
        from PIL import Image

        pil_images = []
        for image in images_bgr:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb_image))
        return pil_images

    @staticmethod
    def _as_tensor(features: object) -> object:
        if hasattr(features, "pooler_output") and getattr(features, "pooler_output") is not None:
            return getattr(features, "pooler_output")
        return features

    def encode_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Encode BGR images into normalised embeddings of shape (N, D)."""
        self._ensure_loaded()
        import torch

        if not images_bgr:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings: list[np.ndarray] = []
        with self._lock:
            try:
                for start in range(0, len(images_bgr), self._batch_size):
                    batch = self._to_pil_images(images_bgr[start : start + self._batch_size])
                    inputs = self._processor(images=batch, return_tensors="pt")  # type: ignore[operator]
                    pixel_values = inputs["pixel_values"].to(self._device, dtype=self._model.dtype)  # type: ignore[attr-defined]
                    with torch.inference_mode():
                        features = self._as_tensor(self._model.get_image_features(pixel_values=pixel_values))  # type: ignore[attr-defined]
                        features = torch.nn.functional.normalize(features.float(), dim=-1)  # type: ignore[attr-defined]
                    embeddings.append(features.cpu().numpy())  # type: ignore[attr-defined]
            except Exception as error:
                raise ModelInferenceError(f"MedSigLIP image encoding failed: {error}") from error
        return np.concatenate(embeddings, axis=0).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode text prompts into normalised embeddings of shape (N, D)."""
        self._ensure_loaded()
        import torch

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with self._lock:
            try:
                inputs = self._processor(  # type: ignore[operator]
                    text=texts, padding="max_length", truncation=True, return_tensors="pt"
                )
                input_ids = inputs["input_ids"].to(self._device)
                with torch.inference_mode():
                    features = self._as_tensor(self._model.get_text_features(input_ids=input_ids))  # type: ignore[attr-defined]
                    features = torch.nn.functional.normalize(features.float(), dim=-1)  # type: ignore[attr-defined]
            except Exception as error:
                raise ModelInferenceError(f"MedSigLIP text encoding failed: {error}") from error
        return features.cpu().numpy().astype(np.float32)  # type: ignore[attr-defined]

    def logit_parameters(self) -> tuple[float, float]:
        """Return the learned (logit_scale, logit_bias) used to turn similarities into logits."""
        self._ensure_loaded()
        model = self._model
        scale = float(model.logit_scale.exp().item())  # type: ignore[attr-defined]
        bias_parameter = getattr(model, "logit_bias", None)
        bias = float(bias_parameter.item()) if bias_parameter is not None else 0.0
        return scale, bias
