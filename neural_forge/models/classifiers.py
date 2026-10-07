"""Image classifier strategies: MedSigLIP zero-shot and trained linear heads behind one interface."""

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np

from neural_forge.core.exceptions import ConfigurationError, ModelNotAvailableError
from neural_forge.models.medsiglip_neural_forge import MedSigLIPEncoder
from neural_forge.schemas.readings import ClassificationResult


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def build_results(task: str, method: Literal["zero_shot", "trained_head"], labels: list[str], probabilities: np.ndarray) -> list[ClassificationResult]:
    """Convert a probability matrix into classification results."""
    results: list[ClassificationResult] = []
    for row in probabilities:
        order = np.argsort(row)[::-1]
        top_index = int(order[0])
        second_score = float(row[order[1]]) if len(order) > 1 else 0.0
        top_score = float(row[top_index])
        results.append(
            ClassificationResult(
                task=task,
                method=method,
                label_scores={label: float(score) for label, score in zip(labels, row)},
                top_label=labels[top_index],
                top_score=float(np.clip(top_score, 0.0, 1.0)),
                margin=float(np.clip(top_score - second_score, 0.0, 1.0)),
            )
        )
    return results


class ImageClassifier(ABC):
    """Common interface so agents never depend on the classification strategy."""

    @abstractmethod
    def supports(self, task: str) -> bool:
        """Return whether this classifier can handle the task."""

    @abstractmethod
    def labels(self, task: str) -> list[str]:
        """Return the label names for a task."""

    @abstractmethod
    def classify(self, images_bgr: list[np.ndarray], task: str) -> list[ClassificationResult]:
        """Classify BGR image crops for a task."""


class ZeroShotClassifier(ImageClassifier):
    """Compares MedSigLIP image embeddings against averaged text-prompt embeddings."""

    def __init__(self, encoder: MedSigLIPEncoder, prompt_config: dict[str, Any]) -> None:
        self._encoder = encoder
        self._tasks: dict[str, dict[str, list[str]]] = {}
        for task_name, task_definition in prompt_config.get("tasks", {}).items():
            label_prompts = task_definition.get("labels", {})
            if len(label_prompts) < 2:
                raise ConfigurationError(f"Zero-shot task '{task_name}' needs at least two labels")
            self._tasks[task_name] = {label: list(prompts) for label, prompts in label_prompts.items()}
        self._label_embedding_cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def supports(self, task: str) -> bool:
        """Return whether prompts exist for the task."""
        return task in self._tasks

    def labels(self, task: str) -> list[str]:
        """Return label names in prompt-file order."""
        if task not in self._tasks:
            raise ConfigurationError(f"Unknown zero-shot task '{task}'")
        return list(self._tasks[task].keys())

    def _label_embeddings(self, task: str) -> np.ndarray:
        with self._lock:
            if task in self._label_embedding_cache:
                return self._label_embedding_cache[task]
        label_vectors: list[np.ndarray] = []
        for label, prompts in self._tasks[task].items():
            if not prompts:
                raise ConfigurationError(f"Zero-shot label '{label}' in task '{task}' has no prompts")
            prompt_embeddings = self._encoder.encode_texts(prompts)
            mean_vector = prompt_embeddings.mean(axis=0)
            norm = float(np.linalg.norm(mean_vector))
            label_vectors.append(mean_vector / norm if norm > 0 else mean_vector)
        matrix = np.stack(label_vectors, axis=0)
        with self._lock:
            self._label_embedding_cache[task] = matrix
        return matrix

    def classify(self, images_bgr: list[np.ndarray], task: str) -> list[ClassificationResult]:
        """Return softmax-normalised zero-shot probabilities per image."""
        if not images_bgr:
            return []
        if task not in self._tasks:
            raise ConfigurationError(f"Unknown zero-shot task '{task}'")
        label_matrix = self._label_embeddings(task)
        image_embeddings = self._encoder.encode_images(images_bgr)
        logit_scale, logit_bias = self._encoder.logit_parameters()
        logits = logit_scale * (image_embeddings @ label_matrix.T) + logit_bias
        probabilities = softmax_rows(logits)
        return build_results(task, "zero_shot", self.labels(task), probabilities)


class TrainedHeadClassifier(ImageClassifier):
    """Applies a trained linear head (stored as .npz) to MedSigLIP image embeddings."""

    def __init__(self, encoder: MedSigLIPEncoder, heads_dir: Path) -> None:
        self._encoder = encoder
        self._heads_dir = heads_dir
        self._heads: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        self._lock = threading.Lock()

    def _head_path(self, task: str) -> Path:
        return self._heads_dir / f"{task}.npz"

    def supports(self, task: str) -> bool:
        """Return whether a trained head file exists for the task."""
        return self._head_path(task).exists()

    def _load_head(self, task: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        with self._lock:
            if task in self._heads:
                return self._heads[task]
            path = self._head_path(task)
            if not path.exists():
                raise ModelNotAvailableError(f"No trained classifier head for task '{task}' at {path}")
            archive = np.load(path, allow_pickle=False)
            weights = np.asarray(archive["weights"], dtype=np.float32)
            bias = np.asarray(archive["bias"], dtype=np.float32)
            labels = [str(label) for label in archive["labels"]]
            if weights.shape[0] != len(labels) or bias.shape[0] != len(labels):
                raise ConfigurationError(f"Trained head for '{task}' has inconsistent shapes")
            self._heads[task] = (weights, bias, labels)
            return self._heads[task]

    def labels(self, task: str) -> list[str]:
        """Return the label names stored in the head file."""
        return self._load_head(task)[2]

    def classify(self, images_bgr: list[np.ndarray], task: str) -> list[ClassificationResult]:
        """Return softmax probabilities from the trained head."""
        if not images_bgr:
            return []
        weights, bias, labels = self._load_head(task)
        image_embeddings = self._encoder.encode_images(images_bgr)
        logits = image_embeddings @ weights.T + bias
        return build_results(task, "trained_head", labels, softmax_rows(logits))


class ClassifierRouter(ImageClassifier):
    """Selects zero-shot or trained-head classification per task according to the strategy."""

    def __init__(
        self,
        zero_shot: ZeroShotClassifier,
        trained_head: TrainedHeadClassifier,
        strategy: Literal["zero_shot", "trained_head", "auto"],
    ) -> None:
        self._zero_shot = zero_shot
        self._trained_head = trained_head
        self._strategy = strategy

    def _select(self, task: str) -> ImageClassifier:
        if self._strategy == "zero_shot":
            return self._zero_shot
        if self._strategy == "trained_head":
            if not self._trained_head.supports(task):
                raise ModelNotAvailableError(f"Strategy 'trained_head' selected but no head exists for '{task}'")
            return self._trained_head
        if self._trained_head.supports(task):
            return self._trained_head
        return self._zero_shot

    def supports(self, task: str) -> bool:
        """Return whether any strategy can classify the task."""
        return self._trained_head.supports(task) or self._zero_shot.supports(task)

    def labels(self, task: str) -> list[str]:
        """Return the labels of the selected strategy."""
        return self._select(task).labels(task)

    def classify(self, images_bgr: list[np.ndarray], task: str) -> list[ClassificationResult]:
        """Classify using the selected strategy."""
        return self._select(task).classify(images_bgr, task)
