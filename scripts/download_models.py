"""Download every model weight and task file into the model cache directory.

Runs inside the Hugging Face Space container on startup. Already-downloaded files are skipped.
HF_TOKEN is read from the environment only (Space secret). Do not run this on a development laptop.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_forge.core.protocol_loader import ConfigRepository  # noqa: E402
from neural_forge.core.settings import Settings, get_settings  # noqa: E402

logger = logging.getLogger("download_models")


def download_file(url: str, destination: Path, attempts: int = 3) -> None:
    """Download a URL atomically to a destination path with retries."""
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Already present: %s", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            logger.info("Downloading %s (attempt %d)", url, attempt)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, temporary)
            temporary_path.replace(destination)
            return
        except Exception as error:
            last_error = error
            logger.warning("Download failed: %s", error)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def download_mediapipe(settings: Settings) -> None:
    """Download the MediaPipe face and hand landmarker task files."""
    download_file(settings.face_landmarker_url, settings.face_landmarker_path)
    download_file(settings.hand_landmarker_url, settings.hand_landmarker_path)


def download_hub_model(repository_id: str) -> None:
    """Download a Hugging Face Hub model snapshot into HF_HOME."""
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    logger.info("Fetching %s from the Hugging Face Hub", repository_id)
    snapshot_download(
        repo_id=repository_id,
        token=token,
        ignore_patterns=["*.gguf", "*.onnx", "original/*", "*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )


def download_yoloe(settings: Settings, config: ConfigRepository) -> None:
    """Download YOLOE weights and warm the MobileCLIP text encoder used for text prompts."""
    target = settings.yoloe_weights_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(str(target)))
        if downloaded.resolve() != target.resolve() and downloaded.exists():
            shutil.move(str(downloaded), str(target))
    from ultralytics import YOLOE

    prompts = list(config.zero_shot_prompts().get("lesion_detection_prompts", ["skin lesion"]))
    model = YOLOE(str(target))
    model.get_text_pe(prompts)
    logger.info("YOLOE weights and text encoder ready at %s", target)


def download_rtmw(settings: Settings) -> None:
    """Trigger rtmlib's own downloader for the RTMW whole-body detector and pose ONNX files."""
    from rtmlib import Wholebody

    Wholebody(mode=settings.rtmw_mode, backend=settings.rtmw_backend, device="cpu", to_openpose=False)
    logger.info("RTMW (%s) ONNX files are cached", settings.rtmw_mode)


def main() -> int:
    """Download all assets and return a non-zero exit code if a required asset failed in strict mode."""
    parser = argparse.ArgumentParser(description="Download Neural Forge model assets.")
    parser.add_argument("--skip-llms", action="store_true", help="Do not download the vLLM-served models.")
    parser.add_argument("--strict", action="store_true", help="Exit with an error if any download fails.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    config = ConfigRepository(settings.config_dir)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set; gated models (MedGemma, MedSigLIP) will fail to download.")
    tasks: list[tuple[str, object]] = [
        ("mediapipe", lambda: download_mediapipe(settings)),
        ("medsiglip", lambda: download_hub_model(settings.medsiglip_model_id)),
        ("yoloe", lambda: download_yoloe(settings, config)),
        ("rtmw", lambda: download_rtmw(settings)),
    ]
    if not arguments.skip_llms:
        tasks.insert(0, ("reasoning_llm", lambda: download_hub_model(settings.reasoning_model_id)))
        tasks.insert(1, ("consultant_llm", lambda: download_hub_model(settings.consultant_model_id)))
    failures: list[str] = []
    for name, task in tasks:
        try:
            task()  # type: ignore[operator]
            logger.info("Asset ready: %s", name)
        except Exception as error:
            failures.append(name)
            logger.error("Asset failed: %s (%s)", name, error)
    if failures:
        logger.error("Failed assets: %s", ", ".join(failures))
        return 1 if arguments.strict else 0
    logger.info("All model assets are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
