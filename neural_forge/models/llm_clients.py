"""Async HTTP clients for the local vLLM OpenAI-compatible servers."""

import asyncio
import re
from pathlib import Path
from typing import Any

import httpx

from neural_forge.core.exceptions import ConfigurationError, LLMServiceError

CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_text(content: str) -> str:
    """Extract the outermost JSON object from an LLM response, tolerating code fences and prose."""
    stripped = content.strip()
    fence_match = CODE_FENCE_PATTERN.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()
    start_index = stripped.find("{")
    end_index = stripped.rfind("}")
    if start_index == -1 or end_index <= start_index:
        raise ValueError("No JSON object found in the model response")
    return stripped[start_index : end_index + 1]


def load_prompt(prompts_dir: Path, file_name: str) -> str:
    """Read a prompt template from the prompts directory."""
    path = prompts_dir / file_name
    if not path.exists():
        raise ConfigurationError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def text_part(text: str) -> dict[str, Any]:
    """Build a text content part for a chat message."""
    return {"type": "text", "text": text}


def image_part(jpeg_base64: str) -> dict[str, Any]:
    """Build an inline JPEG image content part for a chat message."""
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_base64}"}}


class VLLMChatClient:
    """Calls /v1/chat/completions on a vLLM server using httpx; never loads a model locally."""

    def __init__(
        self,
        role: str,
        base_url: str,
        served_model_name: str,
        timeout_seconds: float,
        health_timeout_seconds: float,
        default_max_tokens: int,
        default_temperature: float,
    ) -> None:
        self._role = role
        self._base_url = base_url.rstrip("/")
        self._served_model_name = served_model_name
        self._timeout_seconds = timeout_seconds
        self._health_timeout_seconds = health_timeout_seconds
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def role(self) -> str:
        """Role of this server (reasoning or consultant)."""
        return self._role

    @property
    def base_url(self) -> str:
        """Base URL of the vLLM server."""
        return self._base_url

    @property
    def served_model_name(self) -> str:
        """Model name served by vLLM."""
        return self._served_model_name

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout_seconds, connect=10.0),
                )
            return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        payload: dict[str, Any] = {
            "model": self._served_model_name,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self._default_max_tokens,
            "temperature": temperature if temperature is not None else self._default_temperature,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_response", "schema": json_schema},
            }
        try:
            return await self._post_completion(payload)
        except LLMServiceError as error:
            if json_schema is not None and error.status_code == 400:
                payload.pop("response_format", None)
                return await self._post_completion(payload)
            raise

    async def _post_completion(self, payload: dict[str, Any]) -> str:
        client = await self._get_client()
        try:
            response = await client.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as error:
            raise LLMServiceError(f"{self._role} LLM server at {self._base_url} is unreachable: {error}") from error
        if response.status_code != 200:
            raise LLMServiceError(
                f"{self._role} LLM server returned {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
            )
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise LLMServiceError(f"{self._role} LLM server returned an unexpected payload: {error}") from error
        if not isinstance(content, str) or not content.strip():
            raise LLMServiceError(f"{self._role} LLM server returned empty content")
        return content

    async def is_healthy(self) -> bool:
        """Return whether the server's /health endpoint responds with 200."""
        try:
            async with httpx.AsyncClient(timeout=self._health_timeout_seconds) as client:
                response = await client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        async with self._client_lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None
