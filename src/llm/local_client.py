"""Local tiny LLM/VLM client (optional fallback)."""

import logging
from typing import List, Optional

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class LocalClient(BaseLLMClient):
    """
    Minimal local client.

    Tries to use llama-cpp-python if available; otherwise returns a
    clear error so callers can fall back to cloud providers.
    """

    def __init__(
        self,
        model: str = "local-llm",
        model_path: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self._model = model
        self._model_path = model_path
        self._max_tokens = max_tokens
        self._temperature = temperature

        self._llama = None
        if model_path:
            try:
                from llama_cpp import Llama  # type: ignore

                self._llama = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                )
                logger.info("Initialized local llama.cpp model")
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("Local LLM not available: %s", exc)

    def _not_ready_response(self, reason: str) -> LLMResponse:
        return LLMResponse(
            text=f"[local provider unavailable] {reason}",
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            model=self._model,
            provider=self.provider_name,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        if self._llama is None:
            return self._not_ready_response(
                "install `llama-cpp-python` and provide `model_path`"
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        text = result["choices"][0]["message"]["content"]

        return LLMResponse(
            text=text,
            input_tokens=0,
            output_tokens=len(text.split()),
            cost=0.0,
            model=self._model,
            provider=self.provider_name,
        )

    def generate_with_images(
        self,
        prompt: str,
        images: List[str],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        # Local vision support not implemented; return informative message.
        return self._not_ready_response("local vision not implemented; use cloud VLM")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "local"

