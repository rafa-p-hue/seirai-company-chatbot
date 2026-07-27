"""Hugging Face Inference API generation provider."""

from __future__ import annotations

import logging
from typing import List, Sequence

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from app.config import Settings
from app.generation.base import LLMProvider
from app.generation.prompts import FALLBACK_ANSWER, SYSTEM_PROMPT, build_user_prompt
from app.generation.qwen_provider import format_sources
from app.models.api import ChatMessage, CitationSource, RetrievedChunk

logger = logging.getLogger(__name__)


class HuggingFaceGenerationProvider(LLMProvider):
    """Hosted Hugging Face chat completion via InferenceClient."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.name = "huggingface"
        self.model_name = settings.llm_model
        if not settings.hf_token:
            raise ValueError("HF_TOKEN is required when LLM_PROVIDER=huggingface")
        self._client = InferenceClient(
            token=settings.hf_token,
            timeout=settings.request_timeout_seconds,
        )

    async def generate(
        self,
        *,
        question: str,
        evidence: Sequence[RetrievedChunk],
        history: Sequence[ChatMessage] | None = None,
    ) -> tuple[str, List[CitationSource]]:
        if not evidence:
            return FALLBACK_ANSWER, []

        sources = format_sources(evidence)
        user_prompt = build_user_prompt(question, evidence)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in (history or [])[-6:]:
            if item.role in {"user", "assistant"} and item.content.strip():
                messages.append({"role": item.role, "content": item.content.strip()})
        messages.append({"role": "user", "content": user_prompt})

        try:
            # InferenceClient is sync; fine for this prototype.
            completion = self._client.chat_completion(
                messages=messages,
                model=self.model_name,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            )
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None) or getattr(exc, "status_code", None)
            detail = str(exc)
            if status in {401, 403}:
                raise RuntimeError(
                    "Hugging Face authentication failed. Check HF_TOKEN."
                ) from exc
            if status == 404:
                raise RuntimeError(
                    f"Hugging Face model unavailable: '{self.model_name}'. "
                    "Check LLM_MODEL and that the model supports chat_completion."
                ) from exc
            if status == 429:
                raise RuntimeError(
                    "Hugging Face rate limit exceeded. Retry later."
                ) from exc
            if status and status >= 500:
                raise RuntimeError(
                    f"Hugging Face service error ({status}). Retry later. Detail: {detail[:300]}"
                ) from exc
            raise RuntimeError(f"Hugging Face request failed: {detail[:500]}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                "Hugging Face request timed out. Increase REQUEST_TIMEOUT_SECONDS or retry."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "timeout" in msg:
                raise RuntimeError(
                    "Hugging Face request timed out. Increase REQUEST_TIMEOUT_SECONDS or retry."
                ) from exc
            if "401" in msg or "403" in msg or "unauthorized" in msg or "invalid token" in msg:
                raise RuntimeError(
                    "Hugging Face authentication failed. Check HF_TOKEN."
                ) from exc
            if "429" in msg or "rate limit" in msg:
                raise RuntimeError(
                    "Hugging Face rate limit exceeded. Retry later."
                ) from exc
            logger.exception("Hugging Face generation failed")
            raise RuntimeError(
                f"Hugging Face provider failed for model '{self.model_name}': {exc}"
            ) from exc

        answer = _extract_answer(completion)
        cleaned = (answer or "").strip()
        if not cleaned:
            raise RuntimeError(
                "Hugging Face returned an empty response. Check LLM_MODEL and provider status."
            )
        if "could not find that information" in cleaned.lower():
            return FALLBACK_ANSWER, []
        return cleaned, sources


def _extract_answer(completion: object) -> str:
    # chat_completion may return object with choices or a plain string depending on version
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    choices = getattr(completion, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content:
            return str(content)
        text = getattr(choices[0], "text", None)
        if text:
            return str(text)
    if isinstance(completion, dict):
        choices = completion.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            return str(msg.get("content") or choices[0].get("text") or "")
    return str(completion)
