from __future__ import annotations

import logging
import re
from typing import List, Sequence

import httpx

from app.config import Settings
from app.generation.base import LLMProvider
from app.generation.prompts import FALLBACK_ANSWER, SYSTEM_PROMPT, build_user_prompt
from app.models.api import ChatMessage, CitationSource, RetrievedChunk, SourceType

logger = logging.getLogger(__name__)


class QwenCompatibleProvider(LLMProvider):
    """Calls a Qwen-compatible chat model via an OpenAI-style HTTP API.

    Works with Ollama (`/api/chat` or OpenAI-compatible `/v1/chat/completions`)
    depending on configuration. Default assumes Ollama native chat endpoint.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.name = settings.llm_provider
        self.model_name = settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")

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

        try:
            answer = await self._call_model(user_prompt, history or [])
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM generation failed")
            raise RuntimeError(
                f"LLM provider '{self.name}' failed for model '{self.model_name}': {exc}"
            ) from exc

        cleaned = (answer or "").strip()
        if not cleaned:
            return FALLBACK_ANSWER, []
        if "could not find that information" in cleaned.lower():
            return FALLBACK_ANSWER, []
        return cleaned, sources

    async def _call_model(self, user_prompt: str, history: Sequence[ChatMessage]) -> str:
        # Prefer OpenAI-compatible endpoint when available; fall back to Ollama native.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history[-6:]:
            if item.role in {"user", "assistant"} and item.content.strip():
                messages.append({"role": item.role, "content": item.content.strip()})
        messages.append({"role": "user", "content": user_prompt})

        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            openai_url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
            }
            response = await client.post(openai_url, json=payload)
            if response.status_code == 404:
                # Ollama native chat API
                ollama_payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.settings.llm_temperature,
                        "num_predict": self.settings.llm_max_tokens,
                    },
                }
                response = await client.post(f"{self.base_url}/api/chat", json=ollama_payload)
            if response.status_code >= 400:
                detail = response.text[:500]
                raise RuntimeError(
                    f"HTTP {response.status_code} from LLM endpoint. "
                    f"Check LLM_BASE_URL and that model '{self.model_name}' is installed. "
                    f"Detail: {detail}"
                )
            data = response.json()
            if "choices" in data:
                return str(data["choices"][0]["message"]["content"])
            if "message" in data:
                return str(data["message"].get("content") or "")
            return str(data.get("response") or "")


class DeterministicFallbackProvider(LLMProvider):
    """Used in tests / when no LLM is configured to run."""

    def __init__(self) -> None:
        self.name = "deterministic-fallback"
        self.model_name = "deterministic-v1"

    async def generate(
        self,
        *,
        question: str,
        evidence: Sequence[RetrievedChunk],
        history: Sequence[ChatMessage] | None = None,
    ) -> tuple[str, List[CitationSource]]:
        if not evidence:
            return FALLBACK_ANSWER, []

        # Refuse when the question is clearly unsupported by retrieved text
        # (e.g. refund policy questions against a document with no refund content).
        from app.retrieval.hybrid_search import hybrid_score

        best = max(hybrid_score(question, item.content) for item in evidence)
        if best < 0.15:
            return FALLBACK_ANSWER, []

        sources = format_sources(evidence)
        snippets = []
        for index, item in enumerate(evidence[:3], start=1):
            snippet = re.sub(r"\s+", " ", item.content).strip()
            snippets.append(f"{snippet[:240]} [{index}]")
        answer = " ".join(snippets).strip()
        return answer or FALLBACK_ANSWER, sources


def format_sources(evidence: Sequence[RetrievedChunk]) -> List[CitationSource]:
    sources: List[CitationSource] = []
    for index, item in enumerate(evidence, start=1):
        source_type = SourceType.website if item.source_url else SourceType.pdf
        sources.append(
            CitationSource(
                number=index,
                document_name=item.document_name,
                page_number=item.page_number,
                source_url=item.source_url,
                source_type=source_type,
            )
        )
    return sources


def create_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower().strip()
    if provider in {"ollama", "qwen", "openai_compatible", "vllm"}:
        return QwenCompatibleProvider(settings)
    if provider in {"deterministic", "fallback", "none"}:
        return DeterministicFallbackProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
