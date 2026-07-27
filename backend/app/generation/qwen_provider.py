from __future__ import annotations

import logging
import re
from typing import List, Sequence

import httpx

from app.config import Settings
from app.generation.base import LLMProvider
from app.generation.prompts import FALLBACK_ANSWER, SYSTEM_PROMPT, build_user_prompt
from app.models.api import ChatMessage, CitationSource, RetrievedChunk

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
    """Used in tests / when no LLM is configured to run.

    Still routes evidence through a generation path (compose_answer) rather than
    returning the first retrieved chunk verbatim.
    """

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

        from app.generation.answer_composer import compose_answer
        from app.generation.citations import sources_from_answer
        from app.retrieval.query_understanding import understand_query

        # Trust retrieval + fact filtering; do not drop evidence solely due to
        # weak lexical overlap (e.g. "found" vs "founder").
        understanding = understand_query(question, history=history)
        answer, _composed_sources = compose_answer(
            understanding=understanding, evidence=evidence
        )
        if not answer or answer == FALLBACK_ANSWER:
            return FALLBACK_ANSWER, []

        from app.generation.answer_composer import _finalize_answer, _looks_like_raw_chunk_dump

        answer = _finalize_answer(answer)
        if not answer or answer == FALLBACK_ANSWER or _looks_like_raw_chunk_dump(
            answer, evidence
        ):
            return FALLBACK_ANSWER, []

        # Attach citation markers for the evidence that supports the answer.
        cited_indices: List[int] = []
        answer_l = answer.lower()
        for index, item in enumerate(evidence, start=1):
            tokens = [
                token
                for token in re.findall(r"[a-z0-9]{4,}", item.content.lower())
                if token
                not in {"with", "that", "this", "from", "have", "record", "type", "nominee"}
            ]
            if any(token in answer_l for token in tokens[:12]):
                cited_indices.append(index)
            if len(cited_indices) >= 3:
                break
        if cited_indices:
            markers = " ".join(f"[{n}]" for n in cited_indices)
            if markers not in answer:
                answer = f"{answer.rstrip()} {markers}"

        sources = sources_from_answer(
            answer, evidence, query_type=understanding.query_type
        )
        return answer, sources


def format_sources(evidence: Sequence[RetrievedChunk]) -> List[CitationSource]:
    from app.generation.citations import dedupe_sources

    return dedupe_sources(evidence)


def create_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower().strip()
    if provider in {"ollama", "qwen", "openai_compatible", "vllm"}:
        return QwenCompatibleProvider(settings)
    if provider in {"huggingface", "hf", "huggingface_hub"}:
        if not settings.hf_token.strip():
            raise ValueError("HF_TOKEN is required when LLM_PROVIDER=huggingface")
        from app.generation.huggingface_provider import HuggingFaceGenerationProvider
        return HuggingFaceGenerationProvider(settings)
    if provider in {"deterministic", "fallback", "none"}:
        return DeterministicFallbackProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
