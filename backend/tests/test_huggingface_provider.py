"""Unit tests for HuggingFaceGenerationProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.generation.huggingface_provider import HuggingFaceGenerationProvider
from app.generation.prompts import FALLBACK_ANSWER
from app.generation.qwen_provider import create_llm_provider
from app.models.api import ChatMessage, RetrievedChunk


def _settings(**overrides):
    base = {
        "hf_token": "hf_test_token",
        "llm_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llm_temperature": 0.1,
        "llm_max_tokens": 200,
        "request_timeout_seconds": 60,
        "llm_provider": "huggingface",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _evidence() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            content="Acme provides inventory automation for mid-market retailers.",
            document_name="overview.pdf",
            page_number=1,
            score=0.9,
        )
    ]


@pytest.mark.asyncio
async def test_generate_returns_answer_and_sources():
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Acme provides inventory automation for mid-market retailers. [1]"
                )
            )
        ]
    )

    with patch(
        "app.generation.huggingface_provider.InferenceClient",
        return_value=mock_client,
    ) as client_cls:
        provider = HuggingFaceGenerationProvider(_settings())
        answer, sources = await provider.generate(
            question="What does Acme provide?",
            evidence=_evidence(),
            history=[
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello"),
            ],
        )

    client_cls.assert_called_once_with(token="hf_test_token", timeout=60)
    mock_client.chat_completion.assert_called_once()
    call_kwargs = mock_client.chat_completion.call_args.kwargs
    assert call_kwargs["model"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["max_tokens"] == 200
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Hi"}
    assert messages[2] == {"role": "assistant", "content": "Hello"}
    assert messages[-1]["role"] == "user"
    assert answer.startswith("Acme provides inventory automation")
    assert len(sources) == 1
    assert sources[0].document_name == "overview.pdf"
    assert sources[0].number == 1


@pytest.mark.asyncio
async def test_generate_empty_evidence_returns_fallback():
    with patch("app.generation.huggingface_provider.InferenceClient"):
        provider = HuggingFaceGenerationProvider(_settings())
    answer, sources = await provider.generate(question="Anything?", evidence=[])
    assert answer == FALLBACK_ANSWER
    assert sources == []


@pytest.mark.asyncio
async def test_generate_maps_not_found_phrase_to_fallback():
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = (
        "I could not find that information in the available document."
    )
    with patch(
        "app.generation.huggingface_provider.InferenceClient",
        return_value=mock_client,
    ):
        provider = HuggingFaceGenerationProvider(_settings())
        answer, sources = await provider.generate(
            question="Unknown?",
            evidence=_evidence(),
        )
    assert answer == FALLBACK_ANSWER
    assert sources == []


def test_init_requires_hf_token():
    with pytest.raises(ValueError, match="HF_TOKEN"):
        HuggingFaceGenerationProvider(_settings(hf_token=""))


def test_factory_creates_huggingface_provider():
    with patch("app.generation.huggingface_provider.InferenceClient"):
        provider = create_llm_provider(_settings(llm_provider="huggingface"))
    assert provider.name == "huggingface"


def test_factory_rejects_blank_hf_token():
    with pytest.raises(ValueError, match="HF_TOKEN"):
        create_llm_provider(_settings(llm_provider="hf", hf_token="   "))
