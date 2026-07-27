from __future__ import annotations

import logging
import uuid
from typing import Optional

from app.generation.answer_composer import compose_answer
from app.generation.base import LLMProvider
from app.generation.prompts import FALLBACK_ANSWER
from app.models.api import ChatRequest, ChatResponse
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, *, retriever: Retriever, llm: LLMProvider) -> None:
        self.retriever = retriever
        self.llm = llm

    async def chat(self, request: ChatRequest) -> ChatResponse:
        subject_name = _guess_subject(request)
        evidence, understanding = await self.retriever.retrieve(
            company_id=request.company_id,
            question=request.question,
            top_k=request.top_k or 5,
            history=request.history,
            subject_name=subject_name,
        )

        # Prefer structured deterministic answers for known query types.
        if understanding.query_type in {
            "greeting",
            "identity",
            "education",
            "experience",
            "internship",
            "research",
            "leadership",
            "skills",
            "unsupported",
        }:
            answer, sources = compose_answer(
                understanding=understanding, evidence=evidence
            )
        else:
            # General questions: try deterministic first, then LLM if configured.
            answer, sources = compose_answer(
                understanding=understanding, evidence=evidence
            )
            if answer == FALLBACK_ANSWER and evidence and self.llm.name != "deterministic-fallback":
                try:
                    answer, sources = await self.llm.generate(
                        question=understanding.expanded_question,
                        evidence=evidence,
                        history=request.history,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("LLM generation failed; using fallback")
                    answer, sources = FALLBACK_ANSWER, []

        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(
            "Chat company=%s type=%s evidence=%s",
            request.company_id,
            understanding.query_type,
            len(evidence),
        )
        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
        )


def _guess_subject(request: ChatRequest) -> Optional[str]:
    import re

    texts = [request.question] + [msg.content for msg in request.history[-6:]]
    for text in texts:
        match = re.search(
            r"who is\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\??",
            text,
            re.I,
        )
        if match:
            return match.group(1).strip()
    return None
