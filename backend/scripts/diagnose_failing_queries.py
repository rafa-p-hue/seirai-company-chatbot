"""Temporary diagnosis harness — capture A–E evidence; no retrieval logic changes.

Writes a structured report for the four failing queries against the enriched
seven-file markdown corpus (facts required by the queries). Also checks the
PDF service-domain corpus for missing source text (pets / sofa).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["VECTOR_STORE"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["LLM_PROVIDER"] = "deterministic"
os.environ["APP_ENV"] = "development"
os.environ["NOT_FOUND_CONTACT_NAME"] = "Hikari City"
os.environ["LOG_LEVEL"] = "INFO"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("diagnose_failures")

QUERIES = [
    "I just moved to Hikari City. When must I register, and what do I bring?",
    "Which evacuation shelter accepts pets?",
    "How much does a residence certificate (juminhyo) cost?",
    "How do I throw away a sofa?",
]

ENRICHED_CORPUS: List[Dict[str, Any]] = [
    {
        "filename": "Resident_Registration_Moving_In.md",
        "lines": [
            "# Resident Registration and Moving-In Guide",
            "Status: current",
            "## Moving In Notification",
            "You must register within 14 days of moving in.",
            "Bring your residence card, passport or national ID.",
            "Submit the move-in notification at the Citizen Services Window.",
        ],
    },
    {
        "filename": "National_Health_Insurance_Enrollment.md",
        "lines": [
            "# National Health Insurance Enrollment Guide",
            "Status: current",
            "If you leave your employer's insurance, enroll within 14 days.",
            "Bring your residence card and certificate of loss of eligibility.",
            "## Patient Cost Share",
            "With National Health Insurance, you normally pay 30% of medical costs at the counter.",
        ],
    },
    {
        "filename": "Child_Allowance_Application.md",
        "lines": [
            "# Child Allowance and Childcare Support Guide",
            "Status: current",
            "For a 2-year-old child, child allowance is 15,000 yen per month.",
        ],
    },
    {
        "filename": "Waste_and_Recycling_Collection.md",
        "lines": [
            "# Waste and Recycling Collection Guide",
            "Status: current",
            "## Household Waste Collection Days",
            "Burnable garbage is collected every Tuesday and Friday.",
            "Recyclables are collected every Wednesday.",
            "## Oversized Garbage Disposal",
            "Sofas are classified as oversized garbage.",
            "Reserve collection at least one week ahead by phone or online.",
            "Buy the required oversized-garbage sticker.",
            "Sticker fees range from 200 to 1,500 yen depending on size.",
            "Attach the sticker to the item.",
            "Place the item out by 8:00 a.m. on the reserved collection day.",
        ],
    },
    {
        "filename": "Certificates_and_Fees_Current.md",
        "lines": [
            "# Certificates and Fees — Current Fee Schedule",
            "Status: current",
            "Effective date: April 1, 2025",
            "## Certificate Issuance Fees",
            "Residence certificate (juminhyo) counter fee: 300 yen.",
            "Residence certificate (juminhyo) kiosk fee: 200 yen.",
            "Family register abstract: 450 yen.",
        ],
    },
    {
        "filename": "Certificates_and_Fees_Archived_2024.md",
        "lines": [
            "# Certificates and Fees — Archived Fee Schedule 2024",
            "Status: archived",
            "Residence certificate counter fee: 400 yen.",
            "This archived schedule is no longer valid.",
        ],
    },
    {
        "filename": "Disaster_Preparedness_Guide.md",
        "lines": [
            "# Disaster Preparedness Guide",
            "Status: current",
            "## Evacuation Shelters",
            "Greenfield Community Center accepts pets. Capacity: 300 people.",
            "Riverside Gym does not accept pets. Capacity: 500 people.",
            "Harborlight School accepts pets only in carriers. Capacity: 200 people.",
            "Pet-friendly shelter summary: Greenfield Community Center accepts pets and has capacity for 300 people.",
        ],
    },
]

# PDF service-domain fixture content (for absence check only).
PDF_FIXTURE_ABSENCE = {
    "Disaster_Preparedness_Guide.pdf": {
        "has_pets": False,
        "note": "PDF service-domain fixture has stockpile text only — no pet shelters.",
    },
    "Waste_and_Recycling_Collection.pdf": {
        "has_sofa": False,
        "note": "PDF service-domain fixture has collection days only — no oversized/sofa.",
    },
    "Certificates_and_Fees_Current.pdf": {
        "has_juminhyo": False,
        "note": "PDF fixture says 'Residence certificate' without '(juminhyo)' alias.",
    },
}


def _md(lines: List[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


def _mid_sentence(text: str) -> Dict[str, bool]:
    t = (text or "").strip()
    if not t:
        return {"starts_mid_sentence": False, "ends_mid_sentence": False}
    starts = t[0].islower()
    ends = not t.endswith((".", "!", "?", ":", ";")) and len(t.split()) > 3
    return {"starts_mid_sentence": starts, "ends_mid_sentence": ends}


def _classify(
    *,
    query: str,
    answer: str,
    sources: List[str],
    indexed_hits: List[Dict[str, Any]],
    diag: Dict[str, Any],
    pdf_missing: Optional[str],
) -> str:
    answer_l = (answer or "").lower()
    not_found = "could not find" in answer_l
    rejected = diag.get("rejected_cross_domain") or []
    entity = diag.get("entity_validation") or {}
    evidence_sent = diag.get("evidence_sent_to_llm") or diag.get("final_chunks_sent_to_llm") or []
    top = diag.get("top_candidates_across_files") or []

    # Expected source keywords per query.
    expected_kw = {
        "register": ["14", "residence card"],
        "pets": ["greenfield", "accepts pets", "pet"],
        "juminhyo": ["300", "200", "juminhyo", "residence certificate"],
        "sofa": ["oversized", "sofa", "sticker"],
    }
    key = (
        "register"
        if "register" in query.lower()
        else "pets"
        if "pets" in query.lower()
        else "juminhyo"
        if "juminhyo" in query.lower() or "certificate" in query.lower()
        else "sofa"
    )
    need = expected_kw[key]
    answer_ok = all(
        any(tok in answer_l for tok in ([n] if isinstance(n, str) else n))
        for n in need[:1]
    )  # loose: first token present

    if pdf_missing:
        return "1. extraction problem" if "fixture" in (pdf_missing or "") else "3. indexing/filter problem"

    if not indexed_hits:
        return "3. indexing/filter problem"

    # Correct content indexed?
    blob = "\n".join(h.get("content") or "" for h in indexed_hits).lower()
    content_present = any(tok.lower() in blob for tok in need)

    if not content_present:
        return "1. extraction problem"

    if not top:
        return "3. indexing/filter problem"

    # Was correct candidate in top 20?
    top_blob = "\n".join(
        f"{c.get('content') or ''} {c.get('preview') or ''}" for c in top
    ).lower()
    in_top = any(tok.lower() in top_blob for tok in need)

    if not in_top:
        return "4. ranking problem"

    # Correct in top but rejected?
    if rejected or entity.get("rejection_reason"):
        kept = diag.get("evidence_after_fact_filter") or evidence_sent
        kept_blob = "\n".join(
            f"{c.get('preview') or c.get('content') or ''}" for c in kept
        ).lower()
        if not any(tok.lower() in kept_blob for tok in need):
            return "6. validation too strict"

    if evidence_sent:
        sent_blob = "\n".join(
            f"{c.get('preview') or c.get('content') or ''}" for c in evidence_sent
        ).lower()
        if any(tok.lower() in sent_blob for tok in need) and (
            not_found or not any(tok.lower() in answer_l for tok in need)
        ):
            return "7. generation omitted evidence"

    if diag.get("resolved_contextual_question") and diag.get(
        "resolved_contextual_question"
    ) != diag.get("original_question"):
        # Only classify as context if that caused wrong domain/evidence.
        if (diag.get("service_domain") or "") in {
            "childcare_support",
            "certificate_fees",
            "health_insurance",
        } and key == "pets":
            return "5. context-resolution problem"

    if not_found:
        return "6. validation too strict" if entity.get("rejection_reason") else "4. ranking problem"

    if answer_ok:
        return "PASS (no failure in this harness)"

    return "7. generation omitted evidence"


def main() -> None:
    from fastapi.testclient import TestClient

    tmp = tempfile.mkdtemp(prefix="fail_diag_")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/diag.db"
    os.chdir(tmp)

    from app import dependencies
    from app.config import get_settings
    from app.main import create_app
    from app.retrieval.multi_question import split_questions

    get_settings.cache_clear()
    dependencies.get_embedding_provider.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm_provider.cache_clear()

    settings = get_settings()
    report: Dict[str, Any] = {
        "runtime": {},
        "pdf_fixture_absence_check": PDF_FIXTURE_ABSENCE,
        "queries": [],
    }

    with TestClient(create_app()) as client:
        # Runtime fingerprint
        store = dependencies.get_vector_store()
        emb = dependencies.get_embedding_provider()
        dim = getattr(emb, "dimension", None) or getattr(emb, "dims", None)
        try:
            probe = __import__("asyncio").get_event_loop().run_until_complete(
                emb.embed_query("probe")
            )
            dim = len(probe)
        except Exception:
            try:
                import asyncio

                dim = len(asyncio.run(emb.embed_query("probe")))
            except Exception as exc:  # noqa: BLE001
                dim = f"error:{exc}"

        report["runtime"] = {
            "active_project_path": str(ROOT.parent),
            "active_python_interpreter": sys.executable,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": getattr(settings, "embedding_model", None)
            or getattr(emb, "model_name", type(emb).__name__),
            "embedding_dimension": dim,
            "vector_store": settings.vector_store,
            "vector_collection": getattr(settings, "qdrant_collection", None)
            or getattr(store, "collection", type(store).__name__),
            "llm_provider": settings.llm_provider,
            "temperature": getattr(settings, "llm_temperature", None),
            "live_session_route": "POST /api/chat/sessions/{id}/messages",
            "live_chat_route_with_diagnostics": "POST /api/chat",
            "service": "ChatHistoryService.add_message -> ChatService.chat",
        }

        session = client.post("/api/chat/sessions").json()
        session_id = session["id"]
        company_id = "fail-diag"
        for item in ENRICHED_CORPUS:
            resp = client.post(
                f"/api/chat/sessions/{session_id}/attachments",
                data={"company_id": company_id},
                files={
                    "file": (item["filename"], _md(item["lines"]), "text/markdown")
                },
            )
            assert resp.status_code == 201, resp.text

        payloads = __import__("asyncio").get_event_loop().run_until_complete(
            store.list_payloads(
                company_id, session_id=session_id, include_company_docs=False
            )
        ) if hasattr(__import__("asyncio"), "get_event_loop") else []
        try:
            import asyncio

            payloads = asyncio.get_event_loop().run_until_complete(
                store.list_payloads(
                    company_id, session_id=session_id, include_company_docs=False
                )
            )
        except Exception:
            import asyncio

            payloads = asyncio.run(
                store.list_payloads(
                    company_id, session_id=session_id, include_company_docs=False
                )
            )

        report["runtime"]["indexed_chunks_for_session"] = len(payloads)

        # Ingestion dump
        ingestion_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for idx, payload in enumerate(payloads):
            name = str(payload.get("document_name") or "")
            content = str(payload.get("content") or "")
            row = {
                "chunk_order": idx,
                "chunk_id": payload.get("chunk_id") or payload.get("content_hash"),
                "content": content,
                "filename": name,
                "file_type": payload.get("file_type") or Path(name).suffix.lstrip("."),
                "page": payload.get("page_number"),
                "slide": payload.get("slide_number"),
                "row": payload.get("row_index") or payload.get("row_number"),
                "section": payload.get("section_title"),
                "service_domain": payload.get("service_domain"),
                "session_id": payload.get("session_id"),
                "document_scope": payload.get("document_scope"),
                "document_status": payload.get("document_status"),
                **_mid_sentence(content),
            }
            ingestion_by_file.setdefault(name, []).append(row)

        for q in QUERIES:
            print("\n" + "=" * 88)
            print(f"QUERY: {q}")
            print("=" * 88)

            # Prefer /api/chat so diagnostics are returned.
            resp = client.post(
                "/api/chat",
                json={
                    "company_id": company_id,
                    "question": q,
                    "session_id": session_id,
                    "include_company_docs": False,
                    "top_k": 8,
                    "conversation_id": session_id,
                },
            )
            data = resp.json()
            answer = data.get("answer") or ""
            sources = [
                s.get("document_name") for s in (data.get("sources") or []) if s.get("document_name")
            ]
            diag = data.get("diagnostics") or {}

            # Relevant files for this query
            relevant_files = []
            ql = q.lower()
            if "register" in ql or "bring" in ql:
                relevant_files = ["Resident_Registration_Moving_In.md"]
            elif "pets" in ql:
                relevant_files = ["Disaster_Preparedness_Guide.md"]
            elif "certificate" in ql or "juminhyo" in ql:
                relevant_files = [
                    "Certificates_and_Fees_Current.md",
                    "Certificates_and_Fees_Archived_2024.md",
                ]
            elif "sofa" in ql:
                relevant_files = ["Waste_and_Recycling_Collection.md"]

            indexed_hits = []
            for fname in relevant_files:
                indexed_hits.extend(ingestion_by_file.get(fname, []))

            pdf_missing = None
            if "pets" in ql:
                pdf_missing = PDF_FIXTURE_ABSENCE["Disaster_Preparedness_Guide.pdf"]["note"]
            elif "sofa" in ql:
                pdf_missing = PDF_FIXTURE_ABSENCE["Waste_and_Recycling_Collection.pdf"]["note"]

            classification = _classify(
                query=q,
                answer=answer,
                sources=sources,
                indexed_hits=indexed_hits,
                diag=diag,
                pdf_missing=None,  # enriched corpus used for live path
            )

            entry = {
                "query": q,
                "split_sub_questions": split_questions(q),
                "answer": answer,
                "sources": sources,
                "classification": classification,
                "A_ingestion": {
                    "source_lines": [
                        item["lines"]
                        for item in ENRICHED_CORPUS
                        if item["filename"] in relevant_files
                    ],
                    "chunks": indexed_hits,
                    "pdf_fixture_note": pdf_missing,
                },
                "B_retrieval": {
                    "original_query": diag.get("original_question"),
                    "normalized_query": diag.get("normalized_question"),
                    "resolved_query": diag.get("resolved_question")
                    or diag.get("resolved_contextual_question"),
                    "inherited_conversation_context": diag.get("conversation_anchors")
                    or diag.get("procedure_context"),
                    "service_domain": diag.get("service_domain"),
                    "top_20_before_rerank": diag.get("candidates_before_rerank")
                    or diag.get("top_candidates_across_files"),
                    "top_20_scored": diag.get("top_candidates_across_files"),
                    "rejected_cross_domain": diag.get("rejected_cross_domain"),
                },
                "C_validation": {
                    "entity_validation": diag.get("entity_validation"),
                    "rejection_reason": diag.get("rejection_reason"),
                    "evidence_after_fact_filter": diag.get("evidence_after_fact_filter"),
                    "validation_rescue": diag.get("validation_rescue"),
                    "rejected_cross_domain": diag.get("rejected_cross_domain"),
                },
                "D_generation": {
                    "evidence_sent_to_llm": diag.get("evidence_sent_to_llm")
                    or diag.get("final_chunks_sent_to_llm"),
                    "generated_answer_pre_cite": diag.get("generated_answer_pre_cite"),
                    "validation_result": diag.get("validation_result"),
                    "live_generation_path": diag.get("live_generation_path"),
                    "final_answer": answer,
                },
                "E_runtime": report["runtime"],
            }
            report["queries"].append(entry)

            print(f"ANSWER: {answer[:400]}")
            print(f"SOURCES: {sources}")
            print(f"DOMAIN: {diag.get('service_domain')} FACT: {diag.get('fact_type')}")
            print(f"SPLIT: {split_questions(q)}")
            print(f"CLASSIFICATION: {classification}")
            print(f"INGESTED RELEVANT CHUNKS: {len(indexed_hits)}")
            for c in indexed_hits[:8]:
                print(
                    f"  chunk {c['chunk_order']} id={c['chunk_id']} "
                    f"sec={c['section']!r} mid={c['starts_mid_sentence']}/{c['ends_mid_sentence']} "
                    f"dom={c['service_domain']} scope={c['document_scope']} "
                    f"text={c['content'][:100]!r}"
                )
            print("TOP CANDIDATES:")
            for i, c in enumerate((diag.get("top_candidates_across_files") or [])[:12], 1):
                print(
                    f"  {i}. {c.get('document_name')} | {c.get('section_title')} | "
                    f"dom={c.get('service_domain')} final={c.get('final_score')} "
                    f"vec={c.get('vector_score')} lex={c.get('lexical_score')} "
                    f"head={c.get('heading_score') or c.get('heading_boost')} "
                    f"ent={c.get('entity_score') or c.get('entity_boost')} "
                    f"cont={c.get('continuity_score') or c.get('continuity_boost')} "
                    f"arch={c.get('archive_penalty') or c.get('version_boost')} "
                    f"text={(c.get('clean_text') or c.get('content') or c.get('preview') or '')[:90]!r}"
                )
            print(
                "VALIDATION:",
                json.dumps(
                    {
                        "rejection": diag.get("rejection_reason"),
                        "entity": diag.get("entity_validation"),
                        "after_fact": [
                            {
                                "doc": x.get("document_name"),
                                "preview": (x.get("preview") or "")[:80],
                            }
                            for x in (diag.get("evidence_after_fact_filter") or [])
                        ],
                        "cross_domain": diag.get("rejected_cross_domain"),
                    },
                    default=str,
                )[:1200],
            )
            print(
                "GENERATION:",
                json.dumps(diag.get("live_generation_path") or {}, default=str),
                "pre_cite=",
                (diag.get("generated_answer_pre_cite") or "")[:200],
            )

        out = Path(tmp) / "failure_diagnosis_report.json"
        # Also copy beside backend for convenience
        local_out = ROOT / "scripts" / "failure_diagnosis_report.json"
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        try:
            local_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
        print(f"\nWrote report: {out}")
        print(f"Also: {local_out if local_out.exists() else 'n/a'}")

        print("\n=== FAILURE CLASSIFICATIONS ===")
        for entry in report["queries"]:
            print(f"- {entry['query'][:60]}... => {entry['classification']}")

        print("\n=== PDF FIXTURE ABSENCE (if live UI used those PDFs) ===")
        for k, v in PDF_FIXTURE_ABSENCE.items():
            print(f"- {k}: {v['note']}")


if __name__ == "__main__":
    main()
