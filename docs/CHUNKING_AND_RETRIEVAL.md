# Chunking and retrieval quality

How this project judges **good chunks** and **good retrieval** for document-agnostic RAG (handbooks, policies, profiles, and similar multi-section docs).

Implementation lives mainly in:

| Area | Modules |
|---|---|
| Chunking | `backend/app/ingestion/universal_chunker.py`, `pipeline.py` |
| Ranking signals | `backend/app/retrieval/section_match.py`, `numeric_facts.py`, `chunk_quality.py`, `retriever.py` |
| Fact gating | `backend/app/retrieval/fact_types.py` |
| Answer scrubbing | `backend/app/generation/text_scrub.py`, `prompts.py` |

---

## What “good” chunking means

A high-quality chunk is something a reader (or LLM) can answer from **without neighboring chunks**.

| Criterion | Expectation |
|---|---|
| **Section context** | Chunk carries the active section / subsection heading so policy facts stay attributed to the right topic. |
| **Complete facts** | Prices, dates, quantities, and rules are not split mid-sentence across chunks when avoidable. |
| **Intact lists** | Related list items (prohibitions, steps, fee rows) stay together when they fit the token budget; packing prefers keeping list groups whole. |
| **Overview isolation** | Intro / about / summary material is its own chunk(s), not mixed into Membership, Refunds, Accessibility, etc. |
| **No internal metadata in answers** | Structured scaffolding (`Record type:`, `Person:`, `Profile Description:`, bare `Section:`) is stripped before user-facing text; answers must not echo it. |

Weak chunks (isolated contact labels, form instructions, bare headings, link chrome) are down-ranked or filtered at retrieval time.

---

## How chunks are structured

Universal chunks store content plus retrieval metadata:

| Field | Role |
|---|---|
| `document_name` / title | Document identity |
| `section_title` | Current major heading |
| `subsection_title` | Nested heading when present |
| `page_number` | Source page (citations) |
| `content_type` | `paragraph`, `list`, `key_value`, `heading`, `prompt`, or structured record types |
| `content` | The text unit (paragraph, packed list group, or labeled field) |
| `label` / `value` | For key-value lines (`Major: CS`) |

Surrounding sentence or list item stays with its section heading so ranking can boost heading↔query alignment without relying on org-specific keywords in production code.

Typical size targets: roughly **40–220 tokens** per packed unit (short labeled fields may be shorter).

---

## How retrieval ranking judges quality

Pipeline (simplified):

1. Hybrid dense + lexical search with a **candidate pool of ~15–20** (`RETRIEVAL_CANDIDATE_K`, clamped 15–20).
2. Score fusion and optional light rerank.
3. Diversity / fact-type filtering down to **~3–6** final evidence chunks (`RETRIEVAL_TOP_K` / chat `top_k`).

Important ranking signals (document-agnostic):

| Signal | Effect |
|---|---|
| **Section heading match** | Query cues (membership, pets, refund, accessibility, …) that align with `section_title` / `subsection_title` get a strong boost. |
| **Overview downrank** | Overview / about / intro / thin profile dumps are penalized for specific fact questions (price, policy, date, etc.). Broad “tell me about…” stays neutral. |
| **Numeric / date / currency / place** | Chunks whose content shape matches the question (currency for price, dates/periods for when, quantities for how many, place/address or explicit “not announced” for where) are boosted. |
| **Chunk quality** | Complete list entries and paragraphs score up; fragments, identity-only fields, and form prompts score down. |
| **Content-type compatibility** | Lists / paragraphs / key-value preferred for policy-style questions; bare headings without a section match are slightly demoted. |

After ranking, evidence may be filtered so only chunks that can support the detected **fact type** reach generation.

---

## Fact-type validation expectations

Questions are mapped to a fact type; evidence and answers must carry the right *shape* of support:

| Fact type | Evidence / answer must show |
|---|---|
| **price** | Currency (e.g. `$…`) |
| **date** | Date or period, **or** an explicit unannounced/TBD statement |
| **quantity** | A number / amount in context |
| **location** | Place/address signals, **or** explicit “not announced” |
| **policy** | A rule, permission, prohibition, or clearly related policy text |
| **accessibility** | Accessibility evidence, **or** explicit unannounced |

If typed evidence is missing, or the composed answer fails the same checks, the system falls back rather than inventing or answering from the wrong section (e.g. overview welcome text for a fee question).

---

## “Not announced” vs “not found”

| Situation | Expected behavior |
|---|---|
| Document **explicitly** says something is not announced / TBD / TBA / not determined | Answer that gap clearly (with citation when possible). |
| Relevant fact type is **absent** from retrieved evidence | Fallback: *“I could not find that information in the available document.”* |

Do not treat missing retrieval as “not announced,” and do not invent a date/address when the doc only says it is unannounced.

---

## How quality is tested

Regression coverage is **fixture-based**, not hardcoded product APIs:

- `backend/tests/test_harborlight_policy.py` — multi-section community-handbook **fixture only** (section routing, overview downrank, prices/policies/lists, accessibility, unannounced location, metadata scrub). Harborlight-style names and numbers belong in that test fixture, **not** in production guidance or ranking rules.
- Related suites: `test_fact_validation.py`, `test_document_agnostic.py`, `test_unsupported_and_research.py`, `test_rag_quality.py`, and other backend tests under `backend/tests/`.

Run:

```bash
./scripts/dev.sh backend:test
# or focus:
cd backend && pytest tests/test_harborlight_policy.py -q
```

When adding a new handbook-style behavior, prefer a generic sectioned fixture and assertions on **retrieval behavior** (right section, right fact shape, no overview bleed, no internal metadata), not org-specific golden answers in production modules.
