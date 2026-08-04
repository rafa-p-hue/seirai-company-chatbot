# Seirai Company Chatbot

Ask questions about your company documents and get answers grounded in those sources, with citations.

This is a **local-first RAG chatbot**: upload PDFs (and other files), ingest website pages, then chat through a Next.js UI backed by a FastAPI retrieval service.

## How it works

1. You upload documents (or ingest a website) for a `company_id`.
2. The backend extracts text, cleans it, splits it into chunks, and embeds them.
3. Vectors are stored in **Qdrant** (or an in-memory store for local testing).
4. When someone asks a question, the backend retrieves the best chunks, then an LLM writes a grounded answer with `[1]`, `[2]` citations.
5. The chat UI shows the answer and sources.

```text
Upload / crawl
  → extract → clean → chunk → embed
  → vector store (Qdrant or memory)
  → ask a question
  → hybrid retrieval (+ checklist assembly when needed)
  → LLM answer + citations
  → chat / embed UI
```

## Repository layout

```text
seirai-company-chatbot/
├── frontend/          # Next.js UI (chat, admin, embed widget)
├── backend/           # FastAPI RAG API
├── docs/              # Design notes (chunking & retrieval)
├── scripts/dev.sh     # Local helper commands
├── docker-compose.yml # Qdrant (+ optional backend)
└── README.md
```

## Quick start

You need three terminals (or run the helper script in each).

### 1. Vector store

**Preferred (Qdrant):**

```bash
chmod +x scripts/dev.sh
./scripts/dev.sh qdrant:up
```

**No Docker?** In `backend/.env` set:

```env
VECTOR_STORE=memory
```

Memory mode is fine for local demos. Data is lost when the backend restarts.

### 2. Backend API

```bash
./scripts/dev.sh backend:venv
cp backend/.env.example backend/.env
# edit backend/.env if needed
./scripts/dev.sh backend:dev
```

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

### 3. Frontend

```bash
cp frontend/.env.example frontend/.env.local
./scripts/dev.sh frontend:install
./scripts/dev.sh frontend:dev
```

| Page | URL |
|------|-----|
| Home | http://localhost:3000 |
| Chat (with history) | http://localhost:3000/chat |
| Admin / uploads | http://localhost:3000/admin |
| Embeddable chat | http://localhost:3000/embed/seirai |
| Per-company embed | http://localhost:3000/embed/{company_id} |

Frontend calls the API through `/rag-api` (proxied to FastAPI), so you usually do not need to fight CORS locally.

### 4. LLM (for real answers)

**Option A — Ollama (local):**

```bash
ollama pull qwen2.5:7b
```

In `backend/.env`:

```env
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b
```

**Option B — Hugging Face Inference:**

```env
LLM_PROVIDER=huggingface
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
HF_TOKEN=your_token_here
```

**Option C — Deterministic (tests / no LLM):**

```env
LLM_PROVIDER=deterministic
```

Useful for pytest and offline smoke checks. Answers are grounded snippets, not full generative replies.

## What you can ask

The system is **document-agnostic**: it answers from whatever you uploaded for that company.

It is especially careful with:

- **Fees / prices** — keeps labeled amounts and current vs archived schedules when present  
- **Checklists** (“What do I bring?”) — pulls the full required-item list from the same section, including alternatives, conditions, and household-wide items  
- **Compound questions** — e.g. “When must I register, and what do I bring?” is split and answered in parts  

Answers should stay grounded in retrieved evidence and cite sources. If nothing relevant is found, the bot says it could not find that information.

## Environment variables

Copy examples, then edit:

- `backend/.env` ← from `backend/.env.example`  
- `frontend/.env.local` ← from `frontend/.env.example`  

| Variable | Where | Purpose |
|----------|--------|---------|
| `NEXT_PUBLIC_RAG_API_URL` | Frontend | Usually `/rag-api` (proxied) |
| `RAG_API_URL` | Frontend | Backend origin for the proxy (`http://127.0.0.1:8000`) |
| `VECTOR_STORE` | Backend | `qdrant` or `memory` |
| `QDRANT_URL` | Backend | Qdrant URL when using Qdrant |
| `EMBEDDING_PROVIDER` | Backend | `local` (sentence-transformers), or `hash` for tests |
| `LLM_PROVIDER` | Backend | `ollama`, `huggingface`, or `deterministic` |
| `HF_TOKEN` | Backend | Required for Hugging Face |
| `DATABASE_URL` | Backend | SQLite chat history (default `sqlite:///./chat_history.db`) |
| `CORS_ORIGINS` | Backend | Allowed browser origins |
| `ALLOWED_CRAWL_DOMAINS` | Backend | Optional allowlist for website ingest |

## Main API endpoints

Base path is usually `/api` (see `API_PREFIX`).

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `POST` | `/api/documents/upload` | Upload a document |
| `GET` | `/api/documents?company_id=` | List documents |
| `DELETE` | `/api/documents/{id}?company_id=` | Delete a document |
| `POST` | `/api/documents/{id}/reprocess?company_id=` | Re-ingest |
| `POST` | `/api/websites/ingest` | Crawl / ingest a URL |
| `POST` | `/api/retrieve` | Retrieve chunks only |
| `POST` | `/api/chat` | One-shot Q&A |
| `*` | `/api/chat/sessions/...` | Persistent chat sessions, messages, attachments |

Every stored chunk is scoped by `company_id`. Retrieval always filters by company.

Interactive docs: http://localhost:8000/docs  

## Embed widget

```html
<script
  src="http://localhost:3000/widget.js"
  data-company-id="seirai"
  data-title="Ask Seirai AI"
  defer
></script>
```

## Testing

```bash
./scripts/dev.sh backend:test
./scripts/dev.sh frontend:lint
./scripts/dev.sh frontend:typecheck
./scripts/dev.sh frontend:build
```

Backend tests force safe local settings (`VECTOR_STORE=memory`, `EMBEDDING_PROVIDER=hash`, `LLM_PROVIDER=deterministic`).

Deeper notes on chunking and retrieval: [docs/CHUNKING_AND_RETRIEVAL.md](docs/CHUNKING_AND_RETRIEVAL.md).

## Python version

- Docker image targets **Python 3.11**.  
- Local macOS often has **3.9+**; that works for development. Prefer 3.11+ when you can (`brew install python@3.11`).

## Security (prototype)

Already in place:

- Upload size / type checks  
- Filename sanitization  
- Company ID validation  
- Crawl SSRF protections + optional domain allowlist  
- CORS allowlist  
- Company-scoped vector filters  
- Documents treated as untrusted evidence in prompts  

Not production-ready yet:

- Auth / API keys between frontend and backend  
- Strong rate limiting for public embeds  
- Full OCR for scanned PDFs  

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| Frontend can’t reach API | Confirm backend is on `:8000` and `RAG_API_URL=http://127.0.0.1:8000` |
| Empty / weak answers | Check that documents finished ingesting; try Admin upload status |
| Checklist missing items | Re-upload or reprocess; checklist retrieval needs same-section chunks |
| Qdrant connection errors | Run `./scripts/dev.sh qdrant:up`, or set `VECTOR_STORE=memory` |
| Ollama / HF failures | Switch to `LLM_PROVIDER=deterministic` for local testing |
| Native package / iCloud issues | Prefer a non-iCloud project path; reinstall `backend/.venv` and `frontend/node_modules` |

## Next improvements

1. Auth between frontend and backend  
2. Optional Supabase / pgvector instead of Qdrant  
3. OCR for scanned PDFs  
4. CI for backend + frontend checks  
