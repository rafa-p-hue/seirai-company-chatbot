#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cmd="${1:-}"
case "$cmd" in
  frontend:install)
    cd "$ROOT/frontend" && npm install
    ;;
  frontend:dev)
    cd "$ROOT/frontend" && npm run dev
    ;;
  frontend:build)
    cd "$ROOT/frontend" && npm run build
    ;;
  frontend:lint)
    cd "$ROOT/frontend" && npm run lint
    ;;
  frontend:typecheck)
    cd "$ROOT/frontend" && npx tsc --noEmit
    ;;
  frontend:test)
    cd "$ROOT/frontend" && npm test
    ;;
  backend:venv)
    cd "$ROOT/backend"
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    ;;
  backend:test)
    cd "$ROOT/backend"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    VECTOR_STORE=memory EMBEDDING_PROVIDER=hash LLM_PROVIDER=deterministic APP_ENV=test pytest -q
    ;;
  backend:dev)
    cd "$ROOT/backend"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    cp -n .env.example .env || true
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ;;
  qdrant:up)
    cd "$ROOT" && docker compose up -d qdrant
    ;;
  qdrant:down)
    cd "$ROOT" && docker compose down
    ;;
  *)
    echo "Usage: $0 {frontend:install|frontend:dev|frontend:build|frontend:lint|frontend:typecheck|frontend:test|backend:venv|backend:dev|backend:test|qdrant:up|qdrant:down}"
    exit 1
    ;;
esac
