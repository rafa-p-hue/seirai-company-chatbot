"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

import {
  checkBackendHealth,
  deleteDocument,
  ingestWebsite,
  listDocuments,
  reprocessDocument,
  retrieveEvidence,
  uploadPdf,
  type BackendDocument,
  type RetrieveResponse,
  type UploadResponse,
} from "@/lib/rag-api";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

export default function AdminPage() {
  const [companyId, setCompanyId] = useState("seirai");
  const [backendReady, setBackendReady] = useState(false);
  const [documents, setDocuments] = useState<BackendDocument[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [websiteUrl, setWebsiteUrl] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [retrievalQuestion, setRetrievalQuestion] = useState("");
  const [retrieval, setRetrieval] = useState<RetrieveResponse | null>(null);

  async function refresh() {
    const healthy = await checkBackendHealth();
    setBackendReady(healthy);
    if (!healthy) {
      setDocuments([]);
      return;
    }
    const data = await listDocuments(companyId);
    setDocuments(data.documents);
  }

  useEffect(() => {
    let cancelled = false;

    async function tick() {
      try {
        await refresh();
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load documents.");
        }
      }
    }

    void tick();
    const timer = window.setInterval(() => {
      void tick();
    }, 3000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
    // Intentionally refresh when company changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [companyId]);

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    setError("");
    setUploadResult(null);
    if (!selectedFile) {
      setError("Choose a PDF file first.");
      return;
    }
    if (selectedFile.size > MAX_FILE_SIZE_BYTES) {
      setError("PDF must be 10 MB or smaller.");
      return;
    }
    setBusy(true);
    try {
      const result = await uploadPdf(companyId, selectedFile);
      setUploadResult(result);
      setSelectedFile(null);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleWebsite(event: FormEvent) {
    event.preventDefault();
    setError("");
    setUploadResult(null);
    setBusy(true);
    try {
      const result = await ingestWebsite(companyId, websiteUrl, 20);
      setUploadResult(result);
      setWebsiteUrl("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Website ingest failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleRetrieve(event: FormEvent) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const result = await retrieveEvidence(companyId, retrievalQuestion, 5);
      setRetrieval(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retrieval failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-8 text-slate-950 sm:px-8">
      <div className="mx-auto max-w-5xl space-y-8">
        <header className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-cyan-700">
              Knowledge Base
            </p>
            <h1 className="text-3xl font-semibold">Admin Ingestion Console</h1>
            <p className="mt-2 text-sm text-slate-600">
              Upload PDFs or crawl approved websites into Qdrant via the FastAPI
              RAG backend.
            </p>
          </div>
          <Link
            href={`/embed/seirai?company_id=${encodeURIComponent(companyId)}`}
            className="rounded-full bg-slate-950 px-4 py-2 text-sm font-semibold text-white"
          >
            Open Chatbot
          </Link>
        </header>

        <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <label className="text-sm font-semibold" htmlFor="company-id">
            Company ID
          </label>
          <input
            id="company-id"
            value={companyId}
            onChange={(event) => setCompanyId(event.target.value.toLowerCase())}
            className="mt-2 w-full rounded-2xl border border-slate-300 px-4 py-3 text-sm outline-none focus:border-cyan-700 focus:ring-2 focus:ring-cyan-700/20"
          />
          <p className="mt-3 text-sm">
            Backend:{" "}
            <span className={backendReady ? "text-emerald-700" : "text-amber-700"}>
              {backendReady ? "online" : "offline"}
            </span>
          </p>
        </section>

        {error ? (
          <div className="rounded-3xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            {error}
          </div>
        ) : null}

        <section className="grid gap-6 lg:grid-cols-2">
          <form
            onSubmit={handleUpload}
            className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm"
          >
            <h2 className="text-xl font-semibold">Upload PDF</h2>
            <p className="mt-2 text-xs text-slate-500">
              Choose a text-based PDF (selectable text). Scanned image PDFs are
              not supported yet. Need a test file?{" "}
              <a
                href="/sample-company.pdf"
                className="font-semibold text-cyan-800 underline"
                download
              >
                Download sample-company.pdf
              </a>
            </p>
            <label
              htmlFor="pdf-upload"
              className="mt-4 flex cursor-pointer flex-col items-start gap-2 rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-4 py-6 text-sm hover:border-cyan-700"
            >
              <span className="font-semibold text-slate-800">
                {selectedFile ? "Change PDF" : "Choose PDF file"}
              </span>
              <span className="text-xs text-slate-500">
                {selectedFile
                  ? `${selectedFile.name} (${Math.round(selectedFile.size / 1024)} KB)`
                  : "PDF only · max 10 MB"}
              </span>
              <input
                id="pdf-upload"
                type="file"
                accept="application/pdf,.pdf"
                className="sr-only"
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  setSelectedFile(file);
                  setError("");
                }}
              />
            </label>
            <button
              type="submit"
              disabled={busy || !selectedFile || !backendReady}
              className="mt-4 rounded-full bg-cyan-700 px-5 py-3 text-sm font-semibold text-white disabled:bg-slate-300"
            >
              {busy ? "Uploading..." : "Upload & Embed"}
            </button>
            {!backendReady ? (
              <p className="mt-2 text-xs text-amber-700">
                Backend is offline. Start FastAPI on port 8000, then refresh.
              </p>
            ) : null}
          </form>

          <form
            onSubmit={handleWebsite}
            className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm"
          >
            <h2 className="text-xl font-semibold">Ingest Website</h2>
            <input
              type="url"
              value={websiteUrl}
              onChange={(event) => setWebsiteUrl(event.target.value)}
              placeholder="https://example.com"
              className="mt-4 w-full rounded-2xl border border-slate-300 px-4 py-3 text-sm outline-none focus:border-cyan-700 focus:ring-2 focus:ring-cyan-700/20"
            />
            <button
              type="submit"
              disabled={busy || !websiteUrl.trim()}
              className="mt-4 rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white disabled:bg-slate-300"
            >
              {busy ? "Working..." : "Crawl & Embed"}
            </button>
          </form>
        </section>

        {uploadResult ? (
          <section className="rounded-3xl border border-cyan-100 bg-cyan-50 p-5 text-sm text-cyan-950">
            <p className="font-semibold">{uploadResult.message}</p>
            <p className="mt-2">
              {uploadResult.document.document_name} · {uploadResult.document.chunk_count}{" "}
              chunks · {uploadResult.document.embedding_count} embeddings ·{" "}
              {uploadResult.document.embedding_model} (
              {uploadResult.document.embedding_dimension}d)
            </p>
            <details className="mt-4">
              <summary className="cursor-pointer font-semibold">Extracted pages</summary>
              <div className="mt-2 max-h-64 space-y-3 overflow-y-auto">
                {uploadResult.pages.map((page) => (
                  <pre key={page.page_number} className="whitespace-pre-wrap text-xs">
                    Page {page.page_number}
                    {"\n"}
                    {page.text.slice(0, 1200)}
                  </pre>
                ))}
              </div>
            </details>
            <details className="mt-4">
              <summary className="cursor-pointer font-semibold">Chunks</summary>
              <div className="mt-2 max-h-64 space-y-3 overflow-y-auto">
                {uploadResult.chunks.map((chunk) => (
                  <div key={chunk.chunk_id} className="rounded-2xl bg-white p-3">
                    <p className="text-xs text-slate-500">
                      #{chunk.chunk_index + 1} · page {chunk.page_number ?? "n/a"} ·{" "}
                      {chunk.section_title ?? "no section"}
                    </p>
                    <p className="mt-2 whitespace-pre-wrap text-xs">{chunk.content}</p>
                  </div>
                ))}
              </div>
            </details>
          </section>
        ) : null}

        <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-xl font-semibold">Stored Documents</h2>
          <div className="mt-4 space-y-3">
            {documents.length === 0 ? (
              <p className="text-sm text-slate-500">No documents yet.</p>
            ) : (
              documents.map((document) => (
                <div
                  key={document.document_id}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-4"
                >
                  <div>
                    <p className="font-semibold">{document.document_name}</p>
                    <p className="text-xs text-slate-500">
                      {document.source_type} · {document.chunk_count} chunks ·{" "}
                      {document.embedding_count} embeddings · {document.status}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      className="rounded-full border border-slate-300 px-3 py-2 text-xs font-semibold"
                      onClick={() =>
                        void reprocessDocument(companyId, document.document_id)
                          .then(refresh)
                          .catch((err) =>
                            setError(
                              err instanceof Error ? err.message : "Reprocess failed.",
                            ),
                          )
                      }
                    >
                      Reprocess
                    </button>
                    <button
                      type="button"
                      className="rounded-full border border-red-300 px-3 py-2 text-xs font-semibold text-red-700"
                      onClick={() =>
                        void deleteDocument(companyId, document.document_id)
                          .then(refresh)
                          .catch((err) =>
                            setError(
                              err instanceof Error ? err.message : "Delete failed.",
                            ),
                          )
                      }
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-xl font-semibold">Retrieval Tester</h2>
          <form onSubmit={handleRetrieve} className="mt-4 flex gap-2">
            <input
              value={retrievalQuestion}
              onChange={(event) => setRetrievalQuestion(event.target.value)}
              placeholder="What services does the company provide?"
              className="min-h-11 flex-1 rounded-2xl border border-slate-300 px-4 text-sm outline-none focus:border-cyan-700 focus:ring-2 focus:ring-cyan-700/20"
            />
            <button
              type="submit"
              disabled={busy || !retrievalQuestion.trim()}
              className="rounded-2xl bg-slate-950 px-5 text-sm font-semibold text-white disabled:bg-slate-300"
            >
              Test
            </button>
          </form>
          {retrieval ? (
            <div className="mt-4 space-y-3">
              <div className="rounded-2xl border border-cyan-100 bg-cyan-50 p-3 text-xs text-cyan-950">
                <p>
                  <span className="font-semibold">Original:</span>{" "}
                  {retrieval.original_query || retrievalQuestion}
                </p>
                <p className="mt-1">
                  <span className="font-semibold">Expanded:</span>{" "}
                  {retrieval.expanded_query || "—"}
                </p>
                <p className="mt-1">
                  <span className="font-semibold">Query type:</span>{" "}
                  {retrieval.query_type || "—"}
                </p>
                {retrieval.expanded_terms?.length ? (
                  <p className="mt-1">
                    <span className="font-semibold">Terms:</span>{" "}
                    {retrieval.expanded_terms.join(", ")}
                  </p>
                ) : null}
              </div>
              {retrieval.results.map((item, index) => (
                <div key={`${item.document_name}-${index}`} className="rounded-2xl bg-slate-50 p-3 text-sm">
                  <p className="text-xs text-slate-500">
                    score {item.score.toFixed(4)} · {item.record_type || "unknown"} ·{" "}
                    {item.title || "untitled"}
                    {item.organization ? ` @ ${item.organization}` : ""}
                    {item.page_number ? ` · page ${item.page_number}` : ""}
                  </p>
                  <p className="mt-2 whitespace-pre-wrap">{item.content}</p>
                  {item.diagnostics ? (
                    <pre className="mt-2 overflow-x-auto text-xs text-slate-500">
                      {JSON.stringify(item.diagnostics, null, 2)}
                    </pre>
                  ) : null}
                </div>
              ))}
            </div>
          ) : null}
        </section>
      </div>
    </main>
  );
}
