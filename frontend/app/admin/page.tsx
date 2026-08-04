"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

import {
  checkBackendHealth,
  deleteDocument,
  ingestWebsite,
  listDocumentChunks,
  listDocuments,
  reprocessDocument,
  retrieveEvidence,
  uploadDocument,
  type BackendDocument,
  type RetrieveResponse,
  type StoredChunkView,
  type UploadResponse,
} from "@/lib/rag-api";
import {
  fileTypeLabel,
  isSupportedUploadFile,
  SUPPORTED_FORMAT_LABEL,
  UPLOAD_ACCEPT_EXTENSIONS,
} from "@/lib/supported-formats";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

type AdminUploadItem = {
  localId: string;
  filename: string;
  fileType: string;
  status:
    | "selected"
    | "uploading"
    | "parsing"
    | "chunking"
    | "embedding"
    | "ready"
    | "failed";
  error?: string;
  result?: UploadResponse;
  sourceFile?: File;
};

export default function AdminPage() {
  const [companyId, setCompanyId] = useState("seirai");
  const [backendReady, setBackendReady] = useState(false);
  const [documents, setDocuments] = useState<BackendDocument[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<AdminUploadItem[]>([]);
  const [websiteUrl, setWebsiteUrl] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [retrievalQuestion, setRetrievalQuestion] = useState("");
  const [retrieval, setRetrieval] = useState<RetrieveResponse | null>(null);
  const [storedChunks, setStoredChunks] = useState<StoredChunkView[] | null>(null);
  const [chunksDocId, setChunksDocId] = useState<string | null>(null);

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

  function queueFiles(fileList: FileList | null) {
    if (!fileList?.length) return;
    const next: AdminUploadItem[] = [];
    const rejected: string[] = [];
    for (const file of Array.from(fileList)) {
      if (!isSupportedUploadFile(file)) {
        rejected.push(file.name);
        continue;
      }
      if (file.size > MAX_FILE_SIZE_BYTES) {
        rejected.push(`${file.name} (too large)`);
        continue;
      }
      next.push({
        localId: crypto.randomUUID(),
        filename: file.name,
        fileType: fileTypeLabel(file.name, file.type),
        status: "selected",
        sourceFile: file,
      });
    }
    setSelectedFiles((current) => [...current, ...next]);
    setError(
      rejected.length
        ? `Skipped unsupported/oversized files. Supported: ${SUPPORTED_FORMAT_LABEL}.`
        : "",
    );
  }

  async function uploadOne(item: AdminUploadItem): Promise<AdminUploadItem> {
    if (!item.sourceFile) {
      return { ...item, status: "failed", error: "Original file unavailable." };
    }
    const patch = (status: AdminUploadItem["status"]) => {
      setSelectedFiles((current) =>
        current.map((row) =>
          row.localId === item.localId ? { ...row, status, error: undefined } : row,
        ),
      );
    };
    try {
      patch("uploading");
      patch("parsing");
      patch("chunking");
      const result = await uploadDocument(companyId, item.sourceFile);
      patch("embedding");
      const ready: AdminUploadItem = {
        ...item,
        status: "ready",
        result,
        sourceFile: undefined,
      };
      setSelectedFiles((current) =>
        current.map((row) => (row.localId === item.localId ? ready : row)),
      );
      return ready;
    } catch (err) {
      const failed: AdminUploadItem = {
        ...item,
        status: "failed",
        error: err instanceof Error ? err.message : "Upload failed.",
      };
      setSelectedFiles((current) =>
        current.map((row) => (row.localId === item.localId ? failed : row)),
      );
      return failed;
    }
  }

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    setError("");
    setUploadResult(null);
    const pending = selectedFiles.filter(
      (item) => item.status === "selected" || item.status === "failed",
    );
    if (pending.length === 0) {
      setError("Choose one or more supported files first.");
      return;
    }
    setBusy(true);
    try {
      const results = await Promise.all(pending.map((item) => uploadOne(item)));
      const lastReady = [...results].reverse().find((item) => item.status === "ready");
      if (lastReady?.result) setUploadResult(lastReady.result);
      await refresh();
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
              Upload {SUPPORTED_FORMAT_LABEL} files or crawl approved websites
              into Qdrant via the FastAPI RAG backend.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Link
              href="/chat"
              className="rounded-full bg-slate-950 px-4 py-2 text-sm font-semibold text-white"
            >
              Open Chat
            </Link>
            <Link
              href={`/embed/seirai?company_id=${encodeURIComponent(companyId)}`}
              className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700"
            >
              Embed widget
            </Link>
          </div>
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
            <h2 className="text-xl font-semibold">Upload documents</h2>
            <p className="mt-2 text-xs text-slate-500">
              Select or drag multiple files. Supported: {SUPPORTED_FORMAT_LABEL}.
              Each file is validated and processed independently. Need a test
              file?{" "}
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
                Choose files
              </span>
              <span className="text-xs text-slate-500">
                {SUPPORTED_FORMAT_LABEL} · max 10 MB each
              </span>
              <input
                id="pdf-upload"
                type="file"
                multiple
                accept={UPLOAD_ACCEPT_EXTENSIONS}
                className="sr-only"
                onChange={(event) => {
                  queueFiles(event.target.files);
                  event.target.value = "";
                }}
              />
            </label>
            {selectedFiles.length > 0 ? (
              <ul className="mt-4 grid gap-2">
                {selectedFiles.map((file) => (
                  <li
                    key={file.localId}
                    className={`rounded-xl border px-3 py-2 text-xs ${
                      file.status === "failed"
                        ? "border-amber-200 bg-amber-50"
                        : "border-slate-200 bg-slate-50"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate font-medium">{file.filename}</p>
                        <p className="mt-0.5 text-slate-500">
                          {file.fileType} · {file.status}
                          {file.error ? ` · ${file.error}` : ""}
                        </p>
                      </div>
                      <button
                        type="button"
                        className="shrink-0 text-slate-500 hover:text-slate-800"
                        onClick={() =>
                          setSelectedFiles((current) =>
                            current.filter((row) => row.localId !== file.localId),
                          )
                        }
                      >
                        Remove
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            ) : null}
            <button
              type="submit"
              disabled={
                busy ||
                !backendReady ||
                !selectedFiles.some(
                  (item) => item.status === "selected" || item.status === "failed",
                )
              }
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
                        void listDocumentChunks(companyId, document.document_id)
                          .then((data) => {
                            setChunksDocId(document.document_id);
                            setStoredChunks(data.chunks);
                          })
                          .catch((err) =>
                            setError(
                              err instanceof Error ? err.message : "Chunk list failed.",
                            ),
                          )
                      }
                    >
                      View chunks
                    </button>
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
                  <span className="font-semibold">Resolved:</span>{" "}
                  {retrieval.resolved_query || "—"}
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
              {retrieval.inspection ? (
                <details className="rounded-2xl border border-slate-200 bg-white p-3 text-xs">
                  <summary className="cursor-pointer font-semibold">
                    Retrieval inspection (dev)
                  </summary>
                  <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap text-[11px] text-slate-600">
                    {JSON.stringify(retrieval.inspection, null, 2)}
                  </pre>
                </details>
              ) : null}
              {retrieval.results.map((item, index) => (
                <div key={`${item.document_name}-${index}`} className="rounded-2xl bg-slate-50 p-3 text-sm">
                  <p className="text-xs text-slate-500">
                    score {item.score.toFixed(4)} · {item.content_type || item.record_type || "unknown"} ·{" "}
                    {item.section_title || item.title || "untitled"}
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

        {storedChunks && chunksDocId ? (
          <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-xl font-semibold">Stored chunks</h2>
              <button
                type="button"
                className="text-xs font-semibold text-slate-500"
                onClick={() => {
                  setStoredChunks(null);
                  setChunksDocId(null);
                }}
              >
                Close
              </button>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              document {chunksDocId} · {storedChunks.length} chunks
            </p>
            <div className="mt-4 max-h-[32rem] space-y-3 overflow-y-auto">
              {storedChunks.map((chunk) => (
                <div key={chunk.chunk_id} className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                  <p className="text-xs text-slate-500">
                    {chunk.chunk_id.slice(0, 8)} · #{chunk.chunk_index + 1} · page{" "}
                    {chunk.page_number ?? "n/a"} · {chunk.content_type || "unknown"} ·{" "}
                    {chunk.token_count} tokens
                  </p>
                  <p className="mt-1 text-xs font-semibold text-slate-700">
                    {chunk.section_title || "no section"}
                    {chunk.subsection_title ? ` / ${chunk.subsection_title}` : ""}
                  </p>
                  <p className="mt-2 whitespace-pre-wrap text-xs">{chunk.content}</p>
                </div>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </main>
  );
}
