const API_BASE =
  process.env.NEXT_PUBLIC_RAG_API_URL?.replace(/\/$/, "") || "/rag-api";

export type BackendDocument = {
  document_id: string;
  company_id: string;
  document_name: string;
  source_type: "pdf" | "website";
  source_url?: string | null;
  page_count: number;
  chunk_count: number;
  embedding_count: number;
  embedding_model?: string | null;
  embedding_dimension?: number | null;
  status: string;
  uploaded_at: string;
  errors?: string[];
};

export type BackendChunk = {
  chunk_id: string;
  company_id: string;
  document_id: string;
  document_name: string;
  page_number?: number | null;
  section_title?: string | null;
  chunk_index: number;
  content: string;
  content_hash: string;
  source_type: "pdf" | "website";
  source_url?: string | null;
};

export type UploadResponse = {
  document: BackendDocument;
  pages: Array<{ page_number: number; text: string }>;
  chunks: BackendChunk[];
  message: string;
};

export type ChatResponse = {
  answer: string;
  sources: Array<{
    number: number;
    document_name: string;
    page_number?: number | null;
    source_url?: string | null;
    source_type?: "pdf" | "website" | null;
  }>;
  conversation_id?: string | null;
};

export type RetrieveResponse = {
  results: Array<{
    content: string;
    document_name: string;
    page_number?: number | null;
    source_url?: string | null;
    score: number;
    record_type?: string | null;
    title?: string | null;
    organization?: string | null;
    person_name?: string | null;
    diagnostics?: Record<string, unknown> | null;
  }>;
  original_query?: string | null;
  expanded_query?: string | null;
  query_type?: string | null;
  expanded_terms?: string[];
};

async function parseJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const raw =
      (payload as { error?: unknown; detail?: unknown }).error ??
      (payload as { detail?: unknown }).detail ??
      "Request failed.";
    const message =
      typeof raw === "string"
        ? raw
        : Array.isArray(raw)
          ? raw
              .map((item) =>
                typeof item === "object" && item && "msg" in item
                  ? String((item as { msg: unknown }).msg)
                  : JSON.stringify(item),
              )
              .join("; ")
          : "Request failed.";
    throw new Error(message);
  }
  return payload as T;
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

export async function listDocuments(companyId: string) {
  const response = await fetch(
    `${API_BASE}/api/documents?company_id=${encodeURIComponent(companyId)}`,
    { cache: "no-store" },
  );
  return parseJson<{ documents: BackendDocument[] }>(response);
}

export async function uploadPdf(companyId: string, file: File) {
  const body = new FormData();
  body.append("company_id", companyId);
  body.append("file", file);
  const response = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body,
  });
  return parseJson<UploadResponse>(response);
}

export async function ingestWebsite(
  companyId: string,
  url: string,
  maxPages = 20,
) {
  const response = await fetch(`${API_BASE}/api/websites/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      company_id: companyId,
      url,
      max_pages: maxPages,
    }),
  });
  return parseJson<UploadResponse>(response);
}

export async function deleteDocument(companyId: string, documentId: string) {
  const response = await fetch(
    `${API_BASE}/api/documents/${encodeURIComponent(documentId)}?company_id=${encodeURIComponent(companyId)}`,
    { method: "DELETE" },
  );
  return parseJson<{ deleted_chunks: number; document_id: string }>(response);
}

export async function reprocessDocument(companyId: string, documentId: string) {
  const response = await fetch(
    `${API_BASE}/api/documents/${encodeURIComponent(documentId)}/reprocess?company_id=${encodeURIComponent(companyId)}`,
    { method: "POST" },
  );
  return parseJson<UploadResponse>(response);
}

export async function retrieveEvidence(
  companyId: string,
  question: string,
  topK = 5,
) {
  const response = await fetch(`${API_BASE}/api/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      company_id: companyId,
      question,
      top_k: topK,
    }),
  });
  return parseJson<RetrieveResponse>(response);
}

export async function chatWithBackend(input: {
  companyId: string;
  question: string;
  conversationId?: string;
  history?: Array<{ role: string; content: string }>;
}) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      company_id: input.companyId,
      question: input.question,
      conversation_id: input.conversationId,
      history: input.history ?? [],
    }),
  });
  return parseJson<ChatResponse>(response);
}

export { API_BASE };
