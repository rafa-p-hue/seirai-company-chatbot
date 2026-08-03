const API_BASE =
  process.env.NEXT_PUBLIC_RAG_API_URL?.replace(/\/$/, "") || "/rag-api";

export type BackendDocument = {
  document_id: string;
  company_id: string;
  document_name: string;
  source_type:
    | "pdf"
    | "website"
    | "html"
    | "docx"
    | "markdown"
    | "csv"
    | "pptx";
  source_url?: string | null;
  file_type?: string | null;
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
  subsection_title?: string | null;
  content_type?: string | null;
  chunk_index: number;
  content: string;
  content_hash: string;
  source_type:
    | "pdf"
    | "website"
    | "html"
    | "docx"
    | "markdown"
    | "csv"
    | "pptx";
  source_url?: string | null;
};

export type UploadResponse = {
  document: BackendDocument;
  pages: Array<{ page_number: number; text: string }>;
  chunks: BackendChunk[];
  message: string;
};

export type StoredChunkView = {
  chunk_id: string;
  document_id: string;
  document_name: string;
  page_number?: number | null;
  section_title?: string | null;
  subsection_title?: string | null;
  content_type?: string | null;
  token_count: number;
  content: string;
  chunk_index: number;
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
  diagnostics?: Record<string, unknown> | null;
};

export type RetrieveResponse = {
  results: Array<{
    content: string;
    document_name: string;
    page_number?: number | null;
    source_url?: string | null;
    score: number;
    section_title?: string | null;
    subsection_title?: string | null;
    content_type?: string | null;
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
  resolved_query?: string | null;
  subject_name?: string | null;
  inspection?: Record<string, unknown> | null;
};

export type ChatSessionSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type ChatCitation = {
  number?: number;
  document_name?: string;
  page_number?: number | null;
  source_url?: string | null;
  source_type?: "pdf" | "website" | null;
  [key: string]: unknown;
};

export type ChatSessionMessage = {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  citations: ChatCitation[];
  created_at: string;
};

export type ChatTurn = {
  user: ChatSessionMessage;
  assistant: ChatSessionMessage;
};

export type ChatSessionDetail = ChatSessionSummary & {
  messages: ChatSessionMessage[];
};

export type DocumentScope = "chat" | "company";

export type ChatAttachment = {
  id: string;
  session_id: string;
  document_id: string;
  filename: string;
  document_name?: string;
  status: string;
  content_type?: string | null;
  size_bytes?: number | null;
  created_at?: string;
};

export type UploadPdfOptions = {
  sessionId?: string;
  documentScope?: DocumentScope;
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
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(`${API_BASE}/health`, {
      cache: "no-store",
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export async function listDocuments(companyId: string) {
  const response = await fetch(
    `${API_BASE}/api/documents?company_id=${encodeURIComponent(companyId)}`,
    { cache: "no-store" },
  );
  return parseJson<{ documents: BackendDocument[] }>(response);
}

export async function uploadDocument(
  companyId: string,
  file: File,
  options?: UploadPdfOptions,
) {
  const body = new FormData();
  body.append("company_id", companyId);
  body.append("file", file);
  if (options?.sessionId) {
    body.append("session_id", options.sessionId);
  }
  if (options?.documentScope) {
    body.append("document_scope", options.documentScope);
  }
  const response = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body,
  });
  return parseJson<UploadResponse>(response);
}

/** @deprecated Prefer uploadDocument — kept for callers that still say PDF. */
export async function uploadPdf(
  companyId: string,
  file: File,
  options?: UploadPdfOptions,
) {
  return uploadDocument(companyId, file, options);
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

export async function listDocumentChunks(companyId: string, documentId: string) {
  const response = await fetch(
    `${API_BASE}/api/documents/${encodeURIComponent(documentId)}/chunks?company_id=${encodeURIComponent(companyId)}`,
    { cache: "no-store" },
  );
  return parseJson<{
    document_id: string;
    company_id: string;
    chunks: StoredChunkView[];
  }>(response);
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

/** Persistent SQLite-backed chat sessions. */
export async function listChatSessions() {
  const response = await fetch(`${API_BASE}/api/chat/sessions`, {
    cache: "no-store",
  });
  return parseJson<ChatSessionSummary[]>(response);
}

export async function createChatSession(title = "New Chat") {
  const response = await fetch(`${API_BASE}/api/chat/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return parseJson<ChatSessionSummary>(response);
}

export async function getChatSession(sessionId: string) {
  const response = await fetch(
    `${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}`,
    { cache: "no-store" },
  );
  return parseJson<ChatSessionDetail>(response);
}

export async function renameChatSession(sessionId: string, title: string) {
  const response = await fetch(
    `${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    },
  );
  return parseJson<ChatSessionSummary>(response);
}

export async function deleteChatSession(sessionId: string) {
  const response = await fetch(
    `${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}`,
    { method: "DELETE" },
  );
  if (!response.ok && response.status !== 204) {
    await parseJson(response);
  }
}

export async function sendChatSessionMessage(input: {
  sessionId: string;
  content: string;
  companyId?: string;
  topK?: number;
}) {
  const response = await fetch(
    `${API_BASE}/api/chat/sessions/${encodeURIComponent(input.sessionId)}/messages`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: input.content,
        company_id: input.companyId ?? "seirai",
        top_k: input.topK ?? 5,
      }),
    },
  );
  return parseJson<ChatTurn>(response);
}

/** Optional session attachment endpoints (may 404 until backend lands). */
export async function listSessionAttachments(sessionId: string) {
  const response = await fetch(
    `${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/attachments`,
    { cache: "no-store" },
  );
  return parseJson<ChatAttachment[] | { attachments: ChatAttachment[] }>(
    response,
  );
}

export async function uploadSessionAttachment(
  sessionId: string,
  file: File,
  companyId = "seirai",
) {
  const body = new FormData();
  body.append("file", file);
  body.append("company_id", companyId);
  body.append("document_scope", "chat");
  const response = await fetch(
    `${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/attachments`,
    { method: "POST", body },
  );
  return parseJson<ChatAttachment | UploadResponse>(response);
}

export async function deleteSessionAttachment(
  sessionId: string,
  attachmentId: string,
) {
  const response = await fetch(
    `${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/attachments/${encodeURIComponent(attachmentId)}`,
    { method: "DELETE" },
  );
  if (!response.ok && response.status !== 204) {
    await parseJson(response);
  }
}

export { API_BASE };
