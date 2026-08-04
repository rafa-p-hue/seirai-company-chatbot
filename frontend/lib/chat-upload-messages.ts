/** Types and helpers for chat-log upload messages (frontend timeline). */

export type UploadMessageFile = {
  localId: string;
  filename: string;
  fileType: string;
  status: "ready" | "failed" | "error" | "processing";
  documentId?: string;
  attachmentId?: string;
  error?: string;
};

export type ChatUploadMessage = {
  id: string;
  kind: "upload";
  session_id: string;
  created_at: string;
  files: UploadMessageFile[];
};

export type UploadCommitFile = {
  localId: string;
  filename: string;
  fileType: string;
  status: string;
  documentId?: string;
  attachmentId?: string;
  error?: string;
};

const storageKey = (sessionId: string) =>
  `seirai.chat.uploadMessages.${sessionId}`;

export function loadStoredUploadMessages(
  sessionId: string,
): ChatUploadMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(storageKey(sessionId));
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatUploadMessage[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveStoredUploadMessages(
  sessionId: string,
  messages: ChatUploadMessage[],
): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      storageKey(sessionId),
      JSON.stringify(messages),
    );
  } catch {
    // ignore quota / private mode
  }
}

export function clearStoredUploadMessages(sessionId: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(storageKey(sessionId));
  } catch {
    // ignore
  }
}

export function summarizeUploadMessage(message: ChatUploadMessage): {
  title: string;
  subtitle: string;
  readyCount: number;
  failedCount: number;
} {
  const readyCount = message.files.filter((f) => f.status === "ready").length;
  const failedCount = message.files.filter(
    (f) => f.status === "failed" || f.status === "error",
  ).length;
  const total = message.files.length;
  const title =
    total === 1
      ? `Uploaded ${message.files[0]?.filename || "1 document"}`
      : `Uploaded ${total} documents`;
  let subtitle: string;
  if (failedCount > 0 && readyCount > 0) {
    subtitle = `${readyCount} ready · ${failedCount} failed`;
  } else if (failedCount > 0) {
    subtitle = failedCount === 1 ? "1 failed" : `${failedCount} failed`;
  } else {
    subtitle =
      readyCount === 1 ? "1 document ready" : `${readyCount} documents ready`;
  }
  return { title, subtitle, readyCount, failedCount };
}

function toUploadMessageFile(file: UploadCommitFile): UploadMessageFile {
  return {
    localId: file.localId,
    filename: file.filename,
    fileType: file.fileType,
    status:
      file.status === "ready"
        ? "ready"
        : file.status === "failed" || file.status === "error"
          ? "failed"
          : "processing",
    documentId: file.documentId,
    attachmentId: file.attachmentId,
    error: file.error,
  };
}

/**
 * After a successful batch upload: move ready files into sessionDocuments,
 * append one grouped chat-log upload message, and clear successful pending items.
 * Failed pending items remain near the composer for retry/remove.
 */
export function applySuccessfulUploadBatch<T extends UploadCommitFile>(options: {
  sessionId: string;
  pending: T[];
  sessionDocuments: T[];
  uploadMessages: ChatUploadMessage[];
  ready: T[];
  batchLocalIds: string[];
  messageId?: string;
  createdAt?: string;
}): {
  pending: T[];
  sessionDocuments: T[];
  uploadMessages: ChatUploadMessage[];
  composerVisibleCount: number;
} {
  const {
    sessionId,
    pending,
    sessionDocuments,
    uploadMessages,
    ready,
    batchLocalIds,
  } = options;

  let nextSessionDocs = sessionDocuments;
  let nextUploadMessages = uploadMessages;

  if (ready.length > 0) {
    const existingIds = new Set(
      sessionDocuments.map((doc) => doc.documentId).filter(Boolean),
    );
    const additions = ready.filter(
      (file) => !file.documentId || !existingIds.has(file.documentId),
    );
    nextSessionDocs = [...sessionDocuments, ...additions];

    const message: ChatUploadMessage = {
      id: options.messageId ?? crypto.randomUUID(),
      kind: "upload",
      session_id: sessionId,
      created_at: options.createdAt ?? new Date().toISOString(),
      files: ready.map(toUploadMessageFile),
    };
    nextUploadMessages = [...uploadMessages, message];
  }

  const nextPending = pending.filter((file) => {
    if (!batchLocalIds.includes(file.localId)) return true;
    return file.status === "failed" || file.status === "error";
  });

  return {
    pending: nextPending,
    sessionDocuments: nextSessionDocs,
    uploadMessages: nextUploadMessages,
    composerVisibleCount: nextPending.length,
  };
}
