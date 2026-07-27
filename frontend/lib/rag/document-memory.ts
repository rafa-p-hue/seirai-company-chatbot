import type {
  DocumentProcessingStatus,
  PublicDocumentStatus,
  StoredDocument,
} from "./document-types";

type DocumentMemory = {
  activeDocument: StoredDocument | null;
};

declare global {
  // This keeps memory stable across module reloads during local development.
  // The document is still lost when the server process restarts.
  // This is not appropriate for production; Supabase and pgvector will replace
  // it when the app gains permanent storage and semantic vector search.
  var documentChatbotMemory: DocumentMemory | undefined;
}

const memory: DocumentMemory = globalThis.documentChatbotMemory ?? {
  activeDocument: null,
};

globalThis.documentChatbotMemory = memory;

export function saveDocument(document: StoredDocument): void {
  memory.activeDocument = document;
}

export function createUploadedDocument(input: {
  id: string;
  name: string;
  mimeType: string;
  fileSize: number;
  uploadedAt: string;
}): StoredDocument {
  const document: StoredDocument = {
    id: input.id,
    name: input.name,
    mimeType: input.mimeType,
    fileSize: input.fileSize,
    pageCount: 0,
    characterCount: 0,
    chunkCount: 0,
    embeddingCount: 0,
    status: "uploaded",
    progress: 0,
    message: "Uploading document",
    uploadedAt: input.uploadedAt,
    chunks: [],
    records: [],
    rejectedChunks: [],
    inspection: {
      rawSections: [],
      cleanedSections: [],
      removedRepeatedLines: [],
      detectedHeadings: [],
      warnings: [],
      structuredRecords: [],
      rejectedChunks: [],
    },
    processingLog: [
      {
        status: "uploaded",
        message: "Uploading document",
        progress: 0,
        timestamp: input.uploadedAt,
      },
    ],
  };

  saveDocument(document);

  return document;
}

export function getActiveDocument(): StoredDocument | null {
  return memory.activeDocument;
}

export function getPublicDocumentStatus(): PublicDocumentStatus | null {
  const document = getActiveDocument();

  if (!document) {
    return null;
  }

  return {
    id: document.id,
    name: document.name,
    mimeType: document.mimeType,
    fileSize: document.fileSize,
    pageCount: document.pageCount,
    characterCount: document.characterCount,
    chunkCount: document.chunkCount,
    embeddingCount: document.embeddingCount,
    status: document.status,
    progress: document.progress,
    message: document.message,
    error: document.error,
    uploadedAt: document.uploadedAt,
    readyAt: document.readyAt,
    overview: document.overview,
  };
}

export function updateDocumentStatus(input: {
  documentId: string;
  status: DocumentProcessingStatus;
  progress: number;
  message: string;
  error?: string;
  startedAt?: number;
}): StoredDocument | null {
  const document = getActiveDocument();

  if (!document || document.id !== input.documentId) {
    return null;
  }

  document.status = input.status;
  document.progress = input.progress;
  document.message = input.message;
  document.error = input.error;

  if (input.status === "ready") {
    document.readyAt = new Date().toISOString();
  }

  document.processingLog.push({
    status: input.status,
    message: input.message,
    progress: input.progress,
    timestamp: new Date().toISOString(),
    durationMs: input.startedAt ? Date.now() - input.startedAt : undefined,
  });

  return document;
}

export function updateActiveDocument(
  documentId: string,
  update: (document: StoredDocument) => void,
): StoredDocument | null {
  const document = getActiveDocument();

  if (!document || document.id !== documentId) {
    return null;
  }

  update(document);

  return document;
}

export function clearDocument(): void {
  memory.activeDocument = null;
}
