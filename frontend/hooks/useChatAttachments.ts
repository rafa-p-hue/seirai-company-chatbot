"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  applySuccessfulUploadBatch,
  clearStoredUploadMessages,
  loadStoredUploadMessages,
  saveStoredUploadMessages,
  type ChatUploadMessage,
  type UploadMessageFile,
} from "@/lib/chat-upload-messages";
import {
  deleteSessionAttachment,
  listSessionAttachments,
  uploadDocument,
  uploadSessionAttachment,
  type ChatAttachment,
  type UploadResponse,
} from "@/lib/rag-api";
import {
  fileTypeLabel,
  isSupportedUploadFile,
  SUPPORTED_FORMAT_LABEL,
} from "@/lib/supported-formats";

export type LocalChatFile = {
  localId: string;
  filename: string;
  fileType: string;
  status:
    | "selected"
    | "queued"
    | "uploading"
    | "parsing"
    | "chunking"
    | "embedding"
    | "processing"
    | "ready"
    | "failed"
    | "error";
  progress: number;
  error?: string;
  documentId?: string;
  attachmentId?: string;
  contentType?: string;
  sizeBytes?: number;
  createdAt?: string;
  /** Kept for retry after a failed upload. */
  sourceFile?: File;
};

export const MAX_CHAT_FILE_BYTES = 10 * 1024 * 1024;

export function isAcceptedChatFile(file: File): boolean {
  return isSupportedUploadFile(file);
}

function normalizeAttachments(
  payload: ChatAttachment[] | { attachments: ChatAttachment[] },
): ChatAttachment[] {
  return Array.isArray(payload) ? payload : (payload.attachments ?? []);
}

function attachmentToLocal(item: ChatAttachment): LocalChatFile {
  const filename = item.filename || item.document_name || "document.pdf";
  return {
    localId: item.id,
    filename,
    fileType: fileTypeLabel(filename, item.content_type),
    status:
      item.status === "ready" ||
      item.status === "indexed" ||
      item.status === "complete"
        ? "ready"
        : item.status === "failed" || item.status === "error"
          ? "failed"
          : "processing",
    progress: 100,
    documentId: item.document_id,
    attachmentId: item.id,
    contentType: item.content_type ?? undefined,
    sizeBytes: item.size_bytes ?? undefined,
    createdAt: item.created_at,
  };
}

function toUploadMessageFile(file: LocalChatFile): UploadMessageFile {
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

function synthesizeUploadMessage(
  sessionId: string,
  documents: LocalChatFile[],
  createdAt?: string,
): ChatUploadMessage | null {
  const ready = documents.filter((file) => file.status === "ready");
  if (ready.length === 0) return null;
  return {
    id: `upload-synth-${sessionId}`,
    kind: "upload",
    session_id: sessionId,
    created_at: createdAt || new Date().toISOString(),
    files: ready.map(toUploadMessageFile),
  };
}

function isBusyStatus(status: LocalChatFile["status"]): boolean {
  return (
    status === "selected" ||
    status === "queued" ||
    status === "uploading" ||
    status === "parsing" ||
    status === "chunking" ||
    status === "embedding" ||
    status === "processing"
  );
}

export function useChatAttachments(options: {
  sessionId: string | null;
  companyId: string;
  ensureSession: () => Promise<string>;
}) {
  const { sessionId, companyId, ensureSession } = options;
  /** Files selected / in-flight near the composer only. */
  const [pendingAttachments, setPendingAttachments] = useState<LocalChatFile[]>(
    [],
  );
  /** Files already attached to the session (backend). Never shown as composer grid. */
  const [sessionDocuments, setSessionDocuments] = useState<LocalChatFile[]>([]);
  /** Grouped upload events rendered in the chat log. */
  const [uploadMessages, setUploadMessages] = useState<ChatUploadMessage[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [attachmentsSupported, setAttachmentsSupported] = useState<
    boolean | null
  >(null);
  const pendingRef = useRef<LocalChatFile[]>([]);
  const sessionDocsRef = useRef<LocalChatFile[]>([]);
  const uploadMessagesRef = useRef<ChatUploadMessage[]>([]);

  useEffect(() => {
    pendingRef.current = pendingAttachments;
  }, [pendingAttachments]);

  useEffect(() => {
    sessionDocsRef.current = sessionDocuments;
  }, [sessionDocuments]);

  useEffect(() => {
    uploadMessagesRef.current = uploadMessages;
  }, [uploadMessages]);

  const persistUploadMessages = useCallback(
    (sessionKey: string, messages: ChatUploadMessage[]) => {
      setUploadMessages(messages);
      saveStoredUploadMessages(sessionKey, messages);
    },
    [],
  );

  const loadAttachments = useCallback(
    async (id: string) => {
      try {
        const payload = await listSessionAttachments(id);
        setAttachmentsSupported(true);
        const docs = normalizeAttachments(payload).map(attachmentToLocal);
        setSessionDocuments(docs);
        setPendingAttachments([]);
        setUploadError(null);

        const stored = loadStoredUploadMessages(id);
        const docIds = new Set(
          docs.map((doc) => doc.documentId).filter(Boolean) as string[],
        );
        const pruned = stored
          .map((message) => ({
            ...message,
            files: message.files.filter(
              (file) =>
                !file.documentId ||
                docIds.has(file.documentId) ||
                file.status === "failed" ||
                file.status === "error",
            ),
          }))
          .filter((message) => message.files.length > 0);

        if (pruned.length > 0) {
          persistUploadMessages(id, pruned);
        } else {
          const earliest = docs
            .map((doc) => doc.createdAt)
            .filter(Boolean)
            .sort()[0] as string | undefined;
          const synthesized = synthesizeUploadMessage(id, docs, earliest);
          persistUploadMessages(id, synthesized ? [synthesized] : []);
        }
      } catch {
        setAttachmentsSupported(false);
        setSessionDocuments([]);
        setPendingAttachments([]);
        setUploadMessages([]);
      }
    },
    [persistUploadMessages],
  );

  const clearFiles = useCallback(() => {
    setPendingAttachments([]);
    setSessionDocuments([]);
    setUploadMessages([]);
    setUploadError(null);
  }, []);

  const forgetSessionUploads = useCallback((id: string) => {
    clearStoredUploadMessages(id);
  }, []);

  const removeFile = useCallback(
    async (localId: string) => {
      const pendingTarget = pendingRef.current.find(
        (file) => file.localId === localId,
      );
      const sessionTarget = sessionDocsRef.current.find(
        (file) => file.localId === localId,
      );
      const target = pendingTarget || sessionTarget;

      setPendingAttachments((current) =>
        current.filter((file) => file.localId !== localId),
      );
      setSessionDocuments((current) =>
        current.filter((file) => file.localId !== localId),
      );
      setUploadMessages((current) => {
        const next = current
          .map((message) => ({
            ...message,
            files: message.files.filter((file) => file.localId !== localId),
          }))
          .filter((message) => message.files.length > 0);
        if (sessionId) saveStoredUploadMessages(sessionId, next);
        return next;
      });

      if (!target?.documentId || !sessionId) return;
      try {
        await deleteSessionAttachment(sessionId, target.documentId);
      } catch {
        // Soft-fail when delete is unavailable.
      }
    },
    [sessionId],
  );

  const uploadOne = useCallback(
    async (
      file: File,
      localId: string,
      ensuredId: string,
    ): Promise<LocalChatFile> => {
      const setStatus = (
        status: LocalChatFile["status"],
        progress: number,
        extra: Partial<LocalChatFile> = {},
      ) => {
        setPendingAttachments((current) =>
          current.map((item) =>
            item.localId === localId
              ? { ...item, status, progress, error: undefined, ...extra }
              : item,
          ),
        );
      };

      setStatus("uploading", 20);

      let documentId: string | undefined;
      let attachmentId: string | undefined;

      setStatus("parsing", 40);
      if (attachmentsSupported !== false) {
        try {
          const result = await uploadSessionAttachment(
            ensuredId,
            file,
            companyId,
          );
          setAttachmentsSupported(true);
          if ("document" in result && result.document) {
            const upload = result as UploadResponse;
            documentId = upload.document.document_id;
          } else {
            const attachment = result as ChatAttachment;
            documentId = attachment.document_id;
            attachmentId = attachment.id;
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : "";
          if (/unsupported|only pdf|format/i.test(message)) {
            throw err;
          }
          if (/not found|404|does not exist/i.test(message)) {
            setAttachmentsSupported(false);
          }
          setStatus("chunking", 65);
          const upload = await uploadDocument(companyId, file, {
            sessionId: ensuredId,
            documentScope: "chat",
          });
          documentId = upload.document.document_id;
        }
      } else {
        setStatus("chunking", 65);
        const upload = await uploadDocument(companyId, file, {
          sessionId: ensuredId,
          documentScope: "chat",
        });
        documentId = upload.document.document_id;
      }

      setStatus("embedding", 85, { documentId, attachmentId });
      await new Promise((resolve) => setTimeout(resolve, 180));
      const readyFile: LocalChatFile = {
        localId,
        filename: file.name,
        fileType: fileTypeLabel(file.name, file.type),
        status: "ready",
        progress: 100,
        documentId,
        attachmentId,
        contentType: file.type || "application/octet-stream",
        sizeBytes: file.size,
      };
      setStatus("ready", 100, {
        documentId,
        attachmentId,
        sourceFile: undefined,
      });
      return readyFile;
    },
    [attachmentsSupported, companyId],
  );

  const commitReadyFiles = useCallback(
    (sessionKey: string, ready: LocalChatFile[], batchLocalIds: string[]) => {
      const next = applySuccessfulUploadBatch({
        sessionId: sessionKey,
        pending: pendingRef.current,
        sessionDocuments: sessionDocsRef.current,
        uploadMessages: uploadMessagesRef.current,
        ready,
        batchLocalIds,
      });
      setSessionDocuments(next.sessionDocuments);
      setPendingAttachments(next.pending);
      persistUploadMessages(sessionKey, next.uploadMessages);
    },
    [persistUploadMessages],
  );

  const uploadFiles = useCallback(
    async (incoming: FileList | File[]) => {
      const list = Array.from(incoming);
      if (list.length === 0) return;

      setUploadError(null);
      const accepted = list.filter((file) => isAcceptedChatFile(file));
      const rejected = list.filter((file) => !isAcceptedChatFile(file));
      const oversized = accepted.filter(
        (file) => file.size > MAX_CHAT_FILE_BYTES,
      );
      const uploadable = accepted.filter(
        (file) => file.size <= MAX_CHAT_FILE_BYTES,
      );

      if (rejected.length > 0) {
        setUploadError(
          `Unsupported format. Supported: ${SUPPORTED_FORMAT_LABEL}.`,
        );
      } else if (oversized.length > 0) {
        setUploadError("Each file must be 10 MB or smaller.");
      } else {
        setUploadError(null);
      }
      if (uploadable.length === 0) return;

      setIsUploading(true);
      const placeholders: LocalChatFile[] = uploadable.map((file) => ({
        localId: crypto.randomUUID(),
        filename: file.name,
        fileType: fileTypeLabel(file.name, file.type),
        status: "selected",
        progress: 0,
        contentType: file.type || "application/octet-stream",
        sizeBytes: file.size,
        sourceFile: file,
      }));
      const batchLocalIds = placeholders.map((item) => item.localId);
      setPendingAttachments((current) => [...current, ...placeholders]);

      try {
        const ensuredId = await ensureSession();
        const readyBatch: LocalChatFile[] = [];

        await Promise.all(
          uploadable.map(async (file, index) => {
            const localId = placeholders[index].localId;
            try {
              setPendingAttachments((current) =>
                current.map((item) =>
                  item.localId === localId
                    ? { ...item, status: "queued", progress: 5 }
                    : item,
                ),
              );
              const ready = await uploadOne(file, localId, ensuredId);
              readyBatch.push(ready);
            } catch (err) {
              const message =
                err instanceof Error ? err.message : "Upload failed.";
              setPendingAttachments((current) =>
                current.map((item) =>
                  item.localId === localId
                    ? {
                        ...item,
                        status: "failed",
                        progress: 100,
                        error: message,
                        sourceFile: file,
                      }
                    : item,
                ),
              );
            }
          }),
        );

        commitReadyFiles(ensuredId, readyBatch, batchLocalIds);
      } catch (err) {
        setUploadError(
          err instanceof Error
            ? err.message
            : "Could not prepare chat session.",
        );
        setPendingAttachments((current) =>
          current.map((item) =>
            batchLocalIds.includes(item.localId)
              ? {
                  ...item,
                  status: "failed",
                  progress: 100,
                  error: "Session create failed.",
                }
              : item,
          ),
        );
      } finally {
        setIsUploading(false);
      }
    },
    [commitReadyFiles, ensureSession, uploadOne],
  );

  const retryFile = useCallback(
    async (localId: string) => {
      const target = pendingRef.current.find((file) => file.localId === localId);
      if (!target?.sourceFile) {
        setUploadError("Original file is unavailable. Please browse again.");
        return;
      }
      setUploadError(null);
      setIsUploading(true);
      try {
        const ensuredId = await ensureSession();
        const ready = await uploadOne(target.sourceFile, localId, ensuredId);
        commitReadyFiles(ensuredId, [ready], [localId]);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Upload failed.";
        setPendingAttachments((current) =>
          current.map((item) =>
            item.localId === localId
              ? { ...item, status: "failed", progress: 100, error: message }
              : item,
          ),
        );
        setUploadError(message);
      } finally {
        setIsUploading(false);
      }
    },
    [commitReadyFiles, ensureSession, uploadOne],
  );

  const readyCount = sessionDocuments.filter((f) => f.status === "ready").length;
  const busyCount = pendingAttachments.filter((f) =>
    isBusyStatus(f.status),
  ).length;

  return {
    /** @deprecated Prefer pendingAttachments / sessionDocuments. */
    files: pendingAttachments,
    pendingAttachments,
    sessionDocuments,
    uploadMessages,
    uploadError,
    isUploading,
    readyCount,
    busyCount,
    uploadFiles,
    removeFile,
    retryFile,
    clearFiles,
    forgetSessionUploads,
    loadAttachments,
    setUploadError,
  };
}
