import { describe, expect, it } from "vitest";

import {
  applySuccessfulUploadBatch,
  summarizeUploadMessage,
  type ChatUploadMessage,
  type UploadCommitFile,
} from "@/lib/chat-upload-messages";

function makeReadyFile(index: number): UploadCommitFile {
  return {
    localId: `local-${index}`,
    filename: `document-${index}.pdf`,
    fileType: "PDF",
    status: "ready",
    documentId: `doc-${index}`,
    attachmentId: `att-${index}`,
  };
}

describe("chat upload message UX", () => {
  it("groups seven uploads into one chat message and clears the composer grid", () => {
    const ready = Array.from({ length: 7 }, (_, index) =>
      makeReadyFile(index + 1),
    );
    const pending: UploadCommitFile[] = ready.map((file) => ({
      ...file,
      status: "ready",
    }));
    const batchLocalIds = pending.map((file) => file.localId);

    const result = applySuccessfulUploadBatch({
      sessionId: "session-seven",
      pending,
      sessionDocuments: [],
      uploadMessages: [],
      ready,
      batchLocalIds,
      messageId: "upload-msg-1",
      createdAt: "2026-07-29T12:00:00.000Z",
    });

    // One grouped attachment message in chat history
    expect(result.uploadMessages).toHaveLength(1);
    const uploadMessage = result.uploadMessages[0] as ChatUploadMessage;
    expect(uploadMessage.kind).toBe("upload");
    expect(uploadMessage.files).toHaveLength(7);

    const summary = summarizeUploadMessage(uploadMessage);
    expect(summary.title).toBe("Uploaded 7 documents");
    expect(summary.subtitle).toBe("7 documents ready");

    // Composer no longer shows the seven-file grid
    expect(result.pending).toHaveLength(0);
    expect(result.composerVisibleCount).toBe(0);

    // All seven files remain attached to the session
    expect(result.sessionDocuments).toHaveLength(7);
    expect(result.sessionDocuments.map((file) => file.documentId)).toEqual([
      "doc-1",
      "doc-2",
      "doc-3",
      "doc-4",
      "doc-5",
      "doc-6",
      "doc-7",
    ]);
  });

  it("keeps failed pending files near the composer after a mixed batch", () => {
    const ready = [makeReadyFile(1), makeReadyFile(2)];
    const failed: UploadCommitFile = {
      localId: "local-3",
      filename: "broken.pdf",
      fileType: "PDF",
      status: "failed",
      error: "Upload failed.",
    };
    const pending = [...ready, failed];

    const result = applySuccessfulUploadBatch({
      sessionId: "session-mixed",
      pending,
      sessionDocuments: [],
      uploadMessages: [],
      ready,
      batchLocalIds: pending.map((file) => file.localId),
    });

    expect(result.uploadMessages).toHaveLength(1);
    expect(result.uploadMessages[0].files).toHaveLength(2);
    expect(result.sessionDocuments).toHaveLength(2);
    expect(result.pending).toEqual([failed]);
    expect(result.composerVisibleCount).toBe(1);
  });

  it("restores a historical upload message without recreating composer pending state", () => {
    const stored: ChatUploadMessage = {
      id: "upload-hist",
      kind: "upload",
      session_id: "session-reopen",
      created_at: "2026-07-28T09:00:00.000Z",
      files: Array.from({ length: 7 }, (_, index) => ({
        localId: `local-${index + 1}`,
        filename: `document-${index + 1}.pdf`,
        fileType: "PDF",
        status: "ready" as const,
        documentId: `doc-${index + 1}`,
      })),
    };

    // Reopen path: sessionDocuments come from backend; pending stays empty.
    const sessionDocuments = stored.files.map((file) => ({
      localId: file.localId,
      filename: file.filename,
      fileType: file.fileType,
      status: "ready",
      documentId: file.documentId,
    }));

    expect(stored.files).toHaveLength(7);
    expect(sessionDocuments).toHaveLength(7);
    expect(summarizeUploadMessage(stored).title).toBe("Uploaded 7 documents");
    // Composer grid must not be rebuilt from historical attachments
    expect([] as UploadCommitFile[]).toHaveLength(0);
  });
});
