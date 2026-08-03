"use client";

import { DocumentIcon, XIcon } from "@/components/chat/icons";
import type { LocalChatFile } from "@/hooks/useChatAttachments";

function statusLabel(file: LocalChatFile): string {
  if (file.status === "selected") return "Selected";
  if (file.status === "queued") return "Queued";
  if (file.status === "uploading") return "Uploading…";
  if (file.status === "parsing") return "Parsing…";
  if (file.status === "chunking") return "Chunking…";
  if (file.status === "embedding") return "Embedding…";
  if (file.status === "processing") return "Processing…";
  if (file.status === "failed" || file.status === "error") {
    return file.error || "Failed";
  }
  return "Ready";
}

function statusColor(file: LocalChatFile): string {
  if (file.status === "failed" || file.status === "error") return "text-amber-800";
  if (file.status === "ready") return "text-emerald-700";
  return "text-slate-500";
}

function isInFlight(file: LocalChatFile): boolean {
  return (
    file.status === "selected" ||
    file.status === "queued" ||
    file.status === "uploading" ||
    file.status === "parsing" ||
    file.status === "chunking" ||
    file.status === "embedding" ||
    file.status === "processing"
  );
}

function isFailed(file: LocalChatFile): boolean {
  return file.status === "failed" || file.status === "error";
}

/** Compact pending-upload preview for the composer area only. */
export function FileChips({
  files,
  onRemove,
  onRetry,
}: {
  files: LocalChatFile[];
  onRemove: (localId: string) => void;
  onRetry?: (localId: string) => void;
}) {
  if (files.length === 0) return null;

  return (
    <ul className="grid gap-2 sm:grid-cols-2">
      {files.map((file) => (
        <li
          key={file.localId}
          className={`flex min-w-0 items-start gap-2.5 rounded-xl border px-3 py-2.5 text-xs shadow-sm ${
            isFailed(file)
              ? "border-amber-200 bg-amber-50 text-amber-950"
              : "border-slate-200 bg-white text-slate-700"
          }`}
        >
          <DocumentIcon className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
          <span className="min-w-0 flex-1">
            <span className="block truncate font-medium">{file.filename}</span>
            <span className={`mt-0.5 block ${statusColor(file)}`}>
              {file.fileType} · {statusLabel(file)}
            </span>
            {isInFlight(file) ? (
              <span
                className="mt-2 block h-1.5 w-full overflow-hidden rounded-full bg-slate-200"
                aria-hidden="true"
              >
                <span
                  className="block h-full rounded-full bg-sky-600 transition-all"
                  style={{ width: `${Math.max(file.progress, 12)}%` }}
                />
              </span>
            ) : null}
          </span>
          {isFailed(file) && onRetry ? (
            <button
              type="button"
              onClick={() => onRetry(file.localId)}
              className="shrink-0 rounded-lg px-2 py-1 font-semibold text-amber-900 hover:bg-amber-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-500"
            >
              Retry
            </button>
          ) : null}
          {isInFlight(file) || isFailed(file) ? (
            <button
              type="button"
              onClick={() => onRemove(file.localId)}
              className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-400"
              aria-label={`Remove ${file.filename}`}
            >
              <XIcon />
            </button>
          ) : null}
        </li>
      ))}
    </ul>
  );
}
