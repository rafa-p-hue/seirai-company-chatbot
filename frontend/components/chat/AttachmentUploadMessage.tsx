"use client";

import { useState } from "react";

import { DocumentIcon, XIcon } from "@/components/chat/icons";
import {
  summarizeUploadMessage,
  type ChatUploadMessage,
  type UploadMessageFile,
} from "@/lib/chat-upload-messages";

function statusLabel(file: UploadMessageFile): string {
  if (file.status === "failed" || file.status === "error") {
    return file.error || "Failed";
  }
  if (file.status === "processing") return "Processing…";
  return "Ready";
}

function statusColor(file: UploadMessageFile): string {
  if (file.status === "failed" || file.status === "error") return "text-amber-800";
  if (file.status === "ready") return "text-emerald-700";
  return "text-slate-500";
}

export function AttachmentUploadMessage({
  message,
  defaultExpanded = false,
  onRemoveFile,
  onRetryFile,
}: {
  message: ChatUploadMessage;
  defaultExpanded?: boolean;
  onRemoveFile?: (localId: string) => void;
  onRetryFile?: (localId: string) => void;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const { title, subtitle } = summarizeUploadMessage(message);

  return (
    <article className="flex justify-end">
      <div className="w-full max-w-[min(100%,28rem)] overflow-hidden rounded-2xl border border-slate-200 bg-white text-left shadow-sm shadow-slate-900/5">
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="flex w-full items-start gap-3 px-4 py-3 text-left transition hover:bg-slate-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-400"
          aria-expanded={expanded}
        >
          <span className="mt-0.5 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-slate-200 bg-slate-50 text-slate-500">
            <DocumentIcon className="h-4 w-4" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-semibold text-slate-900">
              {title}
            </span>
            <span className="mt-0.5 block text-xs text-slate-500">
              {subtitle}
            </span>
          </span>
          <span className="mt-1 text-[11px] font-medium text-slate-400">
            {expanded ? "Hide" : "Show"}
          </span>
        </button>

        {expanded ? (
          <ul className="space-y-2 border-t border-slate-100 px-3 py-3">
            {message.files.map((file) => {
              const canAct =
                file.status === "failed" ||
                file.status === "error" ||
                file.status === "processing";
              return (
                <li
                  key={file.localId}
                  className={`flex min-w-0 items-start gap-2.5 rounded-xl border px-3 py-2.5 text-xs shadow-sm ${
                    file.status === "failed" || file.status === "error"
                      ? "border-amber-200 bg-amber-50 text-amber-950"
                      : "border-slate-200 bg-slate-50/80 text-slate-700"
                  }`}
                >
                  <DocumentIcon className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-medium">
                      {file.filename}
                    </span>
                    <span className={`mt-0.5 block ${statusColor(file)}`}>
                      {file.fileType} · {statusLabel(file)}
                    </span>
                  </span>
                  {canAct && onRetryFile &&
                  (file.status === "failed" || file.status === "error") ? (
                    <button
                      type="button"
                      onClick={() => onRetryFile(file.localId)}
                      className="shrink-0 rounded-lg px-2 py-1 font-semibold text-amber-900 hover:bg-amber-100"
                    >
                      Retry
                    </button>
                  ) : null}
                  {canAct && onRemoveFile ? (
                    <button
                      type="button"
                      onClick={() => onRemoveFile(file.localId)}
                      className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                      aria-label={`Remove ${file.filename}`}
                    >
                      <XIcon />
                    </button>
                  ) : null}
                </li>
              );
            })}
          </ul>
        ) : null}
      </div>
    </article>
  );
}
