"use client";

import { FormEvent, useEffect, useRef } from "react";

import { FileChips } from "@/components/chat/FileChips";
import { FileDropZone } from "@/components/chat/FileDropZone";
import { SendIcon } from "@/components/chat/icons";
import type { LocalChatFile } from "@/hooks/useChatAttachments";

export function ChatComposer({
  value,
  onChange,
  onSubmit,
  onFiles,
  onRemoveFile,
  onRetryFile,
  /** Temporary pending uploads only — never sessionDocuments. */
  files,
  uploadError,
  disabled,
  isSending,
  showDropHint = false,
  pageDragging = false,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onFiles: (files: FileList | File[]) => void;
  onRemoveFile: (localId: string) => void;
  onRetryFile?: (localId: string) => void;
  files: LocalChatFile[];
  uploadError?: string | null;
  disabled?: boolean;
  isSending?: boolean;
  showDropHint?: boolean;
  pageDragging?: boolean;
}) {
  const formRef = useRef<HTMLFormElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const canSend = Boolean(value.trim()) && !disabled && !isSending;

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSend) return;
    onSubmit();
  }

  return (
    <div className="mx-auto w-full max-w-[840px] px-4 pb-4 sm:pb-6">
      {files.length > 0 || uploadError ? (
        <div className="mb-3 space-y-2">
          <FileChips
            files={files}
            onRemove={onRemoveFile}
            onRetry={onRetryFile}
          />
          {uploadError ? (
            <p className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
              {uploadError}
            </p>
          ) : null}
        </div>
      ) : null}

      <form
        ref={formRef}
        onSubmit={handleSubmit}
        className={`rounded-2xl border bg-white p-2 shadow-sm shadow-slate-900/5 transition ${
          pageDragging
            ? "border-sky-500 ring-2 ring-sky-200"
            : "border-slate-200"
        }`}
      >
        <div className="flex items-end gap-2">
          <FileDropZone
            compact
            onFiles={onFiles}
            disabled={isSending}
          />
          <label htmlFor="chat-composer" className="sr-only">
            Ask anything about your documents
          </label>
          <textarea
            ref={textareaRef}
            id="chat-composer"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                formRef.current?.requestSubmit();
              }
            }}
            rows={1}
            placeholder="Ask anything about your documents..."
            disabled={disabled}
            className="max-h-40 min-h-11 flex-1 resize-none bg-transparent px-2 py-2.5 text-sm text-slate-900 outline-none placeholder:text-slate-400 focus-visible:ring-0 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={!canSend}
            className={`inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-white transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-500 ${
              canSend
                ? "bg-slate-900 hover:bg-slate-800"
                : "cursor-not-allowed bg-slate-300"
            }`}
            aria-label={isSending ? "Sending…" : "Send message"}
          >
            <SendIcon />
          </button>
        </div>
      </form>

      <p className="mt-2 text-center text-[11px] text-slate-400">
        {showDropHint
          ? "Press Enter to send · Shift+Enter for a new line"
          : "Answers cite company sources when available"}
      </p>
    </div>
  );
}
