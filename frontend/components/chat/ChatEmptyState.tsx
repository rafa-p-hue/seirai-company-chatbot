"use client";

import { FileDropZone } from "@/components/chat/FileDropZone";

const SUGGESTED_PROMPTS = [
  "Summarize this document",
  "What are the key dates?",
  "What information is missing?",
] as const;

export function ChatEmptyState({
  isUploading,
  promptsEnabled,
  onFiles,
  onSuggestedPrompt,
}: {
  isUploading: boolean;
  promptsEnabled: boolean;
  onFiles: (files: FileList | File[]) => void;
  onSuggestedPrompt: (prompt: string) => void;
}) {
  return (
    <div className="mx-auto flex w-full max-w-[840px] flex-col items-center px-4 text-center">
      <h1 className="text-[1.65rem] font-semibold tracking-tight text-slate-900 sm:text-3xl">
        Hello, how can I help you today?
      </h1>
      <p className="mt-2.5 max-w-lg text-sm leading-6 text-slate-500">
        Ask about your company knowledge base, or attach files to this chat.
      </p>

      <div className="mt-6 w-full">
        <FileDropZone onFiles={onFiles} disabled={isUploading} />
      </div>

      <div className="mt-5 flex w-full flex-col items-center gap-2">
        <div className="flex w-full flex-wrap items-center justify-center gap-2">
          {SUGGESTED_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              type="button"
              disabled={!promptsEnabled}
              onClick={() => {
                if (!promptsEnabled) return;
                onSuggestedPrompt(prompt);
              }}
              className="rounded-full border border-slate-200 bg-white px-3.5 py-1.5 text-xs font-medium text-slate-700 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-400 disabled:cursor-not-allowed disabled:border-slate-100 disabled:bg-slate-50 disabled:text-slate-400 disabled:hover:bg-slate-50"
            >
              {prompt}
            </button>
          ))}
        </div>
        {!promptsEnabled ? (
          <p className="text-[11px] text-slate-400">Upload a document first.</p>
        ) : null}
      </div>
    </div>
  );
}
