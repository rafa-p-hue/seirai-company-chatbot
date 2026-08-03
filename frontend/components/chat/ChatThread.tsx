"use client";

import { useEffect, useMemo, useRef } from "react";

import { AttachmentUploadMessage } from "@/components/chat/AttachmentUploadMessage";
import {
  buildChatTimeline,
  isTimelineDebugEnabled,
} from "@/lib/chat-timeline";
import type { ChatUploadMessage } from "@/lib/chat-upload-messages";
import type { ChatSessionMessage } from "@/lib/rag-api";

function renderMessageBody(content: string) {
  const lines = content.split("\n");
  const blocks: Array<{ type: "p" | "ul" | "ol"; items: string[] }> = [];

  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    if (bullet) {
      const last = blocks[blocks.length - 1];
      if (last?.type === "ul") last.items.push(bullet[1]);
      else blocks.push({ type: "ul", items: [bullet[1]] });
      continue;
    }
    if (numbered) {
      const last = blocks[blocks.length - 1];
      if (last?.type === "ol") last.items.push(numbered[1]);
      else blocks.push({ type: "ol", items: [numbered[1]] });
      continue;
    }
    if (!line.trim()) continue;
    blocks.push({ type: "p", items: [line] });
  }

  if (blocks.length === 0) {
    return <p className="whitespace-pre-wrap">{content}</p>;
  }

  return (
    <div className="space-y-2">
      {blocks.map((block, index) => {
        if (block.type === "ul") {
          return (
            <ul key={index} className="list-disc space-y-1 pl-5">
              {block.items.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          );
        }
        if (block.type === "ol") {
          return (
            <ol key={index} className="list-decimal space-y-1 pl-5">
              {block.items.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ol>
          );
        }
        return (
          <p key={index} className="whitespace-pre-wrap">
            {block.items[0]}
          </p>
        );
      })}
    </div>
  );
}

export function ChatThread({
  messages,
  uploadMessages = [],
  isSending,
  isLoading,
  sendError,
  onRetry,
  onDismissError,
}: {
  messages: ChatSessionMessage[];
  uploadMessages?: ChatUploadMessage[];
  isSending: boolean;
  isLoading?: boolean;
  sendError?: string | null;
  onRetry?: () => void;
  onDismissError?: () => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);

  const timeline = useMemo(
    () =>
      buildChatTimeline({
        messages,
        uploadMessages,
        debug: isTimelineDebugEnabled(),
      }),
    [messages, uploadMessages],
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [timeline, isSending, sendError]);

  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center px-4 text-sm text-slate-500">
        Loading conversation…
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-[840px] flex-1 flex-col gap-4 px-4 py-6">
      {timeline.map((item) => {
        if (item.kind === "upload" && item.upload) {
          return (
            <AttachmentUploadMessage
              key={item.id}
              message={item.upload}
            />
          );
        }

        const message = item.message;
        if (!message) return null;
        const isUser = message.role === "user";

        return (
          <article
            key={item.id}
            className={`flex ${isUser ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`w-fit max-w-[min(100%,36rem)] rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm ${
                isUser
                  ? "bg-slate-900 text-white"
                  : "border border-slate-200 bg-white text-slate-800"
              }`}
            >
              {isUser ? (
                <p className="whitespace-pre-wrap">{message.content}</p>
              ) : (
                renderMessageBody(message.content)
              )}
              {!isUser && message.citations?.length ? (
                <div className="mt-3 border-t border-slate-200 pt-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                    Sources
                  </p>
                  <ul className="mt-2 space-y-1 text-xs text-slate-600">
                    {message.citations.map((source, index) => {
                      const number = source.number ?? index + 1;
                      const name =
                        typeof source.document_name === "string"
                          ? source.document_name
                          : "Source";
                      return (
                        <li key={`${message.id}-${number}-${name}`}>
                          [{number}]{" "}
                          {source.source_url ? (
                            <a
                              href={String(source.source_url)}
                              target="_blank"
                              rel="noreferrer"
                              className="underline"
                            >
                              {name}
                            </a>
                          ) : (
                            name
                          )}
                          {source.page_number
                            ? `, page ${source.page_number}`
                            : ""}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ) : null}
            </div>
          </article>
        );
      })}

      {isSending ? (
        <div className="flex justify-start">
          <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
            <span className="inline-flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse rounded-full bg-sky-600" />
              Thinking…
            </span>
          </div>
        </div>
      ) : null}

      {sendError && !isSending ? (
        <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950 shadow-sm">
          <p>{sendError}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {onRetry ? (
              <button
                type="button"
                onClick={onRetry}
                className="rounded-lg bg-amber-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-amber-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-700"
              >
                Retry
              </button>
            ) : null}
            {onDismissError ? (
              <button
                type="button"
                onClick={onDismissError}
                className="rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-900 hover:bg-amber-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-500"
              >
                Dismiss
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      <div ref={endRef} />
    </div>
  );
}
