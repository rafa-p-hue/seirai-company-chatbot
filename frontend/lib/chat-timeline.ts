/**
 * Unified chat timeline: messages + uploads, sorted chronologically.
 */

import type { ChatUploadMessage } from "@/lib/chat-upload-messages";
import type { ChatSessionMessage } from "@/lib/rag-api";

export type TimelineSource = "optimistic" | "server" | "upload";

export type TimelineItem = {
  id: string;
  kind: "message" | "upload" | "system";
  role?: "user" | "assistant" | "system";
  type: "message" | "upload" | "system";
  created_at: string;
  sortAt: number;
  sequence_number: number;
  source: TimelineSource;
  content?: string;
  message?: ChatSessionMessage;
  upload?: ChatUploadMessage;
};

const TEMP_ID_RE = /^(local-|temp-|optimistic-)/i;

/** Treat timezone-less datetimes as UTC (SQLite/FastAPI naive UTC). */
export function parseCreatedAt(value: string | null | undefined): number {
  if (!value) return 0;
  const trimmed = String(value).trim();
  if (!trimmed) return 0;
  const hasTz = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(trimmed);
  const normalized = hasTz ? trimmed : `${trimmed}Z`;
  const ms = Date.parse(normalized);
  return Number.isFinite(ms) ? ms : 0;
}

export function isOptimisticMessageId(id: string): boolean {
  return TEMP_ID_RE.test(id);
}

export function dedupeMessagesById(
  messages: ChatSessionMessage[],
): ChatSessionMessage[] {
  const seen = new Set<string>();
  const result: ChatSessionMessage[] = [];
  for (const message of messages) {
    if (!message?.id || seen.has(message.id)) continue;
    seen.add(message.id);
    result.push(message);
  }
  return result;
}

/**
 * Drop optimistic user bubbles once a server user message with the same
 * content exists (prevents temp + server duplicates).
 */
export function reconcileOptimisticMessages(
  messages: ChatSessionMessage[],
): ChatSessionMessage[] {
  const deduped = dedupeMessagesById(messages);
  const serverUserContents = new Set(
    deduped
      .filter(
        (m) => m.role === "user" && !isOptimisticMessageId(m.id),
      )
      .map((m) => m.content.trim()),
  );
  return deduped.filter((message) => {
    if (
      message.role === "user" &&
      isOptimisticMessageId(message.id) &&
      serverUserContents.has(message.content.trim())
    ) {
      return false;
    }
    return true;
  });
}

export function normalizeSessionMessages(
  messages: ChatSessionMessage[],
): ChatSessionMessage[] {
  return reconcileOptimisticMessages(messages).slice().sort((a, b) => {
    const timeDiff = parseCreatedAt(a.created_at) - parseCreatedAt(b.created_at);
    if (timeDiff !== 0) return timeDiff;
    return String(a.id).localeCompare(String(b.id));
  });
}

function compareTimelineItems(a: TimelineItem, b: TimelineItem): number {
  if (a.sortAt !== b.sortAt) return a.sortAt - b.sortAt;
  if (a.sequence_number !== b.sequence_number) {
    return a.sequence_number - b.sequence_number;
  }
  return String(a.id).localeCompare(String(b.id));
}

export function buildChatTimeline(options: {
  messages: ChatSessionMessage[];
  uploadMessages?: ChatUploadMessage[];
  debug?: boolean;
}): TimelineItem[] {
  const messages = reconcileOptimisticMessages(options.messages);
  const uploads = options.uploadMessages ?? [];

  const items: TimelineItem[] = [];

  messages.forEach((message, index) => {
    items.push({
      id: message.id,
      kind: "message",
      role: message.role,
      type: "message",
      created_at: message.created_at,
      sortAt: parseCreatedAt(message.created_at),
      sequence_number: index,
      source: isOptimisticMessageId(message.id) ? "optimistic" : "server",
      content: message.content,
      message,
    });
  });

  uploads.forEach((upload, index) => {
    items.push({
      id: upload.id,
      kind: "upload",
      type: "upload",
      created_at: upload.created_at,
      sortAt: parseCreatedAt(upload.created_at),
      // Keep uploads ahead of chat turns that share the exact same ms by using
      // a sequence just below the message insertion window for that batch.
      sequence_number: -1_000_000 + index,
      source: "upload",
      upload,
    });
  });

  // Deduplicate by id across the unified list.
  const seen = new Set<string>();
  const unique = items.filter((item) => {
    if (seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });

  unique.sort(compareTimelineItems);

  if (options.debug ?? isTimelineDebugEnabled()) {
    logTimelineDebug(unique);
  }

  return unique;
}

export function isTimelineDebugEnabled(): boolean {
  if (process.env.NODE_ENV === "development") return true;
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem("seirai.chat.timelineDebug") === "1";
  } catch {
    return false;
  }
}

export function logTimelineDebug(items: TimelineItem[]): void {
  // Temporary debug output for timeline ordering investigations.
  // Enable with: localStorage.setItem("seirai.chat.timelineDebug", "1")
  // eslint-disable-next-line no-console
  console.groupCollapsed(
    `[chat-timeline] ${items.length} item(s)`,
  );
  items.forEach((item, index) => {
    // eslint-disable-next-line no-console
    console.log({
      final_sorted_index: index,
      id: item.id,
      role: item.role ?? null,
      type: item.type,
      created_at: item.created_at,
      sortAt: item.sortAt,
      sequence_number: item.sequence_number,
      source: item.source,
      content_preview: item.content?.slice(0, 80) ?? null,
    });
  });
  // eslint-disable-next-line no-console
  console.groupEnd();
}
