import { describe, expect, it } from "vitest";

import {
  buildChatTimeline,
  parseCreatedAt,
  reconcileOptimisticMessages,
} from "@/lib/chat-timeline";
import type { ChatUploadMessage } from "@/lib/chat-upload-messages";
import type { ChatSessionMessage } from "@/lib/rag-api";

function msg(
  partial: Partial<ChatSessionMessage> &
    Pick<ChatSessionMessage, "id" | "role" | "content" | "created_at">,
): ChatSessionMessage {
  return {
    session_id: "session-1",
    citations: [],
    ...partial,
  };
}

describe("chat timeline ordering", () => {
  it("interleaves upload + 5 Q&A pairs into 11 chronological items", () => {
    const upload: ChatUploadMessage = {
      id: "upload-1",
      kind: "upload",
      session_id: "session-1",
      created_at: "2026-07-30T01:00:00.000Z",
      files: [
        {
          localId: "f1",
          filename: "a.pdf",
          fileType: "PDF",
          status: "ready",
          documentId: "d1",
        },
      ],
    };

    const messages: ChatSessionMessage[] = [];
    for (let i = 1; i <= 5; i += 1) {
      messages.push(
        msg({
          id: `user-${i}`,
          role: "user",
          content: `Question ${i}`,
          created_at: `2026-07-30T01:0${i}:00.000Z`,
        }),
        msg({
          id: `assistant-${i}`,
          role: "assistant",
          content: `Answer ${i}`,
          created_at: `2026-07-30T01:0${i}:00.500Z`,
        }),
      );
    }

    const timeline = buildChatTimeline({ messages, uploadMessages: [upload] });

    expect(timeline).toHaveLength(11);
    expect(timeline.map((item) => item.id)).toEqual([
      "upload-1",
      "user-1",
      "assistant-1",
      "user-2",
      "assistant-2",
      "user-3",
      "assistant-3",
      "user-4",
      "assistant-4",
      "user-5",
      "assistant-5",
    ]);
    expect(timeline.filter((item) => item.role === "user")).toHaveLength(5);
    expect(timeline.filter((item) => item.role === "assistant")).toHaveLength(5);
    expect(timeline.filter((item) => item.kind === "upload")).toHaveLength(1);
    // Upload stays at original chronological position (index 0), not moved to bottom.
    expect(timeline[0].id).toBe("upload-1");
  });

  it("does not group assistants above users when server timestamps lack Z", () => {
    // Simulates client ISO user bubbles mixed with naive UTC assistant timestamps.
    const messages = [
      msg({
        id: "user-1",
        role: "user",
        content: "Q1",
        created_at: "2026-07-30T06:20:00.000Z",
      }),
      msg({
        id: "assistant-1",
        role: "assistant",
        content: "A1",
        created_at: "2026-07-30T06:20:01", // naive UTC from SQLite/FastAPI
      }),
      msg({
        id: "user-2",
        role: "user",
        content: "Q2",
        created_at: "2026-07-30T06:21:00.000Z",
      }),
      msg({
        id: "assistant-2",
        role: "assistant",
        content: "A2",
        created_at: "2026-07-30T06:21:01",
      }),
    ];

    const timeline = buildChatTimeline({ messages, uploadMessages: [] });
    expect(timeline.map((item) => item.id)).toEqual([
      "user-1",
      "assistant-1",
      "user-2",
      "assistant-2",
    ]);

    // Without UTC normalization, naive stamps parse as local and sort early.
    const naiveLocal = Date.parse("2026-07-30T06:20:01");
    const utcAware = parseCreatedAt("2026-07-30T06:20:01");
    expect(utcAware).not.toBe(naiveLocal);
  });

  it("replaces optimistic duplicates instead of keeping both", () => {
    const merged = reconcileOptimisticMessages([
      msg({
        id: "temp-abc",
        role: "user",
        content: "Same question",
        created_at: "2026-07-30T06:20:00.000Z",
      }),
      msg({
        id: "server-user-1",
        role: "user",
        content: "Same question",
        created_at: "2026-07-30T06:20:00.100Z",
      }),
      msg({
        id: "server-assistant-1",
        role: "assistant",
        content: "Answer",
        created_at: "2026-07-30T06:20:01.000Z",
      }),
    ]);

    expect(merged.map((m) => m.id)).toEqual([
      "server-user-1",
      "server-assistant-1",
    ]);
  });

  it("dedupes by message id", () => {
    const timeline = buildChatTimeline({
      messages: [
        msg({
          id: "u1",
          role: "user",
          content: "Q",
          created_at: "2026-07-30T06:20:00.000Z",
        }),
        msg({
          id: "u1",
          role: "user",
          content: "Q",
          created_at: "2026-07-30T06:20:00.000Z",
        }),
        msg({
          id: "a1",
          role: "assistant",
          content: "A",
          created_at: "2026-07-30T06:20:01.000Z",
        }),
      ],
    });
    expect(timeline).toHaveLength(2);
  });
});
