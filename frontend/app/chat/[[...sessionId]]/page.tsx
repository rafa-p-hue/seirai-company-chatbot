"use client";

import { use } from "react";

import { ChatShell } from "@/components/chat/ChatShell";

/** `/chat` and `/chat/[sessionId]` share one client shell so URL sync does not remount mid-send. */
export default function ChatCatchAllPage({
  params,
}: {
  params: Promise<{ sessionId?: string[] }>;
}) {
  const resolved = use(params);
  const sessionId = resolved.sessionId?.[0] ?? null;

  return (
    <ChatShell
      companyId="seirai"
      variant="app"
      initialSessionId={sessionId}
    />
  );
}
