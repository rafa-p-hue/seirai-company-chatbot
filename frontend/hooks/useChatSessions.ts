"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  logTimelineDebug,
  normalizeSessionMessages,
  reconcileOptimisticMessages,
} from "@/lib/chat-timeline";
import {
  checkBackendHealth,
  createChatSession,
  deleteChatSession,
  getChatSession,
  listChatSessions,
  renameChatSession,
  sendChatSessionMessage,
  type ChatSessionDetail,
  type ChatSessionMessage,
  type ChatSessionSummary,
} from "@/lib/rag-api";

export type UseChatSessionsOptions = {
  companyId?: string;
  autoLoad?: boolean;
  /** Session to open on mount (from `/chat/[sessionId]`). */
  initialSessionId?: string | null;
  /** Called when the active session id changes (for URL sync). */
  onSessionIdChange?: (sessionId: string | null) => void;
  /** Called after a session is loaded from the API (not after local create). */
  onSessionOpened?: (sessionId: string) => void;
};

export function useChatSessions(options: UseChatSessionsOptions = {}) {
  const companyId = options.companyId ?? "seirai";
  const autoLoad = options.autoLoad ?? true;
  const initialSessionId = options.initialSessionId ?? null;
  const onSessionIdChange = options.onSessionIdChange;
  const onSessionOpened = options.onSessionOpened;

  const [backendReady, setBackendReady] = useState(false);
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatSessionMessage[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingRetry, setPendingRetry] = useState<string | null>(null);

  const sendingRef = useRef(false);
  const openInFlightRef = useRef<string | null>(null);
  const activeSessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  const notifySessionChange = useCallback(
    (sessionId: string | null) => {
      setActiveSessionId(sessionId);
      onSessionIdChange?.(sessionId);
    },
    [onSessionIdChange],
  );

  const refreshSessions = useCallback(async () => {
    const healthy = await checkBackendHealth();
    setBackendReady(healthy);
    if (!healthy) {
      setSessions([]);
      setIsLoadingSessions(false);
      return;
    }
    try {
      const list = await listChatSessions();
      setSessions(list);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load chats.");
      setSessions([]);
    } finally {
      setIsLoadingSessions(false);
    }
  }, []);

  useEffect(() => {
    if (!autoLoad) return;
    let cancelled = false;
    let attempt = 0;

    async function load() {
      try {
        const healthy = await checkBackendHealth();
        if (cancelled) return;
        setBackendReady(healthy);
        if (!healthy) {
          setSessions([]);
          // Retry a few times — first paint can race the FastAPI proxy.
          if (attempt < 4) {
            attempt += 1;
            window.setTimeout(() => {
              if (!cancelled) void load();
            }, 700 * attempt);
          }
          return;
        }
        const list = await listChatSessions();
        if (cancelled) return;
        setSessions(list);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load chats.");
        setSessions([]);
      } finally {
        if (!cancelled) setIsLoadingSessions(false);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [autoLoad]);

  // Keep retrying health while offline so uploads unlock after backend restarts.
  useEffect(() => {
    if (!autoLoad || backendReady) return;
    let cancelled = false;

    async function probe() {
      const healthy = await checkBackendHealth();
      if (cancelled || !healthy) return;
      setBackendReady(true);
      void refreshSessions();
    }

    void probe();
    const timer = window.setInterval(() => {
      void probe();
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [autoLoad, backendReady, refreshSessions]);

  const openSession = useCallback(
    async (sessionId: string, options?: { syncUrl?: boolean }) => {
      // Guard against Strict Mode / overlapping clicks loading the same session twice.
      if (openInFlightRef.current === sessionId) {
        return sessionId;
      }
      openInFlightRef.current = sessionId;

      const syncUrl = options?.syncUrl ?? true;
      setIsLoadingMessages(true);
      setError(null);
      setPendingRetry(null);
      try {
        const detail: ChatSessionDetail = await getChatSession(sessionId);
        if (syncUrl) {
          notifySessionChange(detail.id);
        } else {
          setActiveSessionId(detail.id);
        }
        const normalized = normalizeSessionMessages(detail.messages);
        setMessages(normalized);
        if (
          typeof window !== "undefined" &&
          window.localStorage.getItem("seirai.chat.timelineDebug") === "1"
        ) {
          logTimelineDebug(
            normalized.map((message, index) => ({
              id: message.id,
              kind: "message" as const,
              role: message.role,
              type: "message" as const,
              created_at: message.created_at,
              sortAt: Date.parse(message.created_at) || 0,
              sequence_number: index,
              source: "server" as const,
              content: message.content,
              message,
            })),
          );
        }
        onSessionOpened?.(detail.id);
        return detail.id;
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to open chat.");
        if (syncUrl) {
          notifySessionChange(null);
        } else {
          setActiveSessionId(null);
          onSessionIdChange?.(null);
        }
        setMessages([]);
        return null;
      } finally {
        if (openInFlightRef.current === sessionId) {
          openInFlightRef.current = null;
        }
        setIsLoadingMessages(false);
      }
    },
    [notifySessionChange, onSessionIdChange, onSessionOpened],
  );

  // Hydrate from the URL without clobbering an in-progress local session.
  useEffect(() => {
    if (!initialSessionId) return;
    if (activeSessionId === initialSessionId) return;
    if (openInFlightRef.current === initialSessionId) return;
    const timer = window.setTimeout(() => {
      void openSession(initialSessionId, { syncUrl: false });
    }, 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialSessionId]);

  const startNewChat = useCallback(async () => {
    setMessages([]);
    setError(null);
    setPendingRetry(null);
    const session = await createChatSession("New Chat");
    setSessions((current) => [
      session,
      ...current.filter((s) => s.id !== session.id),
    ]);
    notifySessionChange(session.id);
    return session.id;
  }, [notifySessionChange]);

  const clearMessagesLocally = useCallback(() => {
    setMessages([]);
    setError(null);
    setPendingRetry(null);
  }, []);

  const ensureSession = useCallback(async () => {
    if (activeSessionIdRef.current) return activeSessionIdRef.current;
    return startNewChat();
  }, [startNewChat]);

  const sendMessage = useCallback(
    async (content: string) => {
      const trimmed = content.trim();
      if (!trimmed || sendingRef.current) return;

      sendingRef.current = true;
      setIsSending(true);
      setError(null);
      setPendingRetry(null);

      const tempId = `temp-${crypto.randomUUID()}`;
      const optimisticUser: ChatSessionMessage = {
        id: tempId,
        session_id: activeSessionIdRef.current ?? "pending",
        role: "user",
        content: trimmed,
        citations: [],
        created_at: new Date().toISOString(),
      };
      setMessages((current) =>
        reconcileOptimisticMessages([...current, optimisticUser]),
      );

      try {
        const sessionId = await ensureSession();
        const turn = await sendChatSessionMessage({
          sessionId,
          content: trimmed,
          companyId,
        });

        setMessages((current) => {
          const withoutTemp = current.filter((m) => m.id !== tempId);
          return reconcileOptimisticMessages([
            ...withoutTemp,
            turn.user,
            turn.assistant,
          ]);
        });

        if (!activeSessionIdRef.current) {
          notifySessionChange(sessionId);
        }
        await refreshSessions();
      } catch {
        // Keep the single optimistic user bubble; do NOT append a second copy.
        setError("Something went wrong while sending your message.");
        setPendingRetry(trimmed);
      } finally {
        sendingRef.current = false;
        setIsSending(false);
      }
    },
    [companyId, ensureSession, notifySessionChange, refreshSessions],
  );

  const retryLastFailed = useCallback(async () => {
    if (!pendingRetry || sendingRef.current) return;
    const text = pendingRetry;
    setPendingRetry(null);
    setError(null);
    // Drop the last optimistic/temp user bubble that matches the failed text, then resend.
    setMessages((current) => {
      const lastUserIndex = [...current]
        .map((m, i) => ({ m, i }))
        .reverse()
        .find(({ m }) => m.role === "user" && m.content === text)?.i;
      if (lastUserIndex == null) return current;
      return current.filter((_, i) => i !== lastUserIndex);
    });
    await sendMessage(text);
  }, [pendingRetry, sendMessage]);

  const dismissError = useCallback(() => {
    setError(null);
    setPendingRetry(null);
  }, []);

  const renameSession = useCallback(
    async (sessionId: string, title: string) => {
      const updated = await renameChatSession(sessionId, title);
      setSessions((current) =>
        current.map((session) => (session.id === sessionId ? updated : session)),
      );
      return updated;
    },
    [],
  );

  const removeSession = useCallback(
    async (sessionId: string) => {
      await deleteChatSession(sessionId);
      setSessions((current) =>
        current.filter((session) => session.id !== sessionId),
      );
      if (activeSessionId === sessionId) {
        setMessages([]);
        setError(null);
        setPendingRetry(null);
        const session = await createChatSession("New Chat");
        setSessions((current) => [
          session,
          ...current.filter((s) => s.id !== session.id),
        ]);
        notifySessionChange(session.id);
      }
    },
    [activeSessionId, notifySessionChange],
  );

  return {
    backendReady,
    sessions,
    activeSessionId,
    messages,
    isLoadingSessions,
    isLoadingMessages,
    isSending,
    error,
    pendingRetry,
    setError,
    dismissError,
    retryLastFailed,
    refreshSessions,
    openSession,
    startNewChat,
    clearMessagesLocally,
    ensureSession,
    sendMessage,
    renameSession,
    removeSession,
    companyId,
  };
}
