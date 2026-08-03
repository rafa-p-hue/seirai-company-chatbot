"use client";

import { useRouter } from "next/navigation";
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type DragEvent,
} from "react";

import { ChatComposer } from "@/components/chat/ChatComposer";
import { ChatEmptyState } from "@/components/chat/ChatEmptyState";
import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { ChatThread } from "@/components/chat/ChatThread";
import { MenuIcon, MoreVerticalIcon } from "@/components/chat/icons";
import { useChatAttachments } from "@/hooks/useChatAttachments";
import { useChatSessions } from "@/hooks/useChatSessions";
import { listDocuments } from "@/lib/rag-api";

export type ChatShellProps = {
  companyId?: string;
  variant?: "app" | "embed";
  className?: string;
  initialSessionId?: string | null;
};

const SIDEBAR_COLLAPSE_KEY = "seirai.chat.sidebarCollapsed";

function statusSubtitle(options: {
  backendReady: boolean;
  readyCount: number;
  busyCount: number;
}): string {
  if (!options.backendReady) {
    return "Backend offline — start FastAPI to chat";
  }
  if (options.busyCount > 0) {
    return options.busyCount === 1
      ? "1 file processing"
      : `${options.busyCount} files processing`;
  }
  if (options.readyCount > 0) {
    return options.readyCount === 1
      ? "1 file attached"
      : `${options.readyCount} files ready`;
  }
  return "Company knowledge base connected";
}

export function ChatShell({
  companyId = "seirai",
  variant = "app",
  className = "",
  initialSessionId = null,
}: ChatShellProps) {
  const embed = variant === "embed";
  const router = useRouter();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [input, setInput] = useState("");
  const [pageDragging, setPageDragging] = useState(false);
  const [headerMenuOpen, setHeaderMenuOpen] = useState(false);
  const [hasCompanyDocs, setHasCompanyDocs] = useState(false);
  const headerMenuRef = useRef<HTMLDivElement>(null);
  const headerMenuId = useId();
  const dragDepth = useRef(0);
  const loadAttachmentsRef = useRef<(sessionId: string) => Promise<void>>(
    async () => undefined,
  );

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        if (window.localStorage.getItem(SIDEBAR_COLLAPSE_KEY) === "1") {
          setSidebarCollapsed(true);
        }
      } catch {
        // ignore
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  // Auto-collapse the rail on very narrow windows so the column stays visible.
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 900px)");
    function sync() {
      if (mq.matches) setSidebarCollapsed(true);
    }
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        SIDEBAR_COLLAPSE_KEY,
        sidebarCollapsed ? "1" : "0",
      );
    } catch {
      // ignore
    }
  }, [sidebarCollapsed]);

  const onSessionIdChange = useCallback(
    (sessionId: string | null) => {
      if (embed) return;
      if (sessionId) router.replace(`/chat/${sessionId}`);
      else router.replace("/chat");
    },
    [embed, router],
  );

  const onSessionOpened = useCallback((sessionId: string) => {
    void loadAttachmentsRef.current(sessionId);
  }, []);

  const sessionsApi = useChatSessions({
    companyId,
    initialSessionId: embed ? null : initialSessionId,
    onSessionIdChange: embed ? undefined : onSessionIdChange,
    onSessionOpened: embed ? undefined : onSessionOpened,
  });

  const {
    backendReady,
    sessions,
    activeSessionId,
    messages,
    isLoadingSessions,
    isLoadingMessages,
    isSending,
    error,
    pendingRetry,
    dismissError,
    retryLastFailed,
    openSession,
    startNewChat,
    clearMessagesLocally,
    ensureSession,
    sendMessage,
    renameSession,
    removeSession,
  } = sessionsApi;

  const attachments = useChatAttachments({
    sessionId: activeSessionId,
    companyId,
    ensureSession,
  });

  useEffect(() => {
    loadAttachmentsRef.current = attachments.loadAttachments;
  }, [attachments.loadAttachments]);

  useEffect(() => {
    if (!backendReady) return;
    let cancelled = false;
    void listDocuments(companyId)
      .then((payload) => {
        if (!cancelled) {
          setHasCompanyDocs((payload.documents?.length ?? 0) > 0);
        }
      })
      .catch(() => {
        if (!cancelled) setHasCompanyDocs(false);
      });
    return () => {
      cancelled = true;
    };
  }, [backendReady, companyId]);

  const promptsEnabled = attachments.readyCount > 0 || hasCompanyDocs;
  const isEmpty =
    messages.length === 0 &&
    attachments.uploadMessages.length === 0 &&
    !isLoadingMessages;
  const title =
    (activeSessionId &&
      sessions.find((s) => s.id === activeSessionId)?.title) ||
    "New chat";

  const subtitle = useMemo(
    () =>
      statusSubtitle({
        backendReady,
        readyCount: attachments.readyCount,
        busyCount: attachments.busyCount,
      }),
    [attachments.busyCount, attachments.readyCount, backendReady],
  );

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setMobileOpen(false);
        setHeaderMenuOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (!headerMenuOpen) return;
    function onPointerDown(event: MouseEvent) {
      if (!headerMenuRef.current?.contains(event.target as Node)) {
        setHeaderMenuOpen(false);
      }
    }
    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, [headerMenuOpen]);

  async function handleSend() {
    const trimmed = input.trim();
    if (!trimmed) return;
    setInput("");
    await sendMessage(trimmed);
  }

  async function handleNewChat() {
    setHeaderMenuOpen(false);
    attachments.clearFiles();
    setInput("");
    try {
      await startNewChat();
    } catch {
      // keep UI empty; user can retry New chat
    }
  }

  async function handleDeleteSession(sessionId: string) {
    const wasActive = activeSessionId === sessionId;
    await removeSession(sessionId);
    attachments.forgetSessionUploads(sessionId);
    if (wasActive) {
      attachments.clearFiles();
      setInput("");
    }
  }

  async function handleHeaderRename() {
    setHeaderMenuOpen(false);
    if (!activeSessionId) return;
    const next = window.prompt("Rename chat", title);
    if (!next?.trim()) return;
    await renameSession(activeSessionId, next.trim());
  }

  function handleHeaderClearMessages() {
    setHeaderMenuOpen(false);
    if (
      !window.confirm(
        "Clear messages in this view? The chat stays in Recents; use New chat for a fresh session.",
      )
    ) {
      return;
    }
    clearMessagesLocally();
    setInput("");
  }

  async function handleHeaderDelete() {
    setHeaderMenuOpen(false);
    if (!activeSessionId) {
      await handleNewChat();
      return;
    }
    if (!window.confirm(`Delete “${title}”? This cannot be undone.`)) return;
    await handleDeleteSession(activeSessionId);
  }

  function onPageDragEnter(event: DragEvent) {
    if (!event.dataTransfer?.types?.includes("Files")) return;
    event.preventDefault();
    dragDepth.current += 1;
    setPageDragging(true);
  }

  function onPageDragLeave(event: DragEvent) {
    if (!event.dataTransfer?.types?.includes("Files")) return;
    event.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setPageDragging(false);
  }

  function onPageDragOver(event: DragEvent) {
    if (!event.dataTransfer?.types?.includes("Files")) return;
    event.preventDefault();
  }

  function onPageDrop(event: DragEvent) {
    event.preventDefault();
    dragDepth.current = 0;
    setPageDragging(false);
    if (event.dataTransfer.files?.length) {
      void attachments.uploadFiles(event.dataTransfer.files);
    }
  }

  return (
    <div
      className={`chat-shell ${className}`}
      onDragEnter={onPageDragEnter}
      onDragLeave={onPageDragLeave}
      onDragOver={onPageDragOver}
      onDrop={onPageDrop}
    >
      {!embed ? (
        <ChatSidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          collapsed={sidebarCollapsed}
          mobileOpen={mobileOpen}
          isLoading={isLoadingSessions}
          statusText={
            backendReady ? "Connected" : "Backend offline"
          }
          onCollapseToggle={() => setSidebarCollapsed((value) => !value)}
          onMobileClose={() => setMobileOpen(false)}
          onNewChat={() => {
            void handleNewChat();
          }}
          onSelect={(sessionId) => {
            attachments.clearFiles();
            void openSession(sessionId);
          }}
          onRename={renameSession}
          onDelete={handleDeleteSession}
        />
      ) : null}

      <main className="chat-main">
        {pageDragging ? (
          <div className="pointer-events-none absolute inset-0 z-30 border-2 border-dashed border-sky-500 bg-sky-50/40" />
        ) : null}

        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-slate-200/80 bg-white/90 px-3 py-2.5 backdrop-blur sm:px-4">
          <div className="flex min-w-0 items-center gap-2">
            {!embed ? (
              <button
                type="button"
                onClick={() => setMobileOpen(true)}
                className="chat-mobile-menu inline-flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-400"
                aria-label="Open sidebar menu"
              >
                <MenuIcon className="h-4 w-4" />
              </button>
            ) : null}
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-slate-900">
                {title}
              </p>
              <p className="truncate text-[11px] text-slate-500">{subtitle}</p>
            </div>
          </div>

          <div ref={headerMenuRef} className="relative">
            <button
              type="button"
              onClick={() => setHeaderMenuOpen((value) => !value)}
              className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-400"
              aria-label="Chat actions"
              aria-haspopup="menu"
              aria-expanded={headerMenuOpen}
              aria-controls={headerMenuId}
            >
              <MoreVerticalIcon />
            </button>
            {headerMenuOpen ? (
              <div
                id={headerMenuId}
                role="menu"
                className="absolute top-11 right-0 z-20 w-48 overflow-hidden rounded-xl border border-slate-200 bg-white py-1 shadow-lg shadow-slate-900/10"
              >
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 disabled:opacity-40"
                  disabled={!activeSessionId}
                  onClick={() => void handleHeaderRename()}
                >
                  Rename chat
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50"
                  onClick={handleHeaderClearMessages}
                >
                  Clear messages
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full px-3 py-2 text-left text-sm text-red-700 hover:bg-red-50"
                  onClick={() => void handleHeaderDelete()}
                >
                  Delete chat
                </button>
              </div>
            ) : null}
          </div>
        </header>

        <div className="relative flex min-h-0 flex-1 flex-col">
          {isEmpty ? (
            <div className="flex flex-1 flex-col justify-center overflow-y-auto py-6 sm:py-8">
              <ChatEmptyState
                isUploading={attachments.isUploading}
                promptsEnabled={promptsEnabled}
                onFiles={(files) => void attachments.uploadFiles(files)}
                onSuggestedPrompt={(prompt) => {
                  setInput(prompt);
                  void sendMessage(prompt);
                }}
              />
              {error ? (
                <div className="mx-auto mt-4 w-full max-w-[840px] px-4">
                  <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
                    <p>{error}</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {pendingRetry ? (
                        <button
                          type="button"
                          onClick={() => void retryLastFailed()}
                          className="rounded-lg bg-amber-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-amber-800"
                        >
                          Retry
                        </button>
                      ) : null}
                      <button
                        type="button"
                        onClick={dismissError}
                        className="rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-900 hover:bg-amber-100"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                </div>
              ) : null}
              <div className="mt-6">
                <ChatComposer
                  value={input}
                  onChange={setInput}
                  onSubmit={() => void handleSend()}
                  onFiles={(files) => void attachments.uploadFiles(files)}
                  onRemoveFile={(id) => void attachments.removeFile(id)}
                  onRetryFile={(id) => void attachments.retryFile(id)}
                  files={attachments.pendingAttachments}
                  uploadError={attachments.uploadError}
                  disabled={!backendReady}
                  isSending={isSending}
                  showDropHint
                  pageDragging={pageDragging}
                />
              </div>
            </div>
          ) : (
            <>
              <div className="min-h-0 flex-1 overflow-y-auto">
                <ChatThread
                  messages={messages}
                  uploadMessages={attachments.uploadMessages}
                  isSending={isSending}
                  isLoading={isLoadingMessages}
                  sendError={error}
                  onRetry={
                    pendingRetry ? () => void retryLastFailed() : undefined
                  }
                  onDismissError={dismissError}
                />
              </div>
              <ChatComposer
                value={input}
                onChange={setInput}
                onSubmit={() => void handleSend()}
                onFiles={(files) => void attachments.uploadFiles(files)}
                onRemoveFile={(id) => void attachments.removeFile(id)}
                onRetryFile={(id) => void attachments.retryFile(id)}
                files={attachments.pendingAttachments}
                uploadError={attachments.uploadError}
                disabled={!backendReady}
                isSending={isSending}
                pageDragging={pageDragging}
              />
            </>
          )}
        </div>
      </main>
    </div>
  );
}
