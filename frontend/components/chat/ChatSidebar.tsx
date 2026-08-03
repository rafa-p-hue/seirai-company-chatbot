"use client";

import Link from "next/link";
import { FormEvent, useEffect, useId, useRef, useState } from "react";

import {
  MoreVerticalIcon,
  PencilIcon,
  PlusIcon,
  PanelLeftIcon,
  TrashIcon,
  XIcon,
} from "@/components/chat/icons";
import type { ChatSessionSummary } from "@/lib/rag-api";

function formatUpdatedAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const month = date.toLocaleString("en-US", { month: "short", timeZone: "UTC" });
  const day = date.getUTCDate();
  return `${month} ${day}`;
}

function SessionMenu({
  session,
  busy,
  onRename,
  onDelete,
}: {
  session: ChatSessionSummary;
  busy: boolean;
  onRename: () => void;
  onDelete: () => void;
}) {
  const [open, setOpen] = useState(false);
  const menuId = useId();
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    window.addEventListener("mousedown", onPointerDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative shrink-0">
      <button
        type="button"
        disabled={busy}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={menuId}
        onClick={(event) => {
          event.stopPropagation();
          setOpen((value) => !value);
        }}
        className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
        aria-label={`Actions for ${session.title}`}
      >
        <MoreVerticalIcon />
      </button>
      {open ? (
        <div
          id={menuId}
          role="menu"
          className="absolute top-8 right-0 z-30 w-40 overflow-hidden rounded-xl border border-slate-200 bg-white py-1 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50"
            onClick={(event) => {
              event.stopPropagation();
              setOpen(false);
              onRename();
            }}
          >
            <PencilIcon /> Rename
          </button>
          <button
            type="button"
            role="menuitem"
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-red-700 hover:bg-red-50"
            onClick={(event) => {
              event.stopPropagation();
              setOpen(false);
              onDelete();
            }}
          >
            <TrashIcon /> Delete
          </button>
        </div>
      ) : null}
    </div>
  );
}

function RecentsSkeleton() {
  return (
    <ul className="space-y-2 px-1" aria-hidden="true">
      {[0, 1, 2].map((item) => (
        <li key={item} className="h-10 animate-pulse rounded-xl bg-slate-100" />
      ))}
    </ul>
  );
}

function SidebarBody({
  sessions,
  activeSessionId,
  isLoading,
  showCollapse,
  onCollapseToggle,
  onMobileClose,
  onNewChat,
  onSelect,
  onRename,
  onDelete,
  statusText,
}: {
  sessions: ChatSessionSummary[];
  activeSessionId: string | null;
  isLoading?: boolean;
  showCollapse?: boolean;
  onCollapseToggle: () => void;
  onMobileClose: () => void;
  onNewChat: () => void;
  onSelect: (sessionId: string) => void;
  onRename: (sessionId: string, title: string) => Promise<unknown>;
  onDelete: (sessionId: string) => Promise<unknown>;
  statusText?: string;
}) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);

  async function submitRename(event: FormEvent, sessionId: string) {
    event.preventDefault();
    const title = renameValue.trim();
    if (!title) return;
    setBusyId(sessionId);
    try {
      await onRename(sessionId, title);
      setRenamingId(null);
    } finally {
      setBusyId(null);
    }
  }

  function confirmDelete(session: ChatSessionSummary) {
    if (!window.confirm(`Delete “${session.title}”? This cannot be undone.`)) {
      return;
    }
    setBusyId(session.id);
    void onDelete(session.id).finally(() => setBusyId(null));
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 px-3 pt-3 pb-2">
        <p className="text-sm font-semibold tracking-tight text-slate-900">
          Seirai Assistant
        </p>
        <div className="flex items-center gap-1">
          {showCollapse ? (
            <button
              type="button"
              onClick={onCollapseToggle}
              className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-800"
              aria-label="Collapse sidebar"
              title="Collapse"
            >
              <PanelLeftIcon />
            </button>
          ) : (
            <button
              type="button"
              onClick={onMobileClose}
              className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-800"
              aria-label="Close sidebar"
            >
              <XIcon className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      <div className="px-3 pb-3">
        <button
          type="button"
          onClick={() => {
            onNewChat();
            onMobileClose();
          }}
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm font-semibold text-slate-800 shadow-sm hover:bg-white"
        >
          <PlusIcon />
          New chat
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col px-2 pb-2">
        <p className="px-2 pb-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
          Recents
        </p>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {isLoading ? (
            <RecentsSkeleton />
          ) : sessions.length === 0 ? (
            <p className="px-2 py-2 text-xs text-slate-400">
              No saved conversations yet
            </p>
          ) : (
            <ul className="space-y-0.5">
              {sessions.map((session) => {
                const active = session.id === activeSessionId;
                return (
                  <li key={session.id} className="group">
                    {renamingId === session.id ? (
                      <form
                        onSubmit={(event) =>
                          void submitRename(event, session.id)
                        }
                        className="rounded-xl border border-sky-600/30 bg-white p-1.5"
                      >
                        <input
                          autoFocus
                          value={renameValue}
                          onChange={(event) =>
                            setRenameValue(event.target.value)
                          }
                          onBlur={() => setRenamingId(null)}
                          className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm outline-none focus:border-sky-600"
                          maxLength={60}
                          aria-label="Rename chat"
                        />
                      </form>
                    ) : (
                      <div
                        className={`flex w-full items-center gap-1 rounded-xl pr-1 ${
                          active
                            ? "bg-slate-100 text-slate-900 ring-1 ring-slate-200"
                            : "text-slate-700 hover:bg-slate-50"
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => {
                            onSelect(session.id);
                            onMobileClose();
                          }}
                          className="min-w-0 flex-1 px-2.5 py-2 text-left"
                        >
                          <span className="block truncate text-sm font-medium">
                            {session.title}
                          </span>
                          <span className="mt-0.5 block text-[11px] text-slate-400">
                            {formatUpdatedAt(session.updated_at)}
                          </span>
                        </button>
                        <SessionMenu
                          session={session}
                          busy={busyId === session.id}
                          onRename={() => {
                            setRenamingId(session.id);
                            setRenameValue(session.title);
                          }}
                          onDelete={() => confirmDelete(session)}
                        />
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>

      <div className="border-t border-slate-200 px-3 py-3">
        <Link
          href="/admin"
          className="block rounded-xl px-2 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-900"
          onClick={onMobileClose}
        >
          Admin
        </Link>
        {statusText ? (
          <p className="mt-1 px-2 text-[11px] text-slate-400">{statusText}</p>
        ) : null}
      </div>
    </div>
  );
}

export function ChatSidebar({
  sessions,
  activeSessionId,
  collapsed,
  mobileOpen,
  isLoading,
  statusText,
  onCollapseToggle,
  onMobileClose,
  onNewChat,
  onSelect,
  onRename,
  onDelete,
}: {
  sessions: ChatSessionSummary[];
  activeSessionId: string | null;
  collapsed: boolean;
  mobileOpen: boolean;
  isLoading?: boolean;
  statusText?: string;
  onCollapseToggle: () => void;
  onMobileClose: () => void;
  onNewChat: () => void;
  onSelect: (sessionId: string) => void;
  onRename: (sessionId: string, title: string) => Promise<unknown>;
  onDelete: (sessionId: string) => Promise<unknown>;
}) {
  const bodyProps = {
    sessions,
    activeSessionId,
    isLoading,
    onCollapseToggle,
    onMobileClose,
    onNewChat,
    onSelect,
    onRename,
    onDelete,
    statusText,
  };

  return (
    <>
      {/* Always a real left flex column — never display:none */}
      <aside
        className="chat-sidebar chat-sidebar-desktop"
        data-collapsed={collapsed ? "true" : "false"}
        aria-label="Chat sidebar"
        style={{
          display: "flex",
          flexDirection: "column",
          flexShrink: 0,
          width: collapsed ? 68 : 260,
          minWidth: collapsed ? 68 : 260,
          background: "#f8fafc",
          borderRight: "1px solid #cbd5e1",
          boxShadow: "inset -1px 0 0 #e2e8f0",
          alignSelf: "stretch",
        }}
      >
        {collapsed ? (
          <div className="flex h-full flex-col items-center gap-2 px-2 pt-3">
            <button
              type="button"
              onClick={onCollapseToggle}
              className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 shadow-sm hover:bg-slate-50"
              aria-label="Expand sidebar"
              title="Expand"
            >
              <PanelLeftIcon />
            </button>
            <button
              type="button"
              onClick={onNewChat}
              className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-700 shadow-sm hover:bg-slate-50"
              aria-label="New chat"
              title="New chat"
            >
              <PlusIcon />
            </button>
          </div>
        ) : (
          <SidebarBody {...bodyProps} showCollapse />
        )}
      </aside>

      {/* Optional mobile drawer (extra); desktop column stays visible */}
      <div
        className={`chat-sidebar-mobile-host fixed inset-0 z-40 ${
          mobileOpen ? "" : "pointer-events-none"
        }`}
        aria-hidden={!mobileOpen}
      >
        <button
          type="button"
          aria-label="Close menu overlay"
          tabIndex={mobileOpen ? 0 : -1}
          className={`absolute inset-0 bg-slate-900/35 transition-opacity ${
            mobileOpen ? "opacity-100" : "opacity-0"
          }`}
          onClick={onMobileClose}
        />
        <aside
          aria-label="Chat sidebar drawer"
          className={`absolute inset-y-0 left-0 flex w-[min(86vw,280px)] flex-col border-r border-slate-200 bg-white shadow-xl transition-transform duration-200 ${
            mobileOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <SidebarBody {...bodyProps} showCollapse={false} />
        </aside>
      </div>
    </>
  );
}
