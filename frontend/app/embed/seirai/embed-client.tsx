"use client";

import Link from "next/link";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

import {
  chatWithBackend,
  checkBackendHealth,
  listDocuments,
  type ChatResponse,
} from "@/lib/rag-api";

type Source = ChatResponse["sources"][number];

type Message = {
  id: string;
  role: "assistant" | "user";
  content: string;
  sources?: Source[];
  isError?: boolean;
};

const initialMessages: Message[] = [
  {
    id: "welcome",
    role: "assistant",
    content: "Welcome! Ask me a question about this company's knowledge base.",
  },
];

export default function SeiraiEmbedPage() {
  const searchParams = useSearchParams();
  const companyId = useMemo(
    () => (searchParams.get("company_id") || "seirai").toLowerCase(),
    [searchParams],
  );

  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [documentCount, setDocumentCount] = useState(0);
  const [backendReady, setBackendReady] = useState(false);
  const [isChecking, setIsChecking] = useState(true);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    let mounted = true;

    async function refresh() {
      const healthy = await checkBackendHealth();
      if (!mounted) return;
      setBackendReady(healthy);
      if (!healthy) {
        setIsChecking(false);
        return;
      }
      try {
        const data = await listDocuments(companyId);
        if (!mounted) return;
        setDocumentCount(data.documents.length);
      } catch {
        if (!mounted) return;
        setDocumentCount(0);
      } finally {
        if (mounted) setIsChecking(false);
      }
    }

    void refresh();
    const id = window.setInterval(() => void refresh(), 5000);
    return () => {
      mounted = false;
      window.clearInterval(id);
    };
  }, [companyId]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };
    setMessages((current) => [...current, userMessage]);
    setInput("");

    if (!backendReady) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            "The RAG backend is offline. Start Qdrant and the FastAPI service, then try again.",
          isError: true,
        },
      ]);
      return;
    }

    if (documentCount === 0) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            "No documents have been ingested for this company yet. Upload a PDF or website from the Knowledge Base.",
          isError: true,
        },
      ]);
      return;
    }

    setIsLoading(true);
    try {
      const history = messages
        .filter((message) => message.id !== "welcome")
        .map((message) => ({
          role: message.role,
          content: message.content,
        }));
      history.push({ role: "user", content: trimmed });

      const data = await chatWithBackend({
        companyId,
        question: trimmed,
        conversationId,
        history,
      });
      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.answer,
          sources: data.sources ?? [],
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            error instanceof Error
              ? error.message
              : "The chatbot could not answer that question. Please try again.",
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="flex h-screen min-h-[480px] w-full flex-col bg-slate-100 text-slate-950">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3 sm:px-5">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-cyan-700">
            Company Chat
          </p>
          <h1 className="text-lg font-semibold">Document Assistant</h1>
          <p className="text-xs text-slate-500">company: {companyId}</p>
        </div>
        <button
          type="button"
          onClick={() => {
            setMessages(initialMessages);
            setConversationId(undefined);
            setInput("");
            formRef.current?.querySelector("textarea")?.focus();
          }}
          className="rounded-full border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
        >
          Clear
        </button>
      </header>

      <section aria-live="polite" className="flex-1 space-y-4 overflow-y-auto px-4 py-5 sm:px-5">
        <StatusBanner
          isChecking={isChecking}
          backendReady={backendReady}
          documentCount={documentCount}
        />

        {messages.map((message) => (
          <article
            key={message.id}
            className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm ${
                message.role === "user"
                  ? "bg-cyan-700 text-white"
                  : message.isError
                    ? "border border-amber-200 bg-amber-50 text-amber-900"
                    : "border border-slate-200 bg-white text-slate-800"
              }`}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
              {message.sources && message.sources.length > 0 ? (
                <div className="mt-3 border-t border-slate-200 pt-3">
                  <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
                    Sources
                  </p>
                  <ul className="mt-2 space-y-1 text-xs text-slate-600">
                    {message.sources.map((source) => (
                      <li key={`${source.number}-${source.document_name}`}>
                        [{source.number}]{" "}
                        {source.source_url ? (
                          <a
                            href={source.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="underline"
                          >
                            {source.document_name}
                          </a>
                        ) : (
                          source.document_name
                        )}
                        {source.page_number ? `, page ${source.page_number}` : ""}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </article>
        ))}

        {isLoading ? (
          <div className="flex justify-start">
            <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
              <span className="inline-flex items-center gap-2">
                <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-700" />
                Searching company sources...
              </span>
            </div>
          </div>
        ) : null}
      </section>

      <form
        ref={formRef}
        onSubmit={handleSubmit}
        className="flex shrink-0 gap-2 border-t border-slate-200 bg-white p-3 sm:p-4"
      >
        <label htmlFor="chat-message" className="sr-only">
          Message
        </label>
        <textarea
          id="chat-message"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              formRef.current?.requestSubmit();
            }
          }}
          placeholder="Ask about the company..."
          rows={1}
          className="min-h-11 flex-1 resize-none rounded-2xl border border-slate-300 px-4 py-3 text-sm outline-none transition placeholder:text-slate-400 focus:border-cyan-700 focus:ring-2 focus:ring-cyan-700/20"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="rounded-2xl bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Send
        </button>
      </form>
    </main>
  );
}

function StatusBanner({
  isChecking,
  backendReady,
  documentCount,
}: {
  isChecking: boolean;
  backendReady: boolean;
  documentCount: number;
}) {
  if (isChecking) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
        Checking knowledge base status...
      </div>
    );
  }

  if (!backendReady) {
    return (
      <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 shadow-sm">
        Backend offline. Start Qdrant and the FastAPI service.
      </div>
    );
  }

  if (documentCount === 0) {
    return (
      <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-900 shadow-sm">
        <p className="font-semibold">No documents ingested yet.</p>
        <Link href="/admin" className="mt-2 inline-block font-semibold underline">
          Open Knowledge Base
        </Link>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900 shadow-sm">
      Ready. {documentCount} document{documentCount === 1 ? "" : "s"} available.
    </div>
  );
}
