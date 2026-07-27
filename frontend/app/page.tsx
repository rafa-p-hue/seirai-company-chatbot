import Link from "next/link";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-slate-950 px-6 py-12 text-white">
      <section className="w-full max-w-2xl rounded-3xl border border-white/10 bg-white/10 p-8 shadow-2xl shadow-slate-950/30 backdrop-blur">
        <p className="mb-3 text-sm font-medium uppercase tracking-[0.25em] text-cyan-200">
          Phase 1
        </p>
        <h1 className="text-4xl font-semibold tracking-tight">
          Document Chatbot Prototype
        </h1>
        <p className="mt-4 text-base leading-7 text-slate-200">
          A starter interface for an embeddable company knowledge assistant.
          RAG, document upload, local retrieval, and future storage integrations
          will be added in phases.
        </p>
        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <Link
            href="/embed/seirai"
            className="inline-flex items-center justify-center rounded-full bg-cyan-300 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-200"
          >
            Open Chatbot
          </Link>
          <Link
            href="/admin"
            className="inline-flex items-center justify-center rounded-full border border-white/20 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/10"
          >
            Admin Placeholder
          </Link>
        </div>
      </section>
    </main>
  );
}
