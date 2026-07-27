import { Suspense } from "react";

import SeiraiEmbedPage from "./embed-client";

export default function Page() {
  return (
    <Suspense
      fallback={
        <main className="flex h-screen items-center justify-center bg-slate-100 text-slate-600">
          Loading chatbot...
        </main>
      }
    >
      <SeiraiEmbedPage />
    </Suspense>
  );
}
