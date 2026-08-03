"use client";

import { useMemo } from "react";
import { useSearchParams } from "next/navigation";

import { ChatShell } from "@/components/chat/ChatShell";

export default function SeiraiEmbedPage() {
  const searchParams = useSearchParams();
  const companyId = useMemo(
    () => (searchParams.get("company_id") || "seirai").toLowerCase(),
    [searchParams],
  );

  return <ChatShell companyId={companyId} variant="embed" />;
}
