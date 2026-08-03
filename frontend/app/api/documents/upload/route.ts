import { NextResponse } from "next/server";

import {
  isSupportedUploadFile,
  SUPPORTED_FORMAT_LABEL,
} from "@/lib/supported-formats";

export const runtime = "nodejs";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

function backendBase(): string {
  return (
    process.env.RAG_API_URL?.replace(/\/$/, "") ||
    process.env.NEXT_PUBLIC_RAG_API_URL?.replace(/\/$/, "") ||
    "http://127.0.0.1:8000"
  );
}

/**
 * Proxy multipart uploads to the FastAPI ingestion pipeline so all supported
 * formats (PDF, DOCX, HTML, MD, CSV, PPTX) are accepted — not PDF-only.
 */
export async function POST(request: Request) {
  try {
    const contentType = request.headers.get("content-type") ?? "";

    if (!contentType.includes("multipart/form-data")) {
      return jsonError(
        `Please send a multipart/form-data request with a file. Supported: ${SUPPORTED_FORMAT_LABEL}.`,
        415,
      );
    }

    const formData = await request.formData();
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return jsonError(
        `Please upload a supported file (${SUPPORTED_FORMAT_LABEL}).`,
        400,
      );
    }

    if (!isSupportedUploadFile(file)) {
      return jsonError(
        `Unsupported file format. Supported: ${SUPPORTED_FORMAT_LABEL}.`,
        415,
      );
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      return jsonError("Each file must be 10 MB or smaller.", 413);
    }

    if (file.size === 0) {
      return jsonError("The uploaded file is empty.", 400);
    }

    const upstream = await fetch(`${backendBase()}/api/documents/upload`, {
      method: "POST",
      body: formData,
    });

    const text = await upstream.text();
    let payload: unknown = null;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch {
      payload = { error: text || "Upload failed." };
    }

    if (!upstream.ok) {
      const message =
        (payload as { detail?: string; error?: string } | null)?.detail ||
        (payload as { detail?: string; error?: string } | null)?.error ||
        `Upload failed (${upstream.status}).`;
      return jsonError(String(message), upstream.status);
    }

    return NextResponse.json(payload, { status: upstream.status });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "The file could not be processed.";
    return jsonError(`Upload failed. ${message}`, 500);
  }
}

export async function GET() {
  return NextResponse.json(
    {
      message: `Document upload API is ready. Supported formats: ${SUPPORTED_FORMAT_LABEL}. Send multipart/form-data with a file in the 'file' field.`,
    },
    { status: 200 },
  );
}

function jsonError(message: string, status: number) {
  return NextResponse.json({ error: message }, { status });
}
