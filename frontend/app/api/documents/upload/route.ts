import { NextResponse } from "next/server";
import { randomUUID } from "node:crypto";

import { createUploadedDocument } from "@/lib/rag/document-memory";
import { processDocument } from "@/lib/rag/process-document";

export const runtime = "nodejs";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

export async function POST(request: Request) {
  try {
    const contentType = request.headers.get("content-type") ?? "";

    if (!contentType.includes("multipart/form-data")) {
      return jsonError(
        "Please send a multipart/form-data request with one PDF file.",
        415,
      );
    }

    const formData = await request.formData();
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return jsonError("Please upload one PDF file.", 400);
    }

    if (
      file.type !== "application/pdf" &&
      !file.name.toLowerCase().endsWith(".pdf")
    ) {
      return jsonError("Only PDF files can be processed.", 415);
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      return jsonError("PDF files must be 10 MB or smaller.", 413);
    }

    if (file.size === 0) {
      return jsonError("The uploaded PDF is empty.", 400);
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    const document = createUploadedDocument({
      id: randomUUID(),
      name: file.name,
      mimeType: file.type || "application/pdf",
      fileSize: file.size,
      uploadedAt: new Date().toISOString(),
    });

    void processDocument({
      documentId: document.id,
      fileName: document.name,
      mimeType: document.mimeType,
      fileSize: document.fileSize,
      buffer,
    });

    return NextResponse.json(
      {
        document: {
          id: document.id,
          name: document.name,
          status: document.status,
          progress: document.progress,
          message: document.message,
        },
      },
      { status: 202 },
    );
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "The PDF could not be processed.";

    return jsonError(`The PDF could not be processed. ${message}`, 500);
  }
}

export async function GET() {
  return NextResponse.json(
    {
      message:
        "Document upload API is ready. Send multipart/form-data with one PDF in the 'file' field.",
    },
    { status: 200 },
  );
}

function jsonError(message: string, status: number) {
  return NextResponse.json({ error: message }, { status });
}
