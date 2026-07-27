import path from "node:path";
import { PDFParse } from "pdf-parse";

import type { ExtractedDocument, ExtractedSection } from "./document-types";

const PDF_WORKER_PATH = path.join(
  process.cwd(),
  "node_modules/pdf-parse/dist/pdf-parse/cjs/pdf.worker.mjs",
);

export type ExtractDocumentInput = {
  documentId: string;
  fileName: string;
  mimeType: string;
  buffer: Buffer;
};

export async function extractDocument(
  input: ExtractDocumentInput,
): Promise<ExtractedDocument> {
  if (isPdf(input)) {
    return extractPdf(input);
  }

  throw new Error(`Unsupported document type: ${input.mimeType}`);
}

function isPdf(input: ExtractDocumentInput): boolean {
  return (
    input.mimeType === "application/pdf" ||
    input.fileName.toLowerCase().endsWith(".pdf")
  );
}

async function extractPdf(
  input: ExtractDocumentInput,
): Promise<ExtractedDocument> {
  PDFParse.setWorker(PDF_WORKER_PATH);

  const parser = new PDFParse({ data: input.buffer });

  try {
    const result = await parser.getText({
      lineEnforce: true,
      pageJoiner: "",
    });

    const sections: ExtractedSection[] = result.pages.map((page) => ({
      pageNumber: page.num,
      text: page.text,
    }));

    return {
      documentId: input.documentId,
      fileName: input.fileName,
      mimeType: input.mimeType,
      sections,
    };
  } finally {
    await parser.destroy();
  }
}
