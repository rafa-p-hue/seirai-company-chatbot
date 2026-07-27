import { cleanExtractedDocument } from "./clean-document";
import { chunkDocument, chunkStructuredRecords } from "./chunk-document";
import { filterRecordsByQuality } from "./chunk-quality";
import { createEmbeddings } from "./create-embeddings";
import { createDocumentOverview } from "./create-overview";
import { detectDocumentStructure } from "./detect-structure";
import { extractDocument } from "./extract-document";
import {
  updateActiveDocument,
  updateDocumentStatus,
} from "./document-memory";
import { parseStructuredRecords } from "./parse-structured-records";

export type ProcessDocumentInput = {
  documentId: string;
  fileName: string;
  mimeType: string;
  fileSize: number;
  buffer: Buffer;
};

export async function processDocument(
  input: ProcessDocumentInput,
): Promise<void> {
  try {
    let startedAt = Date.now();
    updateDocumentStatus({
      documentId: input.documentId,
      status: "extracting",
      progress: 10,
      message: "Extracting pages",
    });

    const extractedDocument = await extractDocument({
      documentId: input.documentId,
      fileName: input.fileName,
      mimeType: input.mimeType,
      buffer: input.buffer,
    });

    updateActiveDocument(input.documentId, (document) => {
      document.pageCount = extractedDocument.sections.length;
      document.inspection.rawSections = extractedDocument.sections;
    });
    updateDocumentStatus({
      documentId: input.documentId,
      status: "extracting",
      progress: 25,
      message: `Extracting: ${extractedDocument.sections.length}/${extractedDocument.sections.length} pages`,
      startedAt,
    });

    startedAt = Date.now();
    updateDocumentStatus({
      documentId: input.documentId,
      status: "cleaning",
      progress: 35,
      message: "Cleaning repeated content",
    });
    const cleaned = cleanExtractedDocument(extractedDocument);

    if (cleaned.characterCount < 100) {
      throw new Error(
        "No selectable text was found in this document. It may be scanned and require OCR.",
      );
    }

    updateActiveDocument(input.documentId, (document) => {
      document.characterCount = cleaned.characterCount;
      document.inspection = cleaned.inspection;
    });
    updateDocumentStatus({
      documentId: input.documentId,
      status: "cleaning",
      progress: 50,
      message: "Cleaning: complete",
      startedAt,
    });

    startedAt = Date.now();
    updateDocumentStatus({
      documentId: input.documentId,
      status: "chunking",
      progress: 55,
      message: "Parsing structured records",
    });

    const parsed = parseStructuredRecords(cleaned.document);
    const quality = filterRecordsByQuality(parsed.records);
    const structuredSections = detectDocumentStructure(cleaned.document);
    const detectedHeadings = Array.from(
      new Set(
        [
          ...structuredSections.map((section) => section.heading),
          ...quality.accepted.map((record) => record.heading ?? record.title),
        ].filter(Boolean),
      ),
    ) as string[];

    const chunks =
      quality.accepted.length > 0
        ? chunkStructuredRecords({
            documentId: input.documentId,
            documentName: input.fileName,
            records: quality.accepted,
          })
        : chunkDocument({
            documentId: input.documentId,
            documentName: input.fileName,
            sections: structuredSections,
          });

    if (chunks.length === 0) {
      throw new Error("No useful chunks could be created from this document.");
    }

    const overview = createDocumentOverview(structuredSections);

    updateActiveDocument(input.documentId, (document) => {
      document.chunkCount = chunks.length;
      document.chunks = chunks;
      document.records = quality.accepted;
      document.rejectedChunks = quality.rejected;
      document.overview = overview;
      document.inspection.detectedHeadings = detectedHeadings;
      document.inspection.structuredRecords = quality.accepted;
      document.inspection.rejectedChunks = quality.rejected;
      document.inspection.warnings = [
        ...document.inspection.warnings,
        ...(parsed.fallbackReason ? [parsed.fallbackReason] : []),
        ...(quality.rejected.length > 0
          ? [`Rejected ${quality.rejected.length} low-quality record(s).`]
          : []),
      ];
    });
    updateDocumentStatus({
      documentId: input.documentId,
      status: "chunking",
      progress: 70,
      message: parsed.usedStructuredParsing
        ? `Chunking: ${chunks.length} structured records`
        : `Chunking: ${chunks.length} section-based chunks`,
      startedAt,
    });

    startedAt = Date.now();
    updateDocumentStatus({
      documentId: input.documentId,
      status: "embedding",
      progress: 75,
      message: `Generating embeddings: 0/${chunks.length}`,
    });

    const embeddedChunks = await createEmbeddings(chunks, (progress) => {
      updateActiveDocument(input.documentId, (document) => {
        document.embeddingCount = progress.completed;
      });
      updateDocumentStatus({
        documentId: input.documentId,
        status: "embedding",
        progress: 75 + Math.round((progress.completed / progress.total) * 20),
        message: `Generating embeddings: ${progress.completed}/${progress.total}`,
      });
    });
    const failedEmbeddings = embeddedChunks.filter(
      (chunk) => chunk.embeddingStatus !== "complete" || !chunk.embedding,
    );

    if (failedEmbeddings.length > 0) {
      throw new Error(
        `Embedding failed for ${failedEmbeddings.length} chunk(s). The document cannot be marked ready.`,
      );
    }

    updateActiveDocument(input.documentId, (document) => {
      document.chunks = embeddedChunks;
      document.embeddingCount = embeddedChunks.length;
    });
    updateDocumentStatus({
      documentId: input.documentId,
      status: "embedding",
      progress: 95,
      message: `Embedding: ${embeddedChunks.length}/${embeddedChunks.length}`,
      startedAt,
    });

    updateDocumentStatus({
      documentId: input.documentId,
      status: "ready",
      progress: 100,
      message: "Document is ready.",
    });
  } catch (error) {
    updateDocumentStatus({
      documentId: input.documentId,
      status: "failed",
      progress: 100,
      message: "Document processing failed.",
      error:
        error instanceof Error
          ? error.message
          : "Document processing failed.",
    });
  }
}
