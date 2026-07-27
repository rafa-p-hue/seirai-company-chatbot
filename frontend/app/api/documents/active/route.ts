import { NextResponse } from "next/server";

import {
  clearDocument,
  getActiveDocument,
  getPublicDocumentStatus,
} from "@/lib/rag/document-memory";

export const dynamic = "force-dynamic";

export async function GET() {
  const document = getActiveDocument();
  const publicDocument = getPublicDocumentStatus();

  return NextResponse.json(
    {
      document: publicDocument,
      inspection: document?.inspection ?? null,
      records:
        document?.records?.map((record) => ({
          id: record.id,
          recordType: record.recordType,
          title: record.title,
          organization: record.organization,
          dateText: record.dateText,
          startDate: record.startDate,
          endDate: record.endDate,
          location: record.location,
          heading: record.heading,
          employmentType: record.employmentType,
          descriptionLines: record.descriptionLines,
          skills: record.skills,
          pageStart: record.pageStart,
          pageEnd: record.pageEnd,
          rawText: record.rawText,
        })) ?? [],
      rejectedChunks: document?.rejectedChunks ?? [],
      chunks:
        document?.chunks.map((chunk) => ({
          chunkIndex: chunk.metadata.chunkIndex,
          content: chunk.content,
          originalContent: chunk.originalContent,
          characterCount: chunk.characterCount,
          tokenCount: chunk.tokenCount,
          metadata: chunk.metadata,
          embeddingStatus: chunk.embeddingStatus,
          embeddingDimensions: chunk.embedding?.length ?? 0,
          embeddingError: chunk.embeddingError,
          record: chunk.record
            ? {
                id: chunk.record.id,
                recordType: chunk.record.recordType,
                title: chunk.record.title,
                organization: chunk.record.organization,
                dateText: chunk.record.dateText,
                location: chunk.record.location,
              }
            : undefined,
        })) ?? [],
      processingLog: document?.processingLog ?? [],
    },
    {
      headers: {
        "Cache-Control": "no-store",
      },
    },
  );
}

export async function DELETE() {
  clearDocument();

  return NextResponse.json({
    document: null,
    message: "Active document cleared.",
  });
}
