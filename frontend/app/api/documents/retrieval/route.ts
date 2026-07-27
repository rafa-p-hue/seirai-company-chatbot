import { NextResponse } from "next/server";

import { getActiveDocument } from "@/lib/rag/document-memory";
import { generateGroundedAnswer } from "@/lib/rag/generate-answer";
import { publicRetrievedChunk, retrieveDocuments } from "@/lib/rag/retrieve-documents";

export async function POST(request: Request) {
  const document = getActiveDocument();

  if (!document) {
    return jsonError("No active document is loaded.", 404);
  }

  if (document.status !== "ready") {
    return jsonError("The active document is not ready for retrieval.", 409);
  }

  const payload = await request.json().catch(() => null);
  const question =
    payload && typeof payload.question === "string" ? payload.question.trim() : "";

  if (!question) {
    return jsonError("Question is required.", 400);
  }

  const retrieval = retrieveDocuments(question);
  const answer = generateGroundedAnswer(
    question,
    retrieval.finalChunks,
    retrieval.queryType,
  );

  return NextResponse.json({
    originalQuestion: retrieval.originalQuestion,
    normalizedQuestion: retrieval.normalizedQuestion,
    normalizedRetrievalQuery: retrieval.normalizedRetrievalQuery,
    queryType: retrieval.queryType,
    expandedTerms: retrieval.expandedTerms,
    questionEmbeddingDimensions: retrieval.questionEmbeddingDimensions,
    metadataFiltersApplied: retrieval.metadataFiltersApplied,
    initialCandidates: retrieval.initialCandidates.map(publicRetrievedChunk),
    rejectedCandidates: retrieval.rejectedCandidates.map(publicRetrievedChunk),
    deduplicatedCandidates:
      retrieval.deduplicatedCandidates.map(publicRetrievedChunk),
    rerankedChunks: retrieval.rerankedChunks.map(publicRetrievedChunk),
    mergedEntries: retrieval.mergedEntries.map(publicRetrievedChunk),
    finalChunks: retrieval.finalChunks.map(publicRetrievedChunk),
    finalRecords: retrieval.finalRecords,
    answerStrategy: retrieval.answerStrategy,
    answerTemplate: answer.template,
    validationNotes: answer.validationNotes ?? [],
    answer: answer.answer,
    sources: answer.sources,
  });
}

function jsonError(message: string, status: number) {
  return NextResponse.json({ error: message }, { status });
}
