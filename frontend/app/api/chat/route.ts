import { NextResponse } from "next/server";

import { getActiveDocument } from "@/lib/rag/document-memory";
import { generateGroundedAnswer } from "@/lib/rag/generate-answer";
import { retrieveDocuments } from "@/lib/rag/retrieve-documents";

const MAX_QUESTION_LENGTH = 1000;
const FALLBACK_ANSWER =
  "I could not find that information explicitly in the uploaded document.";

export async function POST(request: Request) {
  let payload: unknown;

  try {
    payload = await request.json();
  } catch {
    return jsonError("Please send a JSON request body.", 400);
  }

  const question = getQuestion(payload);

  if (question === null) {
    return jsonError("Question is required and must be a string.", 400);
  }

  const trimmedQuestion = question.trim();

  if (!trimmedQuestion) {
    return jsonError("Question cannot be empty.", 400);
  }

  if (trimmedQuestion.length > MAX_QUESTION_LENGTH) {
    return jsonError("Question is too long. Please keep it under 1,000 characters.", 413);
  }

  const document = getActiveDocument();

  if (!document) {
    return jsonError(
      "No document has been loaded yet. Process a document from the Knowledge Base page first.",
      404,
    );
  }

  if (document.status === "failed") {
    return jsonError(
      document.error ?? "Document processing failed. Upload another document.",
      409,
    );
  }

  if (document.status !== "ready") {
    return jsonError(
      "The document is still being processed. Questions will be available when processing is complete.",
      409,
    );
  }

  const retrieval = retrieveDocuments(trimmedQuestion);

  if (retrieval.queryType === "greeting") {
    const answer = generateGroundedAnswer(trimmedQuestion, [], "greeting");

    return NextResponse.json({
      answer: answer.answer,
      sources: answer.sources,
    });
  }

  if (retrieval.finalChunks.length === 0) {
    return NextResponse.json({
      answer: FALLBACK_ANSWER,
      sources: [],
    });
  }

  const answer = generateGroundedAnswer(
    trimmedQuestion,
    retrieval.finalChunks,
    retrieval.queryType,
  );

  return NextResponse.json({
    answer: answer.answer || FALLBACK_ANSWER,
    sources: answer.sources,
  });
}

export async function GET() {
  return NextResponse.json(
    {
      message:
        "Chat API is ready. Send a POST request with a JSON body containing a question.",
    },
    { status: 200 },
  );
}

function getQuestion(payload: unknown): string | null {
  if (!payload || typeof payload !== "object" || !("question" in payload)) {
    return null;
  }

  const question = (payload as { question: unknown }).question;

  return typeof question === "string" ? question : null;
}

function jsonError(message: string, status: number) {
  return NextResponse.json({ error: message }, { status });
}
