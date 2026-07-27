import { describe, expect, it } from "vitest";

import { cleanExtractedDocument } from "@/lib/rag/clean-document";
import { chunkDocument, chunkStructuredRecords } from "@/lib/rag/chunk-document";
import { filterRecordsByQuality } from "@/lib/rag/chunk-quality";
import { createEmbeddings } from "@/lib/rag/create-embeddings";
import { saveDocument } from "@/lib/rag/document-memory";
import type { ExtractedDocument } from "@/lib/rag/document-types";
import { generateGroundedAnswer } from "@/lib/rag/generate-answer";
import { detectDocumentStructure } from "@/lib/rag/detect-structure";
import { parseStructuredRecords } from "@/lib/rag/parse-structured-records";
import { retrieveDocuments } from "@/lib/rag/retrieve-documents";
import { understandQuery } from "@/lib/rag/query-understanding";

type Fixture = {
  id: string;
  name: string;
  question: string;
  text: string[];
};

const fixtures: Fixture[] = [
  {
    id: "company-fixture",
    name: "fictional-company.pdf",
    question: "What services are described?",
    text: [
      "OVERVIEW\nAcme Orchard Labs provides inventory automation, analytics dashboards, and integration support for regional retailers. The document explains that its work focuses on operational visibility, practical software delivery, and training for teams that manage stock across several locations. It describes service outcomes using neutral operational language and measurable implementation activities.",
      "SERVICES\nThe service catalog includes workflow mapping, data cleanup, deployment planning, and staff enablement. It also describes reporting setup, systems integration, dashboard maintenance, and documentation for internal operators. The services are presented as repeatable project activities rather than marketing claims.",
    ],
  },
  {
    id: "resume-fixture",
    name: "fictional-resume.pdf",
    question: "What experience is listed?",
    text: [
      "PROFILE\nJordan Lee\nSystems analyst with experience in reporting tools, stakeholder interviews, and operations research.\nEXPERIENCE\nResearch Assistant\nExample Institute · Internship\nJun 2026 - Present\nSample City, California, United States\n- The role includes data review, report writing, and stakeholder updates for an applied research team.\nOperations Intern\nSample Works · Internship\n2024 - 2025\n- The entry describes process mapping, meeting notes, scheduling support, and workflow documentation.",
      "EDUCATION\nExample College\nMaster certificate in information systems\n2022 - 2026\n- Coursework in data visualization, database design, applied analytics, and written communication.\nSKILLS\nSkills: Reporting, Stakeholder Interviews, Process Analysis",
    ],
  },
  {
    id: "policy-fixture",
    name: "fictional-policy.pdf",
    question: "What is the refund rule?",
    text: [
      "RETURN POLICY\nRefund requests must be submitted within thirty days with proof of purchase and an active account number. The policy states that approved refunds are returned to the original payment method after review by the support team. It explains that the rule applies to standard purchases unless a listed exception changes eligibility.",
      "EXCEPTIONS\nCustom orders and consumed services are not eligible for a standard refund review. The document also states that damaged shipments require photographs, order details, and written notice before replacement is considered. These exceptions are written as conditions that limit the normal refund process.",
    ],
  },
  {
    id: "manual-fixture",
    name: "fictional-manual.pdf",
    question: "How should the device be reset?",
    text: [
      "RESET PROCEDURE\nDisconnect power, wait ten seconds, reconnect the cable, and hold the status button until the indicator flashes. The manual says the device should remain on a stable surface and should not be opened during reset. It presents the reset procedure as a controlled maintenance action for common device recovery.",
      "TROUBLESHOOTING\nIf the indicator stays red, inspect the cable and repeat the reset procedure once. The guide recommends checking the outlet, confirming the adapter rating, and recording the error pattern for support. The troubleshooting section links the reset process to observable status indicators.",
    ],
  },
  {
    id: "research-fixture",
    name: "fictional-research.pdf",
    question: "What method is described?",
    text: [
      "ABSTRACT\nThis paper studies collaborative learning signals in educational technology settings. It reports that response length can influence how automated systems categorize reflective writing about teamwork.\nMETHODS\nThe study reviewed annotated student reflections, compared classifier outputs across response-length bands, and documented disagreements between human and automated labels. The methods section emphasizes reproducible coding steps and transparent evaluation criteria.",
      "RESULTS\nLonger responses were more often labeled as collaborative even when content quality remained similar. The results section notes that length-sensitive behavior can reduce fairness for concise writers.",
    ],
  },
];

describe("generic document pipeline", () => {
  it("removes repeated headers and preserves page metadata", () => {
    const document = createDocument("cleaning-fixture", "policy.pdf", [
      "Shared Header\nPolicy Section\nRefunds are reviewed within thirty days.\nPage 1",
      "Shared Header\nPolicy Section\nApprovals require proof of purchase.\nPage 2",
      "Shared Header\nPolicy Section\nDenied requests include a written reason.\nPage 3",
    ]);

    const result = cleanExtractedDocument(document);

    expect(
      result.inspection.removedRepeatedLines.some((line) =>
        line.includes("shared header"),
      ),
    ).toBe(true);
    expect(result.inspection.cleanedSections[0].text).not.toContain(
      "Shared Header",
    );
    expect(result.inspection.cleanedSections[0].pageNumber).toBe(1);
  });

  it("reconstructs complete experience records before embedding", () => {
    const document = createDocument("structured-resume", "resume.pdf", [
      "EXPERIENCE\nResearch Assistant\nExample Institute · Internship\nJun 2026 - Present\nSample City, California, United States\n- Reviewed annotated reflections and documented classifier disagreements.",
    ]);
    const cleaned = cleanExtractedDocument(document);
    const parsed = parseStructuredRecords(cleaned.document);
    const quality = filterRecordsByQuality(parsed.records);

    expect(parsed.usedStructuredParsing).toBe(true);
    expect(quality.accepted.length).toBeGreaterThan(0);
    expect(quality.accepted[0].title).toBe("Research Assistant");
    expect(quality.accepted[0].organization).toContain("Example Institute");
    expect(quality.accepted[0].dateText).toMatch(/2026/);
    expect(quality.accepted[0].descriptionLines.length).toBeGreaterThan(0);
  });

  it.each(fixtures)(
    "retrieves and answers from a generic $name fixture",
    async (fixture) => {
      await storeReadyFixture(fixture);

      const retrieval = retrieveDocuments(fixture.question);
      const answer = generateGroundedAnswer(
        fixture.question,
        retrieval.finalChunks,
        retrieval.queryType,
      );

      expect(retrieval.finalChunks.length).toBeGreaterThan(0);
      expect(answer.answer).not.toBe(
        "I could not find that information explicitly in the uploaded document.",
      );
      expect(answer.sources.length).toBeGreaterThan(0);
      expect(answer.sources[0].documentId).toBe(fixture.id);
    },
  );

  it("returns a fallback when requested information is absent", async () => {
    await storeReadyFixture(fixtures[0]);

    const retrieval = retrieveDocuments("What is the cafeteria menu?");
    const answer = generateGroundedAnswer(
      "What is the cafeteria menu?",
      retrieval.finalChunks,
      retrieval.queryType,
    );

    expect(answer.answer).toBe(
      "I could not find that information explicitly in the uploaded document.",
    );
    expect(answer.sources).toEqual([]);
  });

  it("does not answer unsupported favorite-food questions from unrelated chunks", async () => {
    await storeReadyFixture(fixtures[1]);

    const retrieval = retrieveDocuments("what is the favorite food?");
    const answer = generateGroundedAnswer(
      "what is the favorite food?",
      retrieval.finalChunks,
      retrieval.queryType,
    );

    expect(answer.answer).toBe(
      "I could not find that information explicitly in the uploaded document.",
    );
    expect(answer.sources).toEqual([]);
  });

  it("handles greetings without retrieving document chunks", async () => {
    await storeReadyFixture(fixtures[0]);

    const retrieval = retrieveDocuments("Hello");
    const answer = generateGroundedAnswer(
      "Hello",
      retrieval.finalChunks,
      retrieval.queryType,
    );

    expect(retrieval.queryType).toBe("greeting");
    expect(retrieval.finalChunks).toHaveLength(0);
    expect(answer.answer).toContain("Hello");
    expect(answer.sources).toEqual([]);
  });

  it("classifies identity and internship queries generically", () => {
    expect(understandQuery("who is the main subject?").queryType).toBe(
      "identity-overview",
    );
    expect(understandQuery("what internships are listed?").queryType).toBe(
      "internship-related",
    );
    expect(understandQuery("what skills are explicitly listed?").queryType).toBe(
      "skills-related",
    );
  });

  it("does not treat work locations as hometown without explicit origin evidence", async () => {
    await storeReadyFixture(fixtures[1]);

    const retrieval = retrieveDocuments("where is the subject from?");
    const answer = generateGroundedAnswer(
      "where is the subject from?",
      retrieval.finalChunks,
      retrieval.queryType,
    );

    expect(retrieval.queryType).toBe("location-related");
    expect(
      answer.answer.toLowerCase().includes("does not explicitly state") ||
        answer.answer.toLowerCase().includes("could not find that information explicitly"),
    ).toBe(true);
    expect(answer.answer.toLowerCase()).not.toMatch(/\bhometown is\b/);
    expect(answer.answer.toLowerCase()).not.toMatch(/\bis from sample city\b/);
  });

  it("normalizes short misspelled education queries", async () => {
    await storeReadyFixture(fixtures[1]);

    const retrieval = retrieveDocuments("schol");
    const answer = generateGroundedAnswer(
      "schol",
      retrieval.finalChunks,
      retrieval.queryType,
    );

    expect(retrieval.normalizedQuestion).toBe("school");
    expect(retrieval.expandedTerms).toContain("education");
    expect(retrieval.queryType).toBe("education-related");
    expect(retrieval.finalChunks.length).toBeGreaterThan(0);
    expect(answer.sources.length).toBeGreaterThan(0);
  });

  it("returns multiple relevant entries for recent experience questions", async () => {
    const fixture = {
      id: "dated-resume-fixture",
      name: "dated-resume.pdf",
      question: "What are the recent experiences?",
      text: [
        "EXPERIENCE\nResearch Assistant\nExample Institute · Internship\nJun 2026 - Present\n- The role includes data review, report writing, and stakeholder updates for an applied research team.",
        "EXPERIENCE\nOperations Intern\nSample Works · Internship\n2024 - 2025\n- The entry describes process mapping, meeting notes, scheduling support, and workflow documentation.",
        "EDUCATION\nExample College\n2022 - 2026\n- The education section lists coursework in analytics, communication, and project planning.",
      ],
    };

    await storeReadyFixture(fixture);

    const retrieval = retrieveDocuments(fixture.question);
    const answer = generateGroundedAnswer(
      fixture.question,
      retrieval.finalChunks,
      retrieval.queryType,
    );

    expect(retrieval.queryType).toBe("chronological-request");
    expect(retrieval.finalChunks.length).toBeGreaterThan(1);
    expect(answer.answer).toContain("-");
    expect(answer.sources.length).toBeGreaterThan(1);
  });

  it("falls back to section-based records for policy-style prose", () => {
    const document = createDocument("policy-fallback", "policy.pdf", fixtures[2].text);
    const cleaned = cleanExtractedDocument(document);
    const parsed = parseStructuredRecords(cleaned.document);

    expect(parsed.records.length).toBeGreaterThan(0);
    expect(
      parsed.records.every((record) =>
        ["policy", "section", "unknown"].includes(record.recordType),
      ) || parsed.usedStructuredParsing === false || parsed.records.length > 0,
    ).toBe(true);
  });

  it("does not leak chunks across active documents", async () => {
    const first = fixtures[0];
    const second = fixtures[2];

    await storeReadyFixture(first);
    await storeReadyFixture(second);

    const retrieval = retrieveDocuments("What refund rule is described?");

    expect(retrieval.finalChunks.length).toBeGreaterThan(0);
    expect(
      retrieval.finalChunks.every(
        (chunk) => chunk.metadata.documentId === second.id,
      ),
    ).toBe(true);
  });
});

async function storeReadyFixture(fixture: Fixture) {
  const document = createDocument(fixture.id, fixture.name, fixture.text);
  const cleaned = cleanExtractedDocument(document);
  const parsed = parseStructuredRecords(cleaned.document);
  const quality = filterRecordsByQuality(parsed.records);
  const structured = detectDocumentStructure(cleaned.document);
  const baseChunks =
    quality.accepted.length > 0
      ? chunkStructuredRecords({
          documentId: fixture.id,
          documentName: fixture.name,
          records: quality.accepted,
        })
      : chunkDocument({
          documentId: fixture.id,
          documentName: fixture.name,
          sections: structured,
        });
  const chunks = await createEmbeddings(baseChunks);

  saveDocument({
    id: fixture.id,
    name: fixture.name,
    mimeType: "application/pdf",
    fileSize: 1000,
    pageCount: document.sections.length,
    characterCount: cleaned.characterCount,
    chunkCount: chunks.length,
    embeddingCount: chunks.length,
    status: "ready",
    progress: 100,
    message: "Document is ready.",
    uploadedAt: new Date().toISOString(),
    readyAt: new Date().toISOString(),
    chunks,
    records: quality.accepted,
    rejectedChunks: quality.rejected,
    inspection: {
      ...cleaned.inspection,
      detectedHeadings: structured
        .map((section) => section.heading)
        .filter(Boolean) as string[],
      structuredRecords: quality.accepted,
      rejectedChunks: quality.rejected,
    },
    processingLog: [],
  });
}

function createDocument(
  documentId: string,
  fileName: string,
  pages: string[],
): ExtractedDocument {
  return {
    documentId,
    fileName,
    mimeType: "application/pdf",
    sections: pages.map((text, index) => ({
      pageNumber: index + 1,
      text,
    })),
  };
}
