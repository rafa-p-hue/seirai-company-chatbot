import type { DocumentOverview } from "./document-types";
import type { StructuredSection } from "./detect-structure";
import { tokenize } from "./token-utils";

const COMMON_WORDS = new Set([
  "the",
  "and",
  "for",
  "with",
  "that",
  "this",
  "from",
  "are",
  "was",
  "were",
  "will",
  "have",
  "has",
  "not",
  "can",
  "all",
  "you",
  "your",
  "their",
  "about",
  "into",
  "using",
]);

export function createDocumentOverview(
  sections: StructuredSection[],
): DocumentOverview {
  const text = sections.map((section) => section.text).join("\n");
  const headings = Array.from(
    new Set(sections.map((section) => section.heading).filter(Boolean)),
  ) as string[];

  return {
    apparentDocumentType: inferDocumentType(text, headings),
    mainTopics: extractMainTopics(text),
    majorSections: headings.slice(0, 12),
    namedEntities: extractNamedEntities(text),
    summary: createExtractiveSummary(text, headings),
  };
}

function inferDocumentType(text: string, headings: string[]): string {
  const combined = `${headings.join(" ")} ${text}`.toLowerCase();

  if (/\b(resume|curriculum vitae|experience|education|skills)\b/.test(combined)) {
    return "resume or professional profile";
  }

  if (/\b(policy|procedure|compliance|eligibility|requirements)\b/.test(combined)) {
    return "policy or procedure document";
  }

  if (/\b(manual|installation|troubleshooting|configuration|specification)\b/.test(combined)) {
    return "technical manual or guide";
  }

  if (/\b(report|analysis|findings|recommendations)\b/.test(combined)) {
    return "report";
  }

  return "document";
}

function extractMainTopics(text: string): string[] {
  const counts = new Map<string, number>();

  for (const token of tokenize(text)) {
    if (token.length < 4 || COMMON_WORDS.has(token)) {
      continue;
    }

    counts.set(token, (counts.get(token) ?? 0) + 1);
  }

  return [...counts.entries()]
    .sort((first, second) => second[1] - first[1])
    .slice(0, 10)
    .map(([topic]) => topic);
}

function extractNamedEntities(text: string): string[] {
  const matches = text.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b/g) ?? [];
  const counts = new Map<string, number>();

  for (const match of matches) {
    if (match.length < 3) {
      continue;
    }

    counts.set(match, (counts.get(match) ?? 0) + 1);
  }

  return [...counts.entries()]
    .sort((first, second) => second[1] - first[1])
    .slice(0, 12)
    .map(([entity]) => entity);
}

function createExtractiveSummary(text: string, headings: string[]): string {
  const firstParagraph = text
    .split(/\n{2,}|\n/)
    .map((part) => part.trim())
    .find((part) => part.length > 80);

  if (firstParagraph) {
    return firstParagraph.slice(0, 500);
  }

  if (headings.length > 0) {
    return `This document contains sections including ${headings
      .slice(0, 5)
      .join(", ")}.`;
  }

  return "The document contains extractable text, but no concise overview could be generated locally.";
}
