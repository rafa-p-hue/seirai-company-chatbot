import type {
  ChunkContentType,
  DocumentChunk,
  StructuredRecord,
} from "./document-types";
import { extractDateMetadata } from "./date-utils";
import type { StructuredSection } from "./detect-structure";
import {
  buildRecordDisplayText,
  buildRecordEmbeddingText,
} from "./parse-structured-records";
import { countTokens, splitIntoParagraphs, splitIntoSentences } from "./token-utils";

export type ChunkDocumentOptions = {
  targetTokenMin?: number;
  targetTokenMax?: number;
  tokenOverlap?: number;
  minimumUsefulTokens?: number;
};

const DEFAULT_TARGET_TOKEN_MIN = 400;
const DEFAULT_TARGET_TOKEN_MAX = 700;
const DEFAULT_TOKEN_OVERLAP = 80;
const DEFAULT_MINIMUM_USEFUL_TOKENS = 40;

export function cleanExtractedText(text: string): string {
  return text
    .replace(/\u0000/g, "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/[ \t\f\v]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function chunkStructuredRecords(input: {
  documentId: string;
  documentName: string;
  records: StructuredRecord[];
}): DocumentChunk[] {
  return input.records.map((record, chunkIndex) => {
    const content = buildRecordDisplayText(record);
    const embeddingText = buildRecordEmbeddingText(record);
    const dateMetadata =
      extractDateMetadata(record.dateText ?? content) ??
      (record.startDate || record.endDate
        ? {
            raw: record.dateText ?? `${record.startDate ?? ""} - ${record.endDate ?? ""}`,
            start: undefined,
            end: undefined,
            isPresent: /present|current/i.test(record.endDate ?? ""),
          }
        : undefined);

    return {
      id: `${input.documentId}:${chunkIndex}`,
      documentId: input.documentId,
      documentName: input.documentName,
      chunkIndex,
      pageStart: record.pageStart,
      pageEnd: record.pageEnd,
      heading: record.heading ?? record.title,
      sectionPath: record.heading ? [record.heading] : undefined,
      contentType: mapRecordTypeToContentType(record.recordType),
      content,
      originalContent: embeddingText,
      characterCount: content.length,
      tokenCount: countTokens(content),
      metadata: {
        documentId: input.documentId,
        documentName: input.documentName,
        chunkIndex,
        pageStart: record.pageStart,
        pageEnd: record.pageEnd,
        heading: record.heading ?? record.title,
        sectionPath: record.heading ? [record.heading] : undefined,
        contentType: mapRecordTypeToContentType(record.recordType),
        dateText: record.dateText ?? dateMetadata?.raw,
        dateStart: dateMetadata?.start,
        dateEnd: dateMetadata?.end,
        isCurrent: dateMetadata?.isPresent,
        recordId: record.id,
        recordType: record.recordType,
        title: record.title,
        organization: record.organization,
        location: record.location,
        employmentType: record.employmentType,
      },
      embeddingStatus: "pending",
      record,
    } satisfies DocumentChunk;
  });
}

export function chunkDocument(
  input: {
    documentId: string;
    documentName: string;
    sections: StructuredSection[];
  },
  options: ChunkDocumentOptions = {},
): DocumentChunk[] {
  const targetTokenMin = options.targetTokenMin ?? DEFAULT_TARGET_TOKEN_MIN;
  const targetTokenMax = options.targetTokenMax ?? DEFAULT_TARGET_TOKEN_MAX;
  const tokenOverlap = options.tokenOverlap ?? DEFAULT_TOKEN_OVERLAP;
  const minimumUsefulTokens =
    options.minimumUsefulTokens ?? DEFAULT_MINIMUM_USEFUL_TOKENS;

  const chunks: DocumentChunk[] = [];

  for (const section of input.sections) {
    if (isNoiseOnly(section.text)) {
      continue;
    }

    const chunkTexts = splitStructuredSection({
      text: section.text,
      options: {
        targetTokenMin,
        targetTokenMax,
        tokenOverlap,
        minimumUsefulTokens,
      },
    });

    for (const originalContent of chunkTexts) {
      const content = section.heading
        ? `${section.heading}\n\n${originalContent}`
        : originalContent;
      const tokenCount = countTokens(content);

      if (tokenCount < minimumUsefulTokens && !containsStandaloneFact(content)) {
        continue;
      }

      const chunkIndex = chunks.length;
      const dateMetadata = extractDateMetadata(content);

      chunks.push({
        id: `${input.documentId}:${chunkIndex}`,
        documentId: input.documentId,
        documentName: input.documentName,
        chunkIndex,
        pageStart: section.pageNumber,
        pageEnd: section.pageNumber,
        heading: section.heading,
        sectionPath: section.sectionPath,
        contentType: section.contentType,
        content,
        originalContent,
        characterCount: content.length,
        tokenCount,
        metadata: {
          documentId: input.documentId,
          documentName: input.documentName,
          chunkIndex,
          pageStart: section.pageNumber,
          pageEnd: section.pageNumber,
          heading: section.heading,
          sectionPath: section.sectionPath,
          contentType: section.contentType,
          dateText: dateMetadata?.raw,
          dateStart: dateMetadata?.start,
          dateEnd: dateMetadata?.end,
          isCurrent: dateMetadata?.isPresent,
        },
        embeddingStatus: "pending",
      });
    }
  }

  return mergeRelatedShortChunks(chunks, minimumUsefulTokens);
}

function mapRecordTypeToContentType(
  recordType: StructuredRecord["recordType"],
): ChunkContentType {
  switch (recordType) {
    case "policy":
      return "policy-section";
    case "profile":
    case "experience":
    case "education":
    case "role":
    case "internship":
    case "project":
    case "skill":
    case "organization":
      return "profile-entry";
    case "section":
      return "heading-section";
    default:
      return "paragraph";
  }
}

function splitStructuredSection({
  text,
  options,
}: {
  text: string;
  options: Required<ChunkDocumentOptions>;
}): string[] {
  const tokenCount = countTokens(text);

  if (tokenCount <= options.targetTokenMax) {
    return [text];
  }

  const paragraphs = splitIntoParagraphs(text);
  const chunks = packBlocks(paragraphs, options);

  if (chunks.length > 0) {
    return chunks;
  }

  const sentences = splitIntoSentences(text);
  const sentenceChunks = packBlocks(sentences, options);

  if (sentenceChunks.length > 0) {
    return sentenceChunks;
  }

  return splitByTokenFallback(text, options);
}

function packBlocks(
  blocks: string[],
  options: Required<ChunkDocumentOptions>,
): string[] {
  const chunks: string[] = [];
  let current: string[] = [];
  let currentTokens = 0;

  for (const block of blocks) {
    const blockTokens = countTokens(block);

    if (blockTokens > options.targetTokenMax) {
      if (current.length > 0) {
        chunks.push(current.join("\n\n"));
        current = [];
        currentTokens = 0;
      }

      chunks.push(...splitStructuredSection({ text: block, options }));
      continue;
    }

    if (
      current.length > 0 &&
      currentTokens + blockTokens > options.targetTokenMax
    ) {
      chunks.push(current.join("\n\n"));
      current = createOverlapBlocks(current, options.tokenOverlap);
      currentTokens = countTokens(current.join("\n\n"));
    }

    current.push(block);
    currentTokens += blockTokens;
  }

  if (currentTokens >= options.minimumUsefulTokens || current.length > 0) {
    chunks.push(current.join("\n\n"));
  }

  return chunks.filter((chunk) => countTokens(chunk) >= options.minimumUsefulTokens);
}

function createOverlapBlocks(blocks: string[], tokenOverlap: number): string[] {
  const overlap: string[] = [];
  let tokens = 0;

  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index];
    const blockTokens = countTokens(block);

    if (tokens + blockTokens > tokenOverlap && overlap.length > 0) {
      break;
    }

    overlap.unshift(block);
    tokens += blockTokens;
  }

  return overlap;
}

function splitByTokenFallback(
  text: string,
  options: Required<ChunkDocumentOptions>,
): string[] {
  const tokens = text.split(/\s+/).filter(Boolean);
  const chunks: string[] = [];
  let cursor = 0;

  while (cursor < tokens.length) {
    const end = Math.min(cursor + options.targetTokenMax, tokens.length);
    const chunk = tokens.slice(cursor, end).join(" ");

    if (countTokens(chunk) >= options.minimumUsefulTokens) {
      chunks.push(chunk);
    }

    if (end === tokens.length) {
      break;
    }

    cursor = Math.max(cursor + 1, end - options.tokenOverlap);
  }

  return chunks;
}

function containsStandaloneFact(text: string): boolean {
  return /[:：]\s*\S+/.test(text) || /\b\d{4}\b/.test(text);
}

function isNoiseOnly(text: string): boolean {
  const tokenCount = countTokens(text);

  if (tokenCount === 0) {
    return true;
  }

  const letters = text.match(/\p{L}/gu)?.length ?? 0;

  return letters < 5;
}

function mergeRelatedShortChunks(
  chunks: DocumentChunk[],
  minimumUsefulTokens: number,
): DocumentChunk[] {
  const merged: DocumentChunk[] = [];

  for (const chunk of chunks) {
    const previous = merged.at(-1);

    if (
      previous &&
      areRelatedChunks(previous, chunk) &&
      (previous.tokenCount < minimumUsefulTokens * 2 ||
        chunk.tokenCount < minimumUsefulTokens * 2)
    ) {
      const content = `${previous.content}\n\n${chunk.content}`;
      const tokenCount = countTokens(content);
      const dateMetadata =
        extractDateMetadata(content) ??
        (previous.metadata.dateText
          ? {
              raw: previous.metadata.dateText,
              start: previous.metadata.dateStart,
              end: previous.metadata.dateEnd,
              isPresent: Boolean(previous.metadata.isCurrent),
            }
          : undefined);

      previous.content = content;
      previous.originalContent = `${previous.originalContent}\n\n${chunk.originalContent}`;
      previous.characterCount = content.length;
      previous.tokenCount = tokenCount;
      previous.pageEnd = chunk.pageEnd ?? previous.pageEnd;
      previous.metadata.pageEnd = previous.pageEnd;
      previous.metadata.dateText = dateMetadata?.raw;
      previous.metadata.dateStart = dateMetadata?.start;
      previous.metadata.dateEnd = dateMetadata?.end;
      previous.metadata.isCurrent = dateMetadata?.isPresent;
      continue;
    }

    merged.push({ ...chunk, chunkIndex: merged.length });
    const latest = merged[merged.length - 1];
    latest.id = `${latest.documentId}:${latest.chunkIndex}`;
    latest.metadata.chunkIndex = latest.chunkIndex;
  }

  return merged;
}

function areRelatedChunks(first: DocumentChunk, second: DocumentChunk): boolean {
  const sameHeading =
    first.heading && second.heading && first.heading === second.heading;
  const sameSection =
    first.sectionPath?.join(">") &&
    first.sectionPath?.join(">") === second.sectionPath?.join(">");
  const adjacent = second.chunkIndex === first.chunkIndex + 1;
  const overlappingPage =
    first.pageEnd !== undefined &&
    second.pageStart !== undefined &&
    Math.abs(second.pageStart - first.pageEnd) <= 1;

  return adjacent && overlappingPage && Boolean(sameHeading || sameSection);
}
