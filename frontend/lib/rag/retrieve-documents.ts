import { compareRecent, extractDateMetadata } from "./date-utils";
import { cosineSimilarity, createQuestionEmbedding } from "./create-embeddings";
import { getActiveDocument } from "./document-memory";
import type { DocumentChunk, StructuredRecordType } from "./document-types";
import { understandQuery, type QueryType } from "./query-understanding";
import { tokenize } from "./token-utils";

export type RetrievedChunk = DocumentChunk & {
  cosineSimilarity: number;
  vectorSimilarity: number;
  lexicalScore: number;
  headingScore: number;
  phraseScore: number;
  metadataScore: number;
  hybridScore: number;
  rerankScore: number;
  filterDecision?: "kept" | "rejected" | "boosted";
  filterReason?: string;
};

export type RetrievalInspection = {
  originalQuestion: string;
  normalizedQuestion: string;
  normalizedRetrievalQuery: string;
  queryType: QueryType;
  expandedTerms: string[];
  questionEmbeddingDimensions: number;
  metadataFiltersApplied: string[];
  initialCandidates: RetrievedChunk[];
  rejectedCandidates: RetrievedChunk[];
  deduplicatedCandidates: RetrievedChunk[];
  rerankedChunks: RetrievedChunk[];
  mergedEntries: RetrievedChunk[];
  finalChunks: RetrievedChunk[];
  finalRecords: Array<{
    recordId?: string;
    recordType?: StructuredRecordType;
    title?: string;
    organization?: string;
    dateText?: string;
    location?: string;
    pageStart?: number;
    pageEnd?: number;
  }>;
  answerStrategy: string;
};

const STOP_WORDS = new Set([
  "the", "a", "an", "is", "are", "what", "how", "does", "do", "of", "to",
  "and", "in", "for", "with", "about", "that", "this", "it", "as", "on",
  "by", "from", "where", "when", "why", "which", "who",
]);

export const INITIAL_CANDIDATE_COUNT = 24;
export const FINAL_CONTEXT_COUNT = 8;

const PREFERRED_TYPES: Partial<Record<QueryType, StructuredRecordType[]>> = {
  "employment-related": ["experience", "role", "internship"],
  "internship-related": ["internship", "experience", "role"],
  "education-related": ["education"],
  "skills-related": ["skill", "experience", "role"],
  "organization-related": ["organization", "experience", "role", "internship"],
  "chronological-request": ["experience", "role", "internship", "education"],
  "identity-overview": ["profile", "education", "experience", "role", "internship", "organization"],
  "location-related": ["experience", "role", "internship", "profile", "education"],
};

const DEPRIORITIZED_TYPES: Partial<Record<QueryType, StructuredRecordType[]>> = {
  "employment-related": ["skill", "policy"],
  "internship-related": ["skill", "policy"],
  "skills-related": ["organization", "policy"],
  "education-related": ["skill", "policy"],
};

export function normalizeRetrievalQuery(question: string): string {
  const understanding = understandQuery(question);

  return [
    understanding.normalizedQuestion,
    ...understanding.expandedTerms,
  ].join(" ").trim();
}

export function retrieveDocuments(question: string): RetrievalInspection {
  const document = getActiveDocument();
  const understanding = understandQuery(question);
  const normalizedRetrievalQuery = normalizeRetrievalQuery(question);
  const metadataFiltersApplied = describeFilters(understanding.queryType);
  const empty = {
    originalQuestion: question,
    normalizedQuestion: understanding.normalizedQuestion,
    normalizedRetrievalQuery,
    queryType: understanding.queryType,
    expandedTerms: understanding.expandedTerms,
    questionEmbeddingDimensions: 0,
    metadataFiltersApplied,
    initialCandidates: [],
    rejectedCandidates: [],
    deduplicatedCandidates: [],
    rerankedChunks: [],
    mergedEntries: [],
    finalChunks: [],
    finalRecords: [],
    answerStrategy: getAnswerStrategy(understanding.queryType),
  };

  if (!document || document.status !== "ready") {
    return empty;
  }

  if (understanding.queryType === "greeting") {
    return empty;
  }

  if (understanding.queryType === "unsupported-or-unclear" && isClearlyUnsupported(question)) {
    return empty;
  }

  const questionEmbedding = createQuestionEmbedding(normalizedRetrievalQuery);
  const queryTerms = new Set(tokenize(normalizedRetrievalQuery));

  const preferredTypes = PREFERRED_TYPES[understanding.queryType] ?? [];

  const scored = document.chunks
    .filter((chunk) => chunk.metadata.documentId === document.id && chunk.embedding)
    .map((chunk) => ({
      ...chunk,
      ...scoreChunk({
        chunk,
        question,
        queryTerms,
        questionEmbedding,
        queryType: understanding.queryType,
      }),
    }))
    .filter((chunk) => {
      const recordType = chunk.metadata.recordType ?? chunk.record?.recordType;
      const preferredHit =
        preferredTypes.length > 0 &&
        recordType &&
        preferredTypes.includes(recordType);

      return (
        chunk.lexicalScore > 0 ||
        chunk.headingScore > 0 ||
        chunk.phraseScore > 0 ||
        chunk.metadataScore > 0 ||
        chunk.vectorSimilarity >= 0.12 ||
        Boolean(preferredHit)
      );
    })
    .sort((first, second) => second.hybridScore - first.hybridScore);

  const filtered = applyMetadataFilters(scored, understanding.queryType);
  const initialCandidates = filtered.kept.slice(0, INITIAL_CANDIDATE_COUNT);
  const rejectedCandidates = filtered.rejected;
  const deduplicatedCandidates = deduplicateCandidates(initialCandidates);
  const rerankedChunks = deduplicatedCandidates
    .map((chunk) => ({
      ...chunk,
      rerankScore:
        chunk.hybridScore +
        scoreDirectRelevance(question, chunk.content) +
        (chunk.filterDecision === "boosted" ? 0.15 : 0),
    }))
    .sort((first, second) => second.rerankScore - first.rerankScore);
  const diversified = diversifyChunks(rerankedChunks, understanding.queryType);
  const mergedEntries = mergeRelatedEntries(diversified);
  const finalChunks = selectFinalChunks(mergedEntries, understanding.queryType);

  return {
    originalQuestion: question,
    normalizedQuestion: understanding.normalizedQuestion,
    normalizedRetrievalQuery,
    queryType: understanding.queryType,
    expandedTerms: understanding.expandedTerms,
    questionEmbeddingDimensions: questionEmbedding.length,
    metadataFiltersApplied,
    initialCandidates,
    rejectedCandidates,
    deduplicatedCandidates,
    rerankedChunks,
    mergedEntries,
    finalChunks,
    finalRecords: finalChunks.map((chunk) => ({
      recordId: chunk.metadata.recordId,
      recordType: chunk.metadata.recordType,
      title: chunk.metadata.title ?? chunk.record?.title,
      organization: chunk.metadata.organization ?? chunk.record?.organization,
      dateText: chunk.metadata.dateText ?? chunk.record?.dateText,
      location: chunk.metadata.location ?? chunk.record?.location,
      pageStart: chunk.metadata.pageStart,
      pageEnd: chunk.metadata.pageEnd,
    })),
    answerStrategy: getAnswerStrategy(understanding.queryType),
  };
}

function scoreChunk({
  chunk,
  question,
  queryTerms,
  questionEmbedding,
  queryType,
}: {
  chunk: DocumentChunk;
  question: string;
  queryTerms: Set<string>;
  questionEmbedding: number[];
  queryType: QueryType;
}): Omit<RetrievedChunk, keyof DocumentChunk> {
  const vectorSimilarity = cosineSimilarity(questionEmbedding, chunk.embedding ?? []);
  const lexicalScore = scoreLexicalOverlap(queryTerms, chunk.content);
  const headingScore = scoreHeadingRelevance(queryTerms, chunk);
  const phraseScore = scorePhraseMatches(question, chunk.content);
  const metadataScore = scoreMetadataRelevance(queryTerms, chunk, queryType);
  const hybridScore =
    vectorSimilarity * 0.45 +
    lexicalScore * 0.2 +
    headingScore * 0.1 +
    phraseScore * 0.1 +
    metadataScore * 0.15;

  return {
    cosineSimilarity: vectorSimilarity,
    vectorSimilarity,
    lexicalScore,
    headingScore,
    phraseScore,
    metadataScore,
    hybridScore,
    rerankScore: hybridScore,
  };
}

function applyMetadataFilters(
  chunks: RetrievedChunk[],
  queryType: QueryType,
): { kept: RetrievedChunk[]; rejected: RetrievedChunk[] } {
  const preferred = PREFERRED_TYPES[queryType] ?? [];
  const deprioritized = DEPRIORITIZED_TYPES[queryType] ?? [];
  const kept: RetrievedChunk[] = [];
  const rejected: RetrievedChunk[] = [];

  for (const chunk of chunks) {
    const recordType = chunk.metadata.recordType ?? chunk.record?.recordType;

    if (
      queryType === "internship-related" &&
      recordType &&
      recordType !== "internship" &&
      !/intern/i.test(
        `${chunk.metadata.employmentType ?? ""} ${chunk.metadata.title ?? ""} ${chunk.content}`,
      )
    ) {
      rejected.push({
        ...chunk,
        filterDecision: "rejected",
        filterReason: "Not an internship-related record for an internship query.",
      });
      continue;
    }

    if (
      queryType === "skills-related" &&
      recordType === "organization"
    ) {
      rejected.push({
        ...chunk,
        filterDecision: "rejected",
        filterReason: "Organization-only record deprioritized for skills query.",
      });
      continue;
    }

    if (preferred.length > 0 && recordType && preferred.includes(recordType)) {
      kept.push({
        ...chunk,
        hybridScore: chunk.hybridScore + 0.2,
        filterDecision: "boosted",
        filterReason: `Boosted preferred record type: ${recordType}`,
      });
      continue;
    }

    if (deprioritized.length > 0 && recordType && deprioritized.includes(recordType)) {
      kept.push({
        ...chunk,
        hybridScore: chunk.hybridScore * 0.55,
        filterDecision: "kept",
        filterReason: `Deprioritized record type: ${recordType}`,
      });
      continue;
    }

    kept.push({
      ...chunk,
      filterDecision: "kept",
      filterReason: "Passed metadata filters.",
    });
  }

  return {
    kept: kept.sort((first, second) => second.hybridScore - first.hybridScore),
    rejected,
  };
}

function describeFilters(queryType: QueryType): string[] {
  const preferred = PREFERRED_TYPES[queryType] ?? [];
  const deprioritized = DEPRIORITIZED_TYPES[queryType] ?? [];
  const filters: string[] = [];

  if (preferred.length > 0) {
    filters.push(`prefer:${preferred.join(",")}`);
  }

  if (deprioritized.length > 0) {
    filters.push(`deprioritize:${deprioritized.join(",")}`);
  }

  if (queryType === "internship-related") {
    filters.push("require-internship-signal");
  }

  if (queryType === "location-related") {
    filters.push("require-explicit-location-fields");
  }

  return filters;
}

function isClearlyUnsupported(question: string): boolean {
  return /\b(favorite food|favourite food|zodiac|blood type|password)\b/i.test(
    question,
  );
}

function scoreDirectRelevance(question: string, content: string): number {
  const questionTerms = new Set(
    tokenize(question).filter((term) => term.length > 2 && !STOP_WORDS.has(term)),
  );
  const contentTerms = tokenize(content);

  if (questionTerms.size === 0) {
    return 0;
  }

  let matches = 0;

  for (const term of contentTerms) {
    if (questionTerms.has(term)) {
      matches += 1;
    }
  }

  return matches / Math.max(questionTerms.size, 1);
}

function scoreLexicalOverlap(queryTerms: Set<string>, content: string): number {
  const contentTerms = tokenize(content);
  const lowerContent = content.toLowerCase();

  if (queryTerms.size === 0 || contentTerms.length === 0) {
    return 0;
  }

  let matches = 0;

  for (const term of queryTerms) {
    if (contentTerms.includes(term) || lowerContent.includes(term)) {
      matches += 1;
    }
  }

  return Math.min(1, matches / Math.max(queryTerms.size, 1));
}

function scoreHeadingRelevance(
  queryTerms: Set<string>,
  chunk: DocumentChunk,
): number {
  const headingTerms = tokenize(
    [
      chunk.heading,
      chunk.metadata.title,
      chunk.metadata.organization,
      chunk.sectionPath?.join(" "),
    ]
      .filter(Boolean)
      .join(" "),
  );

  if (headingTerms.length === 0) {
    return 0;
  }

  const matches = headingTerms.filter((term) => queryTerms.has(term)).length;

  return matches / headingTerms.length;
}

function scorePhraseMatches(question: string, content: string): number {
  const phrases = question
    .toLowerCase()
    .match(/\b[\p{L}\p{N}][\p{L}\p{N}\s-]{4,}\b/gu) ?? [];
  const normalizedContent = content.toLowerCase();
  const matches = phrases.filter((phrase) => normalizedContent.includes(phrase));

  return phrases.length ? matches.length / phrases.length : 0;
}

function scoreMetadataRelevance(
  queryTerms: Set<string>,
  chunk: DocumentChunk,
  queryType: QueryType,
): number {
  let score = 0;
  const recordType = chunk.metadata.recordType ?? chunk.record?.recordType;

  if (
    queryType === "chronological-request" &&
    (chunk.metadata.dateText || extractDateMetadata(chunk.content))
  ) {
    score += 0.5;
  }

  if (
    queryType === "education-related" &&
    (recordType === "education" ||
      metadataContains(chunk, [
        "education",
        "school",
        "university",
        "college",
        "degree",
        "academic",
        "bachelor",
        "master",
        "fellow",
        "scholar",
      ]))
  ) {
    score += 0.6;
  }

  if (
    queryType === "employment-related" &&
    (recordType === "experience" ||
      recordType === "role" ||
      recordType === "internship" ||
      metadataContains(chunk, ["experience", "employment", "role", "position", "job"]))
  ) {
    score += 0.6;
  }

  if (
    queryType === "internship-related" &&
    (recordType === "internship" ||
      /intern/i.test(chunk.metadata.employmentType ?? "") ||
      metadataContains(chunk, ["internship", "intern"]))
  ) {
    score += 0.8;
  }

  if (
    queryType === "skills-related" &&
    (recordType === "skill" || metadataContains(chunk, ["skill", "skills"]))
  ) {
    score += 0.7;
  }

  if (
    queryType === "organization-related" &&
    (recordType === "organization" ||
      Boolean(chunk.metadata.organization) ||
      metadataContains(chunk, ["organization", "chapter", "council"]))
  ) {
    score += 0.5;
  }

  if (
    queryType === "location-related" &&
    (chunk.metadata.location || chunk.record?.location)
  ) {
    score += 0.7;
  }

  if (
    queryType === "identity-overview" &&
    recordType &&
    ["profile", "education", "experience", "role", "internship"].includes(recordType)
  ) {
    score += 0.4;
  }

  if (queryTerms.has(chunk.contentType ?? "")) {
    score += 0.2;
  }

  return Math.min(1, score);
}

function metadataContains(chunk: DocumentChunk, terms: string[]): boolean {
  const metadataText = [
    chunk.heading,
    chunk.metadata.title,
    chunk.metadata.organization,
    chunk.metadata.recordType,
    chunk.sectionPath?.join(" "),
    chunk.contentType,
    chunk.content,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return terms.some((term) => metadataText.includes(term));
}

function deduplicateCandidates(chunks: RetrievedChunk[]): RetrievedChunk[] {
  const seen = new Set<string>();
  const deduplicated: RetrievedChunk[] = [];

  for (const chunk of chunks) {
    const key =
      chunk.metadata.recordId ??
      tokenize(chunk.content).slice(0, 40).join(" ");

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    deduplicated.push(chunk);
  }

  return deduplicated;
}

function diversifyChunks(chunks: RetrievedChunk[], queryType: QueryType): RetrievedChunk[] {
  const selected: RetrievedChunk[] = [];
  const seenSections = new Set<string>();
  const seenTitles = new Set<string>();
  const maxPerPage = queryType === "broad-summary" || queryType === "identity-overview" ? 3 : 4;
  const pageCounts = new Map<string, number>();

  for (const chunk of chunks) {
    if (selected.length >= FINAL_CONTEXT_COUNT) {
      break;
    }

    if (chunk.rerankScore < 0.05) {
      continue;
    }

    const titleKey = (chunk.metadata.title ?? chunk.heading ?? "").toLowerCase();
    if (titleKey && seenTitles.has(titleKey) && selected.length >= 2) {
      continue;
    }

    const sectionKey =
      chunk.metadata.recordId ??
      chunk.metadata.heading ??
      chunk.metadata.sectionPath?.join(">") ??
      `chunk-${chunk.metadata.chunkIndex}`;
    const pageKey = `${chunk.pageStart ?? "unknown"}`;
    const pageCount = pageCounts.get(pageKey) ?? 0;

    if (
      (queryType === "broad-summary" || queryType === "identity-overview") &&
      seenSections.has(sectionKey) &&
      selected.length >= 3
    ) {
      continue;
    }

    if (pageCount >= maxPerPage) {
      continue;
    }

    selected.push(chunk);
    seenSections.add(sectionKey);
    if (titleKey) {
      seenTitles.add(titleKey);
    }
    pageCounts.set(pageKey, pageCount + 1);
  }

  if (queryType === "chronological-request") {
    return selected.sort((first, second) =>
      compareRecent(
        extractDateMetadata(first.metadata.dateText ?? first.content),
        extractDateMetadata(second.metadata.dateText ?? second.content),
      ),
    );
  }

  return selected;
}

function mergeRelatedEntries(chunks: RetrievedChunk[]): RetrievedChunk[] {
  const merged: RetrievedChunk[] = [];

  for (const chunk of chunks) {
    const previous = merged.at(-1);

    if (previous && shouldMerge(previous, chunk)) {
      previous.content = `${previous.content}\n\n${chunk.content}`;
      previous.originalContent = `${previous.originalContent}\n\n${chunk.originalContent}`;
      previous.characterCount = previous.content.length;
      previous.tokenCount += chunk.tokenCount;
      previous.pageEnd = chunk.pageEnd ?? previous.pageEnd;
      previous.metadata.pageEnd = previous.pageEnd;
      previous.hybridScore = Math.max(previous.hybridScore, chunk.hybridScore);
      previous.rerankScore = Math.max(previous.rerankScore, chunk.rerankScore);
      continue;
    }

    merged.push({ ...chunk });
  }

  return merged;
}

function shouldMerge(first: RetrievedChunk, second: RetrievedChunk): boolean {
  if (first.metadata.recordId && second.metadata.recordId) {
    return first.metadata.recordId === second.metadata.recordId;
  }

  const adjacent = Math.abs(first.chunkIndex - second.chunkIndex) === 1;
  const sameHeading =
    first.heading && second.heading && first.heading === second.heading;
  const sameSection =
    first.sectionPath?.join(">") &&
    first.sectionPath?.join(">") === second.sectionPath?.join(">");

  return adjacent && Boolean(sameHeading || sameSection);
}

function selectFinalChunks(
  chunks: RetrievedChunk[],
  queryType: QueryType,
): RetrievedChunk[] {
  if (
    queryType === "broad-summary" ||
    queryType === "list-request" ||
    queryType === "identity-overview" ||
    queryType === "employment-related" ||
    queryType === "internship-related" ||
    queryType === "organization-related" ||
    queryType === "chronological-request"
  ) {
    return chunks.slice(0, Math.min(FINAL_CONTEXT_COUNT, Math.max(5, chunks.length)));
  }

  return chunks.slice(0, Math.min(FINAL_CONTEXT_COUNT, chunks.length));
}

function getAnswerStrategy(queryType: QueryType): string {
  switch (queryType) {
    case "greeting":
      return "greeting";
    case "identity-overview":
      return "identity-overview";
    case "broad-summary":
      return "deterministic-summary";
    case "list-request":
    case "education-related":
    case "employment-related":
    case "internship-related":
    case "organization-related":
    case "skills-related":
      return "structured-record-list";
    case "location-related":
      return "location-with-evidence";
    case "chronological-request":
      return "date-sorted-list";
    case "unsupported-or-unclear":
      return "unsupported";
    default:
      return "specific-fact";
  }
}

export function publicRetrievedChunk(chunk: RetrievedChunk) {
  return {
    chunkIndex: chunk.metadata.chunkIndex,
    pageStart: chunk.metadata.pageStart,
    pageEnd: chunk.metadata.pageEnd,
    heading: chunk.metadata.heading,
    contentType: chunk.metadata.contentType,
    recordId: chunk.metadata.recordId,
    recordType: chunk.metadata.recordType,
    title: chunk.metadata.title,
    organization: chunk.metadata.organization,
    dateText: chunk.metadata.dateText,
    location: chunk.metadata.location,
    tokenCount: chunk.tokenCount,
    cosineSimilarity: Number(chunk.cosineSimilarity.toFixed(4)),
    vectorSimilarity: Number(chunk.vectorSimilarity.toFixed(4)),
    lexicalScore: Number(chunk.lexicalScore.toFixed(4)),
    headingScore: Number(chunk.headingScore.toFixed(4)),
    phraseScore: Number(chunk.phraseScore.toFixed(4)),
    metadataScore: Number(chunk.metadataScore.toFixed(4)),
    hybridScore: Number(chunk.hybridScore.toFixed(4)),
    rerankScore: Number(chunk.rerankScore.toFixed(4)),
    filterDecision: chunk.filterDecision,
    filterReason: chunk.filterReason,
    content: chunk.content,
  };
}
