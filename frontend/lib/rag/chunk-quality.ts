import type { RejectedChunk, StructuredRecord } from "./document-types";
import { countTokens, tokenize } from "./token-utils";

export type ChunkQualityResult = {
  accepted: StructuredRecord[];
  rejected: RejectedChunk[];
};

export function filterRecordsByQuality(
  records: StructuredRecord[],
): ChunkQualityResult {
  const accepted: StructuredRecord[] = [];
  const rejected: RejectedChunk[] = [];
  const seenFingerprints = new Set<string>();

  for (const record of records) {
    const evaluation = evaluateRecordQuality(record, seenFingerprints);

    if (!evaluation.accepted) {
      rejected.push({
        content: record.rawText,
        reason: evaluation.reason ?? "Failed quality checks.",
        pageStart: record.pageStart,
        pageEnd: record.pageEnd,
        qualityScore: evaluation.score,
      });
      continue;
    }

    seenFingerprints.add(evaluation.fingerprint);
    accepted.push({
      ...record,
    });
  }

  return { accepted, rejected };
}

export function evaluateRecordQuality(
  record: StructuredRecord,
  seenFingerprints: Set<string> = new Set(),
): {
  accepted: boolean;
  reason?: string;
  score: number;
  fingerprint: string;
} {
  const text = record.rawText.trim();
  const words = tokenize(text);
  const fingerprint = words.slice(0, 24).join(" ");
  let score = 1;

  if (words.length < 4) {
    return {
      accepted: false,
      reason: "Too few words to be a useful record.",
      score: 0.1,
      fingerprint,
    };
  }

  if (seenFingerprints.has(fingerprint)) {
    return {
      accepted: false,
      reason: "Highly duplicated with another accepted record.",
      score: 0.2,
      fingerprint,
    };
  }

  if (isMostlyUrls(text)) {
    return {
      accepted: false,
      reason: "Content is mostly URLs.",
      score: 0.1,
      fingerprint,
    };
  }

  if (isMostlyNavigation(text)) {
    return {
      accepted: false,
      reason: "Content looks like navigation or UI chrome.",
      score: 0.15,
      fingerprint,
    };
  }

  if (isNameOnly(record)) {
    return {
      accepted: false,
      reason: "Record is only a name.",
      score: 0.2,
      fingerprint,
    };
  }

  if (isTitleOnly(record)) {
    return {
      accepted: false,
      reason: "Record is only a title/heading.",
      score: 0.25,
      fingerprint,
    };
  }

  if (isDateOnly(record)) {
    return {
      accepted: false,
      reason: "Record is only a date.",
      score: 0.2,
      fingerprint,
    };
  }

  if (hasHighTermRepetition(words)) {
    score -= 0.35;
  }

  if (!hasUsefulFactualContent(record)) {
    return {
      accepted: false,
      reason: "Lacks useful factual content.",
      score: Math.max(0.1, score - 0.5),
      fingerprint,
    };
  }

  if (isGrammaticallyIncomplete(record) && words.length < 12) {
    return {
      accepted: false,
      reason: "Grammatically incomplete fragment.",
      score: 0.3,
      fingerprint,
    };
  }

  if (countTokens(text) < 8 && !record.dateText && !record.organization) {
    score -= 0.2;
  }

  if (score < 0.45) {
    return {
      accepted: false,
      reason: "Quality score below acceptance threshold.",
      score,
      fingerprint,
    };
  }

  return {
    accepted: true,
    score,
    fingerprint,
  };
}

function isMostlyUrls(text: string): boolean {
  const tokens = text.split(/\s+/).filter(Boolean);
  const urls = tokens.filter((token) => /^https?:\/\//i.test(token));
  return tokens.length > 0 && urls.length / tokens.length >= 0.4;
}

function isMostlyNavigation(text: string): boolean {
  const lower = text.toLowerCase();
  const navHits = [
    "connect",
    "follow",
    "message",
    "promoted",
    "show more",
    "profile language",
    "enhance with ai",
    "who your viewers also viewed",
  ].filter((term) => lower.includes(term)).length;

  return navHits >= 2 || /^(home|about|menu|search|settings)$/i.test(text.trim());
}

function isNameOnly(record: StructuredRecord): boolean {
  return (
    Boolean(record.title) &&
    !record.organization &&
    !record.dateText &&
    !record.location &&
    record.descriptionLines.length === 0 &&
    !record.skills?.length &&
    (record.title?.split(/\s+/).length ?? 0) <= 5
  );
}

function isTitleOnly(record: StructuredRecord): boolean {
  return (
    Boolean(record.title || record.heading) &&
    !record.organization &&
    !record.dateText &&
    record.descriptionLines.length === 0 &&
    !record.skills?.length
  );
}

function isDateOnly(record: StructuredRecord): boolean {
  return (
    Boolean(record.dateText) &&
    !record.title &&
    !record.organization &&
    record.descriptionLines.length === 0
  );
}

function hasHighTermRepetition(words: string[]): boolean {
  if (words.length < 6) {
    return false;
  }

  const counts = new Map<string, number>();

  for (const word of words) {
    counts.set(word, (counts.get(word) ?? 0) + 1);
  }

  const max = Math.max(...counts.values());
  return max / words.length >= 0.45;
}

function hasUsefulFactualContent(record: StructuredRecord): boolean {
  if (record.organization || record.dateText || record.location) {
    return true;
  }

  if ((record.skills?.length ?? 0) > 0) {
    return true;
  }

  if (record.descriptionLines.some((line) => line.split(/\s+/).length >= 6)) {
    return true;
  }

  if (record.recordType === "profile" && record.descriptionLines.length > 0) {
    return true;
  }

  if (record.recordType === "section" || record.recordType === "policy") {
    return countTokens(record.rawText) >= 20;
  }

  return countTokens(record.rawText) >= 16;
}

function isGrammaticallyIncomplete(record: StructuredRecord): boolean {
  if (record.title && (record.organization || record.dateText)) {
    return false;
  }

  const text = record.descriptionLines.join(" ") || record.rawText;
  const trimmed = text.trim();

  if (!trimmed) {
    return true;
  }

  if (/^(and|or|but|with|for|to|of|in|on|at)\b/i.test(trimmed)) {
    return true;
  }

  if (trimmed.length < 40 && !/[.!?]$/.test(trimmed) && !record.title) {
    return true;
  }

  return false;
}
