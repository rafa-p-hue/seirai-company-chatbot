import type {
  CleaningInspection,
  ExtractedDocument,
  ExtractedSection,
} from "./document-types";

export type CleanedDocumentResult = {
  document: ExtractedDocument;
  inspection: CleaningInspection;
  characterCount: number;
};

export function cleanExtractedDocument(
  document: ExtractedDocument,
): CleanedDocumentResult {
  const warnings: string[] = [];
  const removedLines: string[] = [];
  const repeatedLines = detectRepeatedTopBottomLines(document.sections);
  const frequentNoise = detectFrequentNoiseLines(document.sections);

  for (const line of repeatedLines) {
    removedLines.push(`repeated-header-footer: ${line}`);
  }

  for (const line of frequentNoise) {
    if (!repeatedLines.has(line)) {
      removedLines.push(`frequent-noise: ${line}`);
    }
  }

  const noiseSet = new Set([...repeatedLines, ...frequentNoise]);
  const cleanedSections = document.sections.map((section) =>
    cleanSection(section, noiseSet, removedLines),
  );
  const characterCount = cleanedSections.reduce(
    (total, section) => total + section.text.length,
    0,
  );

  if (characterCount < 100) {
    warnings.push(
      "No selectable text was found in this document. It may be scanned and require OCR.",
    );
  }

  return {
    document: {
      ...document,
      sections: cleanedSections,
    },
    characterCount,
    inspection: {
      rawSections: document.sections,
      cleanedSections,
      removedRepeatedLines: removedLines,
      detectedHeadings: [],
      warnings,
    },
  };
}

function cleanSection(
  section: ExtractedSection,
  noiseSet: Set<string>,
  removedLog: string[],
): ExtractedSection {
  const seenLines = new Set<string>();
  const cleanedLines = section.text
    .replace(/\u0000/g, "")
    .replace(/\uFFFD/g, "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/-\n(?=\p{L})/gu, "")
    .split("\n")
    .map((line) => line.replace(/[ \t\f\v]+/g, " ").trim())
    .filter((line) => {
      if (!line) {
        return false;
      }

      const normalized = normalizeLine(line);

      if (noiseSet.has(normalized)) {
        return false;
      }

      const noiseReason = classifyNoiseLine(line);

      if (noiseReason) {
        removedLog.push(`${noiseReason}: ${line}`);
        return false;
      }

      if (seenLines.has(normalized)) {
        removedLog.push(`duplicate-in-page: ${line}`);
        return false;
      }

      seenLines.add(normalized);
      return true;
    });

  return {
    ...section,
    text: cleanedLines.join("\n").replace(/\n{3,}/g, "\n\n").trim(),
  };
}

function detectRepeatedTopBottomLines(
  sections: ExtractedSection[],
): Set<string> {
  if (sections.length < 3) {
    return new Set();
  }

  const counts = new Map<string, number>();
  const threshold = Math.max(3, Math.ceil(sections.length * 0.5));

  for (const section of sections) {
    const lines = section.text
      .split(/\r?\n/)
      .map((line) => normalizeLine(line))
      .filter(Boolean);
    const candidates = new Set([
      ...lines.slice(0, 4),
      ...lines.slice(-4),
    ]);

    for (const candidate of candidates) {
      if (candidate.length > 2) {
        counts.set(candidate, (counts.get(candidate) ?? 0) + 1);
      }
    }
  }

  return new Set(
    [...counts.entries()]
      .filter(([line, count]) => count >= threshold && isRemovableLine(line))
      .map(([line]) => line),
  );
}

function detectFrequentNoiseLines(sections: ExtractedSection[]): Set<string> {
  if (sections.length < 3) {
    return new Set();
  }

  const counts = new Map<string, number>();
  const threshold = Math.max(3, Math.ceil(sections.length * 0.4));

  for (const section of sections) {
    const unique = new Set(
      section.text
        .split(/\r?\n/)
        .map((line) => normalizeLine(line))
        .filter((line) => line.length > 2 && line.length < 80),
    );

    for (const line of unique) {
      counts.set(line, (counts.get(line) ?? 0) + 1);
    }
  }

  return new Set(
    [...counts.entries()]
      .filter(([line, count]) => {
        if (count < threshold) {
          return false;
        }

        return (
          isUiChromeLine(line) ||
          /^https?:\/\//.test(line) ||
          /page\s+\d+|\/\d+$/.test(line) ||
          line.split(" ").length <= 4
        );
      })
      .map(([line]) => line),
  );
}

function classifyNoiseLine(line: string): string | null {
  const normalized = normalizeLine(line);

  if (/^https?:\/\/\S+$/i.test(line) || /^https?:\/\//i.test(normalized)) {
    return "url";
  }

  if (/^page\s+\d+(\s+of\s+\d+)?$/i.test(normalized)) {
    return "page-counter";
  }

  if (/\b\d+\/\d+\s*$/.test(normalized) && normalized.includes("http")) {
    return "page-url-footer";
  }

  if (/\.(jpe?g|png|gif|webp|svg)(\s|$)/i.test(normalized)) {
    return "image-filename";
  }

  if (isUiChromeLine(normalized)) {
    return "ui-chrome";
  }

  if (isLikelyAdOrSidebar(normalized)) {
    return "ad-or-sidebar";
  }

  // Connection degree / people-you-may-know style lines
  if (/\b\d+(st|nd|rd|th)\b/.test(normalized) && normalized.split(" ").length <= 6) {
    return "sidebar-person";
  }

  if (/\b(also follow|also viewed|connections? also)\b/i.test(normalized)) {
    return "ad-or-sidebar";
  }

  if (normalized.length <= 2 && !/\p{L}/u.test(normalized)) {
    return "empty-symbol";
  }

  return null;
}

function isUiChromeLine(line: string): boolean {
  return (
    /^(connect|follow|message|promoted|show more|show less|join now|sign in|enhance with ai|profile language|private to you|who your viewers also viewed|i'?m looking for|hitachi is driving|find your welcome|join the conversation|money talks|terms apply|subscribe for|also follow|also viewed)$/i.test(
      line,
    ) ||
    /^(english|spanish|french|german|japanese|korean|chinese)$/i.test(line) ||
    /^[·•|]+$/.test(line)
  );
}

function isLikelyAdOrSidebar(line: string): boolean {
  if (line.length > 140) {
    return false;
  }

  return (
    /\b(promoted|welcome offer|terms apply|subscribe for \$|as high as \d[\d,]* points)\b/i.test(
      line,
    ) ||
    /\b\d+(st|nd|rd|th)\b.*\b(connections?|also follow)\b/i.test(line) ||
    /\bother connections?\b/i.test(line)
  );
}

function normalizeLine(line: string): string {
  return line
    .normalize("NFKC")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function isRemovableLine(line: string): boolean {
  return (
    /^page\s+\d+(\s+of\s+\d+)?$/.test(line) ||
    /^\d+$/.test(line) ||
    /^https?:\/\//.test(line) ||
    line.length <= 120
  );
}
