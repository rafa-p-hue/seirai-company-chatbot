import type {
  ChunkContentType,
  ExtractedDocument,
  ExtractedSection,
} from "./document-types";

export type StructuredSection = ExtractedSection & {
  heading?: string;
  sectionPath: string[];
  contentType: ChunkContentType;
};

export function detectDocumentStructure(
  document: ExtractedDocument,
): StructuredSection[] {
  const sectionPath: string[] = [];

  return document.sections.flatMap((section) => {
    const blocks = splitSectionIntoBlocks(section.text);

    return blocks.map((block) => {
      const heading = detectHeading(block);

      if (heading) {
        sectionPath.splice(0, sectionPath.length, heading);
      }

      return {
        ...section,
        text: block,
        heading: heading ?? sectionPath.at(-1),
        sectionPath: [...sectionPath],
        contentType: classifyContent(block, heading),
      };
    });
  });
}

export function detectHeading(block: string): string | undefined {
  const firstLine = block.split("\n")[0]?.trim();

  if (!firstLine || firstLine.length > 120) {
    return undefined;
  }

  if (/^\d+(\.\d+)*\s+\S+/.test(firstLine)) {
    return firstLine;
  }

  if (/^[A-Z][\p{L}\p{N}\s,&:/-]{2,80}$/u.test(firstLine)) {
    const words = firstLine.split(/\s+/);
    const upperLikeWords = words.filter(
      (word) => word === word.toUpperCase() || /^[A-Z]/.test(word),
    );

    if (upperLikeWords.length / words.length >= 0.75) {
      return firstLine;
    }
  }

  return undefined;
}

export function classifyContent(
  block: string,
  heading?: string,
): ChunkContentType {
  const lines = block.split("\n").filter(Boolean);
  const listLines = lines.filter((line) => /^[-*•]|\d+[.)]\s/.test(line));
  const tableLikeLines = lines.filter(
    (line) => line.includes("|") || line.split(/\s{2,}/).length >= 3,
  );

  if (heading && lines.length > 1) {
    return "heading-section";
  }

  if (listLines.length >= Math.max(2, Math.ceil(lines.length * 0.5))) {
    return "list";
  }

  if (tableLikeLines.length >= Math.max(2, Math.ceil(lines.length * 0.5))) {
    return "table";
  }

  if (/\b(policy|procedure|requirements?|rules?)\b/i.test(block)) {
    return "policy-section";
  }

  if (/\b(experience|education|skills?|profile|summary)\b/i.test(block)) {
    return "profile-entry";
  }

  return lines.length > 0 ? "paragraph" : "unknown";
}

function splitSectionIntoBlocks(text: string): string[] {
  return text
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);
}
