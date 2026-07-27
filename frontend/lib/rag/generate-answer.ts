import type { RetrievedChunk } from "./retrieve-documents";
import type { AnswerGenerator, GroundedAnswerInput } from "./providers";
import type { QueryType } from "./query-understanding";
import type { StructuredRecord } from "./document-types";
import { splitIntoSentences, tokenize } from "./token-utils";

export type AnswerSource = {
  documentId: string;
  documentName: string;
  pageStart?: number;
  pageEnd?: number;
  heading?: string;
  chunkIndex: number;
  recordId?: string;
  recordType?: string;
};

export type GroundedAnswer = {
  answer: string;
  sources: AnswerSource[];
  template?: string;
  validationNotes?: string[];
};

const FALLBACK_ANSWER =
  "I could not find that information explicitly in the uploaded document.";

const STOP_WORDS = new Set([
  "the", "a", "an", "is", "are", "what", "how", "does", "do", "of", "to",
  "and", "in", "for", "with", "about", "that", "this", "it", "as", "on",
  "by", "from", "where", "when", "why", "which", "who",
]);

const NAVIGATION_PATTERNS = [
  /enhance with ai/i,
  /profile language/i,
  /who your viewers also viewed/i,
  /\bpromoted\b/i,
  /\bshow more\b/i,
  /about accessibility/i,
];

export function generateGroundedAnswer(
  question: string,
  chunks: RetrievedChunk[],
  queryType: QueryType = "specific-fact",
): GroundedAnswer {
  if (queryType === "greeting") {
    return {
      answer: "Hello! Ask me anything about the currently uploaded document.",
      sources: [],
      template: "greeting",
    };
  }

  if (queryType === "unsupported-or-unclear" && isUnsupportedTopic(question)) {
    return {
      answer: FALLBACK_ANSWER,
      sources: [],
      template: "unsupported",
    };
  }

  if (chunks.length === 0) {
    return {
      answer: FALLBACK_ANSWER,
      sources: [],
      template: "fallback",
    };
  }

  const candidates = [...chunks];
  let attempt = 0;
  let lastValidationNotes: string[] = [];

  while (attempt < 3 && candidates.length > 0) {
    const answer = buildAnswerForType(question, candidates, queryType);
    const validation = validateAnswer(answer.answer, queryType, candidates);

    if (validation.ok) {
      return {
        ...answer,
        validationNotes: validation.notes,
      };
    }

    lastValidationNotes = validation.notes;
    // Drop weakest / first offending chunk and regenerate
    candidates.pop();
    attempt += 1;
  }

  return {
    answer: FALLBACK_ANSWER,
    sources: [],
    template: "fallback-after-validation",
    validationNotes: lastValidationNotes,
  };
}

export const localDeterministicAnswerGenerator: AnswerGenerator = {
  name: "local-deterministic-grounded-answer-v1",
  async generateAnswer(input: GroundedAnswerInput) {
    return generateGroundedAnswer(
      input.question,
      input.retrieval.finalChunks,
      input.retrieval.queryType,
    );
  },
};

function buildAnswerForType(
  question: string,
  chunks: RetrievedChunk[],
  queryType: QueryType,
): GroundedAnswer {
  switch (queryType) {
    case "identity-overview":
      return createIdentityOverview(chunks);
    case "employment-related":
    case "chronological-request":
      return createRoleListAnswer(
        chunks,
        "The document lists these roles:",
        ["experience", "role", "internship"],
      );
    case "internship-related":
      return createRoleListAnswer(
        chunks,
        "The document lists these internships:",
        ["internship"],
        { requireInternshipSignal: true },
      );
    case "education-related":
      return createEducationAnswer(chunks);
    case "skills-related":
      return createSkillsAnswer(chunks);
    case "organization-related":
      return createOrganizationAnswer(chunks);
    case "location-related":
      return createLocationAnswer(question, chunks);
    case "broad-summary":
      return createSummaryAnswer(chunks);
    case "list-request":
      return createRoleListAnswer(chunks, "The document lists these relevant entries:");
    default:
      return createSpecificFactAnswer(question, chunks);
  }
}

function createIdentityOverview(chunks: RetrievedChunk[]): GroundedAnswer {
  const records = dedupeRecords(recordsFromChunks(chunks)).filter(
    (record) =>
      Boolean(record.title) &&
      !/^\|/.test(record.title ?? "") &&
      !/\+\d+\s*skills?/i.test(record.title ?? "") &&
      (record.organization
        ? !/^\|/.test(record.organization) && record.organization.length < 80
        : true),
  );
  const profile = records.find((record) => record.recordType === "profile");
  const education = records.find((record) => record.recordType === "education");
  const roles = records
    .filter((record) =>
      ["experience", "role", "internship"].includes(record.recordType),
    )
    .filter((record) => Boolean(record.dateText || record.organization))
    .slice(0, 6);
  const organizations = unique(
    roles
      .map((record) => record.organization)
      .filter((value): value is string => typeof value === "string")
      .filter(
        (value) =>
          value.length < 80 &&
          !/^\|/.test(value) &&
          !/\b(and personal issues|while enforcing)\b/i.test(value),
      ),
  );

  const sentences: string[] = [];

  if (profile?.title && profile.descriptionLines[0]) {
    sentences.push(
      `${profile.title} is described as ${trimTrailingPunctuation(profile.descriptionLines[0])}.`,
    );
  } else if (profile?.title) {
    sentences.push(`The document identifies ${profile.title} as the primary subject.`);
  } else if (roles[0]?.title) {
    sentences.push(
      `The document describes a subject associated with roles such as ${roles[0].title}${roles[0].organization ? ` at ${roles[0].organization}` : ""}.`,
    );
  }

  if (education) {
    const educationBits = [
      education.title,
      education.organization,
      education.dateText,
    ].filter(Boolean);
    if (educationBits.length > 0) {
      sentences.push(`Education-related details include ${educationBits.join(", ")}.`);
    }
  }

  if (roles.length > 0) {
    const roleSummaries = roles.slice(0, 3).map((role) => {
      const bits = [role.title, role.organization, role.dateText].filter(Boolean);
      return bits.join(" — ");
    });
    sentences.push(`Recent or listed roles include ${roleSummaries.join("; ")}.`);
  }

  if (organizations.length > 0 && sentences.length < 4) {
    sentences.push(
      `Organizations mentioned in the document include ${organizations.slice(0, 4).join(", ")}.`,
    );
  }

  const answer = sentences.slice(0, 4).join(" ").trim();

  if (!answer || isNameOnlyOverview(answer, profile?.title)) {
    return {
      answer: FALLBACK_ANSWER,
      sources: [],
      template: "identity-overview",
    };
  }

  return {
    answer,
    sources: sourcesFromChunks(chunks.slice(0, 6)),
    template: "identity-overview",
  };
}

function createEducationAnswer(chunks: RetrievedChunk[]): GroundedAnswer {
  const records = dedupeRecords(recordsFromChunks(chunks)).filter((record) => {
    if (record.recordType === "education") {
      return true;
    }

    const text = `${record.title ?? ""} ${record.organization ?? ""} ${record.heading ?? ""} ${record.rawText}`;
    return /\b(university|college|school|bachelor|master|degree|education|fellow|scholar|honors|academic)\b/i.test(
      text,
    );
  });

  if (records.length === 0) {
    return {
      answer:
        "The document does not clearly provide a dedicated education section with institution and program details.",
      sources: sourcesFromChunks(chunks.slice(0, 3)),
      template: "education-missing",
    };
  }

  const lines = records.slice(0, 8).map((record) => {
    const headline = [record.title, record.organization, record.dateText]
      .filter(Boolean)
      .join(" — ");
    return `- ${headline}`;
  });

  return {
    answer: ["The document lists this education-related information:", ...lines].join(
      "\n",
    ),
    sources: sourcesFromRecords(records.slice(0, 8), chunks),
    template: "education-list",
  };
}

function createRoleListAnswer(
  chunks: RetrievedChunk[],
  introduction: string,
  preferredTypes?: string[],
  options?: { requireInternshipSignal?: boolean },
): GroundedAnswer {
  const records = recordsFromChunks(chunks)
    .filter((record) => {
      if (options?.requireInternshipSignal) {
        return (
          record.recordType === "internship" ||
          /intern/i.test(record.employmentType ?? "") ||
          /intern/i.test(record.title ?? "") ||
          /intern/i.test(record.rawText)
        );
      }

      if (!preferredTypes || preferredTypes.length === 0) {
        return true;
      }

      return preferredTypes.includes(record.recordType) || Boolean(record.title);
    })
    .filter((record) => Boolean(record.title || record.organization))
    .filter(
      (record) =>
        !/^\|/.test(record.title ?? "") &&
        !/\+\d+\s*skills?/i.test(record.title ?? "") &&
        (record.title?.trim().length ?? 0) > 1,
    );

  const deduped = dedupeRecords(records).slice(0, 8);

  // For internship lists, require an explicit internship signal on the title or type.
  const filtered =
    options?.requireInternshipSignal
      ? deduped.filter(
          (record) =>
            record.recordType === "internship" ||
            /intern/i.test(record.employmentType ?? "") ||
            /intern/i.test(record.title ?? ""),
        )
      : deduped;

  if (filtered.length === 0) {
    return {
      answer: FALLBACK_ANSWER,
      sources: [],
      template: "structured-record-list",
    };
  }

  const lines = filtered.map((record) => formatRoleEntry(record));
  const answer = [introduction, ...lines].join("\n");

  if (filtered.length >= 5) {
    return {
      answer: `${answer}\n\nAsk if you want additional matching entries.`,
      sources: sourcesFromRecords(filtered, chunks),
      template: "structured-record-list",
    };
  }

  return {
    answer,
    sources: sourcesFromRecords(filtered, chunks),
    template: "structured-record-list",
  };
}

function formatRoleEntry(record: StructuredRecord): string {
  const headline = [
    record.title ?? "Untitled role",
    record.organization,
    record.dateText,
  ]
    .filter(Boolean)
    .join(" — ");

  const description = record.descriptionLines
    .map((line) => line.trim())
    .filter((line) => line.length > 20)
    .slice(0, 1)
    .join(" ");

  if (description) {
    return `- ${headline}\n  ${truncate(description, 220)}`;
  }

  return `- ${headline}`;
}

function createSkillsAnswer(chunks: RetrievedChunk[]): GroundedAnswer {
  const skills = unique(
    chunks.flatMap((chunk) => {
      const fromRecord = chunk.record?.skills ?? [];
      const fromText = extractSkillsFromText(chunk.content);
      return [...fromRecord, ...fromText];
    }),
  ).filter((skill) => !/^\+\d+/.test(skill));

  if (skills.length === 0) {
    // Fall back to role descriptions that mention concrete abilities
    const abilityLines = chunks
      .flatMap((chunk) => chunk.record?.descriptionLines ?? splitIntoSentences(chunk.content))
      .filter((line) =>
        /\b(skill|python|java|research|design|analysis|communication|leadership|management)\b/i.test(
          line,
        ),
      )
      .slice(0, 5);

    if (abilityLines.length === 0) {
      return {
        answer: FALLBACK_ANSWER,
        sources: [],
        template: "skills-list",
      };
    }

    return {
      answer: [
        "The document mentions these skill-related details:",
        ...abilityLines.map((line) => `- ${truncate(line, 180)}`),
      ].join("\n"),
      sources: sourcesFromChunks(chunks.slice(0, 5)),
      template: "skills-list",
    };
  }

  return {
    answer: [
      "The document explicitly lists these skills:",
      ...skills.slice(0, 20).map((skill) => `- ${skill}`),
    ].join("\n"),
    sources: sourcesFromChunks(chunks.slice(0, 5)),
    template: "skills-list",
  };
}

function createOrganizationAnswer(chunks: RetrievedChunk[]): GroundedAnswer {
  const organizations = unique(
    chunks
      .map((chunk) => chunk.record?.organization ?? chunk.metadata.organization)
      .filter((value): value is string => Boolean(value)),
  );

  if (organizations.length === 0) {
    return {
      answer: FALLBACK_ANSWER,
      sources: [],
      template: "organization-list",
    };
  }

  const lines = organizations.slice(0, 10).map((organization) => {
    const related = chunks.find(
      (chunk) =>
        chunk.record?.organization === organization ||
        chunk.metadata.organization === organization,
    );
    const role = related?.record?.title ?? related?.metadata.title;
    const dates = related?.record?.dateText ?? related?.metadata.dateText;
    const detail = [role, dates].filter(Boolean).join(", ");
    return detail ? `- ${organization} (${detail})` : `- ${organization}`;
  });

  return {
    answer: ["The document lists these organizations:", ...lines].join("\n"),
    sources: sourcesFromChunks(chunks.slice(0, 8)),
    template: "organization-list",
  };
}

function createLocationAnswer(
  question: string,
  chunks: RetrievedChunk[],
): GroundedAnswer {
  const asksHometown = /\b(hometown|from|born|birthplace|origin|where .+ from)\b/i.test(
    question,
  );
  const explicitOrigin = findExplicitOriginEvidence(chunks);

  if (asksHometown) {
    if (explicitOrigin) {
      return {
        answer: `The document explicitly indicates location/origin information: ${explicitOrigin.text}.`,
        sources: sourcesFromChunks([explicitOrigin.chunk]),
        template: "location-explicit",
      };
    }

    const workLocations = unique(
      chunks
        .map((chunk) => chunk.record?.location ?? chunk.metadata.location)
        .filter((value): value is string => typeof value === "string")
        .filter(
          (value) =>
            value.length < 80 &&
            !/^-/.test(value) &&
            /,/.test(value) &&
            value.split(/\s+/).length <= 8,
        ),
    );

    if (workLocations.length > 0) {
      return {
        answer: `The document does not explicitly state where the person is from. It does list experience or activity in ${workLocations.slice(0, 5).join(", ")}, but those locations do not confirm a hometown.`,
        sources: sourcesFromChunks(chunks.slice(0, 5)),
        template: "location-no-hometown",
      };
    }

    return {
      answer: FALLBACK_ANSWER,
      sources: [],
      template: "location-missing",
    };
  }

  const locations = unique(
    chunks
      .map((chunk) => chunk.record?.location ?? chunk.metadata.location)
      .filter((value): value is string => typeof value === "string")
      .filter(
        (value) =>
          value.length < 80 &&
          !/^-/.test(value) &&
          /,/.test(value) &&
          value.split(/\s+/).length <= 8,
      ),
  );

  if (locations.length === 0) {
    return {
      answer: FALLBACK_ANSWER,
      sources: [],
      template: "location-missing",
    };
  }

  return {
    answer: [
      "The document lists these locations in experience or activity entries:",
      ...locations.slice(0, 8).map((location) => `- ${location}`),
    ].join("\n"),
    sources: sourcesFromChunks(chunks.slice(0, 6)),
    template: "location-list",
  };
}

function findExplicitOriginEvidence(
  chunks: RetrievedChunk[],
): { text: string; chunk: RetrievedChunk } | null {
  for (const chunk of chunks) {
    const text = chunk.content;
    const match = text.match(
      /\b((?:born in|hometown(?:\s+is)?|originally from|based in|resides? in|lives? in)\s+[^\n.]{2,60})/i,
    );

    if (match) {
      return { text: match[1].trim(), chunk };
    }
  }

  return null;
}

function createSummaryAnswer(chunks: RetrievedChunk[]): GroundedAnswer {
  const themes = getFrequentTerms(chunks.map((chunk) => chunk.content).join(" "));
  const headings = Array.from(
    new Set(
      chunks
        .map((chunk) => chunk.record?.heading ?? chunk.heading)
        .filter(Boolean),
    ),
  );
  const summarySentences = chunks
    .flatMap((chunk) =>
      chunk.record?.descriptionLines?.length
        ? chunk.record.descriptionLines
        : splitIntoSentences(chunk.content),
    )
    .filter((sentence) => sentence.length > 40)
    .slice(0, 4);

  const answerParts = [
    "The document appears to cover several main topics.",
    headings.length > 0
      ? `Major sections include ${headings.slice(0, 6).join(", ")}.`
      : "",
    themes.length > 0
      ? `Recurring terms include ${themes.slice(0, 8).join(", ")}.`
      : "",
    summarySentences.length > 0
      ? `Relevant details include: ${summarySentences.join(" ")}`
      : "",
  ].filter(Boolean);

  return {
    answer: limitAnswerLength(answerParts.join(" ")),
    sources: sourcesFromChunks(chunks.slice(0, 6)),
    template: "deterministic-summary",
  };
}

function createSpecificFactAnswer(
  question: string,
  chunks: RetrievedChunk[],
): GroundedAnswer {
  const questionTerms = new Set(
    tokenize(question).filter((term) => term.length > 2 && !STOP_WORDS.has(term)),
  );
  const lowerQuestion = question.toLowerCase();

  const recordAnswers = recordsFromChunks(chunks)
    .map((record) => ({
      record,
      score:
        scoreRecordAgainstQuestion(record, questionTerms) +
        (questionTerms.size > 0 &&
        [...questionTerms].some((term) =>
          record.rawText.toLowerCase().includes(term),
        )
          ? 2
          : 0),
    }))
    .filter((entry) => entry.score > 0)
    .sort((first, second) => second.score - first.score)
    .slice(0, 3);

  if (recordAnswers.length > 0) {
    const answer = recordAnswers
      .map(({ record }) => {
        const matchedDescription = record.descriptionLines.find((line) =>
          [...questionTerms].some((term) => line.toLowerCase().includes(term)),
        );

        if (matchedDescription) {
          return matchedDescription;
        }

        if (record.title || record.organization) {
          return formatRoleEntry(record).replace(/^- /, "");
        }

        return truncate(record.descriptionLines.join(" ") || record.rawText, 280);
      })
      .join(" ");

    return {
      answer: limitAnswerLength(answer),
      sources: sourcesFromRecords(
        recordAnswers.map((entry) => entry.record),
        chunks,
      ),
      template: "specific-fact-records",
    };
  }

  // Broader content scan for prose documents (policies, manuals, papers)
  const proseHits = chunks
    .flatMap((chunk) => {
      const lines =
        chunk.record?.descriptionLines?.length
          ? chunk.record.descriptionLines
          : splitIntoSentences(chunk.content);

      return lines.map((line) => ({
        line,
        chunk,
        score: [...questionTerms].filter((term) => line.toLowerCase().includes(term))
          .length,
      }));
    })
    .filter((entry) => entry.score > 0 && entry.line.length > 30)
    .sort((first, second) => second.score - first.score)
    .slice(0, 3);

  if (proseHits.length > 0) {
    return {
      answer: limitAnswerLength(proseHits.map((entry) => entry.line).join(" ")),
      sources: sourcesFromChunks(proseHits.map((entry) => entry.chunk)),
      template: "specific-fact-prose",
    };
  }

  const seenSentences = new Set<string>();
  const sentences = chunks
    .flatMap((chunk) =>
      splitIntoSentences(chunk.content).map((sentence) => {
        const sentenceScore = scoreSentence(sentence, questionTerms);
        const containsQueryFragment = [...questionTerms].some((term) =>
          sentence.toLowerCase().includes(term),
        );

        return {
          sentence,
          chunk,
          sentenceScore: sentenceScore + (containsQueryFragment ? 1 : 0),
          score:
            sentenceScore +
            chunk.rerankScore +
            (containsQueryFragment ? 1 : 0) +
            (lowerQuestion.length > 0 &&
            sentence.toLowerCase().includes(lowerQuestion.slice(0, 12))
              ? 0.5
              : 0),
        };
      }),
    )
    .filter(({ sentence, sentenceScore }) => {
      const normalized = sentence.toLowerCase();

      if (sentenceScore <= 0 || seenSentences.has(normalized)) {
        return false;
      }

      if (NAVIGATION_PATTERNS.some((pattern) => pattern.test(sentence))) {
        return false;
      }

      seenSentences.add(normalized);
      return true;
    })
    .sort((first, second) => second.score - first.score)
    .slice(0, 3);

  if (sentences.length === 0) {
    return {
      answer: FALLBACK_ANSWER,
      sources: [],
      template: "specific-fact",
    };
  }

  return {
    answer: limitAnswerLength(
      sentences.map(({ sentence }) => sentence).join(" "),
    ),
    sources: sourcesFromChunks(sentences.map(({ chunk }) => chunk)),
    template: "specific-fact",
  };
}

function validateAnswer(
  answer: string,
  queryType: QueryType,
  chunks: RetrievedChunk[],
): { ok: boolean; notes: string[] } {
  const notes: string[] = [];

  if (!answer || answer === FALLBACK_ANSWER) {
    return { ok: true, notes: ["fallback"] };
  }

  if (NAVIGATION_PATTERNS.some((pattern) => pattern.test(answer))) {
    notes.push("Contains navigation/UI text.");
  }

  if (/https?:\/\//i.test(answer) && queryType === "specific-fact") {
    notes.push("Contains URL/footer text.");
  }

  if (queryType === "identity-overview") {
    const profileName = chunks.find((chunk) => chunk.record?.recordType === "profile")
      ?.record?.title;
    if (isNameOnlyOverview(answer, profileName)) {
      notes.push("Overview answer is only a name.");
    }
  }

  const titleMatches = answer.match(/^- .+$/gm) ?? [];
  const titles = titleMatches.map((line) => line.toLowerCase());
  if (titles.length >= 3 && new Set(titles).size === 1) {
    notes.push("Repeats the same title several times.");
  }

  if (
    queryType === "location-related" &&
    /\bhometown\b/i.test(answer) === false &&
    /\bfrom\b/i.test(answer) &&
    /\bdoes not explicitly state\b/i.test(answer) === false &&
    !findExplicitOriginEvidence(chunks)
  ) {
    // If we claimed hometown without evidence phrase, reject
    if (/\bis from\b|\bhometown is\b/i.test(answer)) {
      notes.push("Uses work location as hometown without explicit evidence.");
    }
  }

  if (/^[a-z]/.test(answer.trim()) && answer.length < 40) {
    notes.push("Incomplete sentence fragment.");
  }

  return {
    ok: notes.length === 0,
    notes,
  };
}

function recordsFromChunks(chunks: RetrievedChunk[]): StructuredRecord[] {
  return chunks
    .map((chunk) => chunk.record)
    .filter((record): record is StructuredRecord => Boolean(record));
}

function dedupeRecords(records: StructuredRecord[]): StructuredRecord[] {
  const seen = new Set<string>();
  const result: StructuredRecord[] = [];

  for (const record of records) {
    const key = [
      record.title?.toLowerCase(),
      record.organization?.toLowerCase(),
      record.dateText?.toLowerCase(),
    ].join("|");

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    result.push(record);
  }

  return result;
}

function sourcesFromRecords(
  records: StructuredRecord[],
  chunks: RetrievedChunk[],
): AnswerSource[] {
  const matched = records
    .map((record) =>
      chunks.find(
        (chunk) =>
          chunk.record?.id === record.id ||
          chunk.metadata.recordId === record.id,
      ),
    )
    .filter((chunk): chunk is RetrievedChunk => Boolean(chunk));

  return sourcesFromChunks(matched.length > 0 ? matched : chunks.slice(0, records.length));
}

function sourcesFromChunks(chunks: RetrievedChunk[]): AnswerSource[] {
  const seen = new Set<string>();
  const sources: AnswerSource[] = [];

  for (const chunk of chunks) {
    const key = `${chunk.documentId}:${chunk.chunkIndex}`;

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    sources.push({
      documentId: chunk.documentId,
      documentName: chunk.documentName,
      pageStart: chunk.pageStart,
      pageEnd: chunk.pageEnd,
      heading: chunk.heading,
      chunkIndex: chunk.chunkIndex,
      recordId: chunk.metadata.recordId,
      recordType: chunk.metadata.recordType,
    });
  }

  return sources;
}

function extractSkillsFromText(text: string): string[] {
  const match = text.match(/Skills?:\s*([^\n]+)/i);

  if (!match) {
    return [];
  }

  return match[1]
    .split(",")
    .map((part) => part.replace(/\+\d+\s*skills?/i, "").trim())
    .filter((part) => part.length > 1);
}

function scoreRecordAgainstQuestion(
  record: StructuredRecord,
  questionTerms: Set<string>,
): number {
  const terms = tokenize(
    [
      record.title,
      record.organization,
      record.heading,
      record.dateText,
      ...record.descriptionLines,
      ...(record.skills ?? []),
    ]
      .filter(Boolean)
      .join(" "),
  );

  let score = 0;

  for (const term of terms) {
    if (questionTerms.has(term)) {
      score += 1;
    }
  }

  return score;
}

function scoreSentence(sentence: string, questionTerms: Set<string>): number {
  const sentenceTerms = tokenize(sentence);
  let score = 0;

  for (const term of sentenceTerms) {
    if (questionTerms.has(term)) {
      score += 1;
    }
  }

  return score;
}

function getFrequentTerms(text: string): string[] {
  const counts = new Map<string, number>();

  for (const token of tokenize(text)) {
    if (token.length < 4 || STOP_WORDS.has(token)) {
      continue;
    }

    counts.set(token, (counts.get(token) ?? 0) + 1);
  }

  return [...counts.entries()]
    .sort((first, second) => second[1] - first[1])
    .map(([term]) => term);
}

function isNameOnlyOverview(answer: string, name?: string): boolean {
  const cleaned = answer.replace(/[^\p{L}\s]/gu, " ").replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return true;
  }

  if (name && cleaned.toLowerCase() === name.toLowerCase()) {
    return true;
  }

  return cleaned.split(" ").length <= 4 && !/[.!]/.test(answer);
}

function isUnsupportedTopic(question: string): boolean {
  return /\b(favorite food|favourite food|zodiac|blood type|password)\b/i.test(
    question,
  );
}

function unique(values: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];

  for (const value of values) {
    const key = value.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(value);
  }

  return result;
}

function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 1).trim()}…`;
}

function trimTrailingPunctuation(text: string): string {
  return text.replace(/[.\s]+$/g, "");
}

function limitAnswerLength(answer: string): string {
  const maxLength = 1200;

  if (answer.length <= maxLength) {
    return answer;
  }

  const trimmed = answer.slice(0, maxLength);
  const finalSentenceEnd = Math.max(
    trimmed.lastIndexOf("."),
    trimmed.lastIndexOf("!"),
    trimmed.lastIndexOf("?"),
    trimmed.lastIndexOf("\n"),
  );

  return finalSentenceEnd > 250
    ? trimmed.slice(0, finalSentenceEnd + 1)
    : `${trimmed.trim()}...`;
}
