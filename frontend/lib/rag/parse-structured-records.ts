import { extractDateMetadata } from "./date-utils";
import type {
  ExtractedDocument,
  StructuredRecord,
  StructuredRecordType,
} from "./document-types";

type AnnotatedLine = {
  text: string;
  pageNumber?: number;
  kind:
    | "section"
    | "date"
    | "location"
    | "employment"
    | "duration"
    | "org"
    | "title"
    | "bullet"
    | "skills"
    | "prose"
    | "noise";
};

type DraftRecord = {
  recordType: StructuredRecordType;
  title?: string;
  organization?: string;
  dateText?: string;
  startDate?: string;
  endDate?: string;
  location?: string;
  heading?: string;
  employmentType?: string;
  descriptionLines: string[];
  skills: string[];
  pageStart?: number;
  pageEnd?: number;
  pending: boolean;
};

const SECTION_HEADINGS = new Set([
  "experience",
  "education",
  "skills",
  "about",
  "summary",
  "profile",
  "projects",
  "project",
  "publications",
  "licenses",
  "certifications",
  "volunteer",
  "volunteering",
  "organizations",
  "involvement",
  "activities",
  "leadership",
  "research",
  "awards",
  "honors",
  "interests",
  "overview",
  "services",
  "policy",
  "policies",
  "procedure",
  "procedures",
  "introduction",
  "abstract",
  "methods",
  "results",
  "discussion",
  "conclusion",
  "references",
  "troubleshooting",
  "requirements",
]);

const EMPLOYMENT_TYPES = [
  "internship",
  "intern",
  "part-time",
  "full-time",
  "contract",
  "freelance",
  "self-employed",
  "on-site",
  "onsite",
  "remote",
  "hybrid",
  "seasonal",
  "temporary",
];

export type StructuredParseResult = {
  records: StructuredRecord[];
  usedStructuredParsing: boolean;
  fallbackReason?: string;
};

export function parseStructuredRecords(
  document: ExtractedDocument,
): StructuredParseResult {
  const lines = annotateDocumentLines(document);
  const drafts = buildDraftsFromLines(lines);
  const merged = mergeCrossPageContinuations(drafts);
  const finalized = merged
    .map((draft) => finalizeDraft(draft, document.documentId))
    .filter((record): record is StructuredRecord => Boolean(record));

  const meaningful = finalized.filter(isMeaningfulRecord);

  if (meaningful.length === 0) {
    return {
      records: buildSectionFallbackRecords(document),
      usedStructuredParsing: false,
      fallbackReason:
        "Structured entry extraction found no complete records; using section-based chunks.",
    };
  }

  // Prefer structured entries whenever at least one complete record exists.
  // Section fallback is only for documents that never formed entry-like records.
  const looksLikeEntries = meaningful.some(
    (record) =>
      Boolean(record.title && (record.organization || record.dateText)) ||
      record.recordType === "profile" ||
      record.recordType === "skill" ||
      (record.descriptionLines.length > 0 &&
        ["experience", "role", "internship", "education", "policy"].includes(
          record.recordType,
        )),
  );

  if (!looksLikeEntries && meaningful.every((record) => record.recordType === "section")) {
    return {
      records: meaningful.length > 0 ? meaningful : buildSectionFallbackRecords(document),
      usedStructuredParsing: false,
      fallbackReason:
        "Structured entry extraction found only generic sections; using section-based chunks.",
    };
  }

  return {
    records: meaningful,
    usedStructuredParsing: true,
  };
}

export function buildRecordEmbeddingText(record: StructuredRecord): string {
  const lines: string[] = [];

  if (record.title) {
    lines.push(`Title: ${record.title}`);
  }

  if (record.organization) {
    lines.push(`Organization: ${record.organization}`);
  }

  if (record.dateText) {
    lines.push(`Dates: ${record.dateText}`);
  }

  if (record.location) {
    lines.push(`Location: ${record.location}`);
  }

  if (record.heading) {
    lines.push(`Section: ${record.heading}`);
  }

  if (record.employmentType) {
    lines.push(`Employment type: ${record.employmentType}`);
  }

  if (record.skills && record.skills.length > 0) {
    lines.push(`Skills: ${record.skills.join(", ")}`);
  }

  if (record.descriptionLines.length > 0) {
    lines.push("Description:");
    lines.push(...record.descriptionLines);
  }

  return lines.join("\n").trim() || record.rawText;
}

export function buildRecordDisplayText(record: StructuredRecord): string {
  return buildRecordEmbeddingText(record);
}

function annotateDocumentLines(document: ExtractedDocument): AnnotatedLine[] {
  const annotated: AnnotatedLine[] = [];

  for (const section of document.sections) {
    const pageLines = section.text.split("\n").map((line) => line.trim()).filter(Boolean);

    for (const text of pageLines) {
      annotated.push({
        text,
        pageNumber: section.pageNumber,
        kind: classifyLine(text),
      });
    }
  }

  return annotated;
}

function classifyLine(text: string): AnnotatedLine["kind"] {
  if (isStandaloneSectionHeading(text)) {
    return "section";
  }

  if (isDateLine(text)) {
    return "date";
  }

  if (isDurationOnly(text)) {
    return "duration";
  }

  if (isEmploymentTypeLine(text)) {
    return "employment";
  }

  if (isLocationLine(text)) {
    return "location";
  }

  if (isSkillsLine(text)) {
    return "skills";
  }

  if (isBulletLine(text)) {
    return "bullet";
  }

  if (isOrganizationLine(text)) {
    return "org";
  }

  if (isTitleLikeLine(text)) {
    return "title";
  }

  if (text.length > 90 || /[.!?]$/.test(text)) {
    return "prose";
  }

  return "prose";
}

function isStandaloneSectionHeading(text: string): boolean {
  const normalized = text.toLowerCase().trim();

  if (!SECTION_HEADINGS.has(normalized) || text.length > 40) {
    return false;
  }

  // Prefer clear heading forms: ALL CAPS, Title Case single token, or known labels.
  // Avoid wrapped fragments such as "Resources &\nEducation".
  if (text === text.toUpperCase() && /[A-Z]/.test(text)) {
    return true;
  }

  if (/^(Experience|Education|Skills|About|Summary|Profile|Projects|Licenses|Certifications|Volunteer|Publications|Awards|Honors|Interests|Overview|Services|Policy|Policies|Procedure|Procedures|Introduction|Abstract|Methods|Results|Discussion|Conclusion|References|Troubleshooting|Requirements)$/i.test(text)) {
    return true;
  }

  return false;
}

function buildDraftsFromLines(lines: AnnotatedLine[]): DraftRecord[] {
  const drafts: DraftRecord[] = [];
  let currentSection: string | undefined;
  let currentOrganization: string | undefined;
  let pendingTitles: string[] = [];
  const state: { active: DraftRecord | null } = { active: null };
  let awaitingOrgForTitle = false;
  let previousRawText = "";

  const flushActive = () => {
    if (state.active) {
      drafts.push(state.active);
      state.active = null;
    }
  };

  const startRecord = (partial: Partial<DraftRecord>, page?: number) => {
    flushActive();
    state.active = {
      recordType: partial.recordType ?? inferRecordType(currentSection, partial),
      title: partial.title,
      organization: partial.organization ?? currentOrganization,
      dateText: partial.dateText,
      startDate: partial.startDate,
      endDate: partial.endDate,
      location: partial.location,
      heading: partial.heading ?? currentSection,
      employmentType: partial.employmentType,
      descriptionLines: [...(partial.descriptionLines ?? [])],
      skills: [...(partial.skills ?? [])],
      pageStart: page,
      pageEnd: page,
      pending: true,
    };
  };

  const touchPage = (page?: number) => {
    if (!state.active || page === undefined) {
      return;
    }

    state.active.pageStart = state.active.pageStart ?? page;
    state.active.pageEnd = page;
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const next = lines[index + 1];

    // Skip wrapped heading false-positives: previous line ends with & / and
    if (
      line.kind === "section" &&
      /[&/]$| and$/i.test(previousRawText.trim())
    ) {
      previousRawText = line.text;
      continue;
    }

    switch (line.kind) {
      case "section": {
        flushActive();
        pendingTitles = [];
        currentOrganization = undefined;
        currentSection = line.text;
        awaitingOrgForTitle = false;
        break;
      }
      case "org": {
        const orgName = stripEmploymentSuffix(line.text);
        const employment = extractEmploymentFromOrgLine(line.text);
        currentOrganization = orgName;

        if (state.active && !state.active.organization && state.active.title) {
          state.active.organization = orgName;
          if (employment) {
            state.active.employmentType = employment;
            if (/intern/i.test(employment)) {
              state.active.recordType = "internship";
            }
          }
          touchPage(line.pageNumber);
        } else if (pendingTitles.length > 0) {
          const title = pendingTitles.pop();
          startRecord(
            {
              title,
              organization: orgName,
              employmentType: employment,
              heading: currentSection,
              recordType: inferRecordType(currentSection, {
                employmentType: employment,
                title,
              }),
            },
            line.pageNumber,
          );
        } else if (
          next &&
          (next.kind === "title" || next.kind === "date" || next.kind === "duration")
        ) {
          // Organization block opener; roles follow.
          awaitingOrgForTitle = false;
        } else if (!state.active) {
          startRecord(
            {
              organization: orgName,
              employmentType: employment,
              heading: currentSection,
              recordType: "organization",
            },
            line.pageNumber,
          );
        }
        break;
      }
      case "duration": {
        // Tenure summary under an organization; keep org context.
        touchPage(line.pageNumber);
        break;
      }
      case "title": {
        // Profile-like name + headline near the start of a profile section
        if (
          (!currentSection || /profile|about|summary/i.test(currentSection)) &&
          drafts.length === 0 &&
          !state.active &&
          looksLikePersonName(line.text) &&
          next?.kind === "prose"
        ) {
          startRecord(
            {
              title: line.text,
              heading: currentSection ?? "Profile",
              recordType: "profile",
              descriptionLines: [next.text],
            },
            line.pageNumber,
          );
          index += 1;
          flushActive();
          break;
        }

        if (
          next?.kind === "date" ||
          next?.kind === "employment" ||
          next?.kind === "org"
        ) {
          startRecord(
            {
              title: line.text,
              organization: currentOrganization,
              heading: currentSection,
              recordType: inferRecordType(currentSection, { title: line.text }),
            },
            line.pageNumber,
          );
          awaitingOrgForTitle = next.kind !== "org";
          break;
        }

        if (next?.kind === "title" && lines[index + 2]?.kind === "date") {
          pendingTitles.push(line.text);
          break;
        }

        if (state.active && !state.active.title && state.active.organization) {
          state.active.title = line.text;
          touchPage(line.pageNumber);
          break;
        }

        // A new title-like line after a completed entry should start a fresh record
        // even when the following line is prose (common in education blocks).
        if (
          state.active &&
          (state.active.dateText || state.active.descriptionLines.length > 0)
        ) {
          startRecord(
            {
              title: line.text,
              organization: currentOrganization,
              heading: currentSection,
              recordType: inferRecordType(currentSection, { title: line.text }),
            },
            line.pageNumber,
          );
          awaitingOrgForTitle = true;
          break;
        }

        pendingTitles.push(line.text);
        if (pendingTitles.length > 3) {
          pendingTitles = pendingTitles.slice(-2);
        }
        break;
      }
      case "date": {
        const dateMeta = extractDateMetadata(line.text);
        if (state.active && state.active.pending && !state.active.dateText) {
          state.active.dateText = dateMeta?.raw ?? line.text;
          state.active.startDate = formatDateLabel(dateMeta?.start);
          state.active.endDate = dateMeta?.isPresent
            ? "Present"
            : formatDateLabel(dateMeta?.end);
          touchPage(line.pageNumber);
        } else if (pendingTitles.length > 0) {
          const title = pendingTitles.pop();
          startRecord(
            {
              title,
              organization: currentOrganization,
              dateText: dateMeta?.raw ?? line.text,
              startDate: formatDateLabel(dateMeta?.start),
              endDate: dateMeta?.isPresent
                ? "Present"
                : formatDateLabel(dateMeta?.end),
              heading: currentSection,
              recordType: inferRecordType(currentSection, { title }),
            },
            line.pageNumber,
          );
        } else if (state.active) {
          // New dated role without an explicit title line — keep as metadata-only if needed
          flushActive();
        }
        awaitingOrgForTitle = false;
        break;
      }
      case "employment": {
        if (state.active) {
          state.active.employmentType = line.text;
          if (/intern/i.test(line.text)) {
            state.active.recordType = "internship";
          }
          touchPage(line.pageNumber);
        }
        break;
      }
      case "location": {
        if (state.active) {
          state.active.location = stripEmploymentSuffix(line.text);
          const employment = extractEmploymentFromOrgLine(line.text);
          if (employment) {
            state.active.employmentType =
              state.active.employmentType ?? employment;
          }
          touchPage(line.pageNumber);
        }
        break;
      }
      case "skills": {
        const skills = parseSkillsLine(line.text);
        if (state.active) {
          state.active.skills.push(...skills);
          touchPage(line.pageNumber);
        } else {
          startRecord(
            {
              title: "Skills",
              heading: currentSection ?? "Skills",
              recordType: "skill",
              skills,
              descriptionLines:
                skills.length > 0 ? [`Skills: ${skills.join(", ")}`] : [],
            },
            line.pageNumber,
          );
          flushActive();
        }
        break;
      }
      case "bullet":
      case "prose": {
        const content =
          line.kind === "bullet" ? normalizeBullet(line.text) : line.text;

        if (!content) {
          break;
        }

        if (
          line.kind === "prose" &&
          state.active &&
          state.active.pending &&
          state.active.descriptionLines.length === 0 &&
          content.length < 80 &&
          !state.active.organization &&
          awaitingOrgForTitle
        ) {
          state.active.organization = content;
          awaitingOrgForTitle = false;
          touchPage(line.pageNumber);
          break;
        }

        if (state.active) {
          state.active.descriptionLines.push(content);
          touchPage(line.pageNumber);
        } else if (currentSection) {
          startRecord(
            {
              heading: currentSection,
              recordType: inferRecordType(currentSection, {}),
              descriptionLines: [content],
            },
            line.pageNumber,
          );
        } else {
          startRecord(
            {
              recordType: "section",
              descriptionLines: [content],
            },
            line.pageNumber,
          );
        }
        break;
      }
      default:
        break;
    }

    previousRawText = line.text;
  }

  flushActive();
  return drafts;
}

function mergeCrossPageContinuations(drafts: DraftRecord[]): DraftRecord[] {
  if (drafts.length === 0) {
    return drafts;
  }

  const merged: DraftRecord[] = [];

  for (const draft of drafts) {
    const previous = merged.at(-1);

    if (previous && shouldMergeContinuation(previous, draft)) {
      previous.descriptionLines.push(...draft.descriptionLines);
      previous.skills.push(...draft.skills);
      previous.pageEnd = draft.pageEnd ?? previous.pageEnd;
      previous.title = previous.title ?? draft.title;
      previous.organization = previous.organization ?? draft.organization;
      previous.dateText = previous.dateText ?? draft.dateText;
      previous.location = previous.location ?? draft.location;
      previous.employmentType = previous.employmentType ?? draft.employmentType;
      continue;
    }

    merged.push({ ...draft, descriptionLines: [...draft.descriptionLines], skills: [...draft.skills] });
  }

  return merged;
}

function shouldMergeContinuation(
  previous: DraftRecord,
  next: DraftRecord,
): boolean {
  if (
    previous.heading &&
    next.heading &&
    previous.heading.toLowerCase() !== next.heading.toLowerCase()
  ) {
    return false;
  }

  if (next.title || next.organization || next.dateText) {
    return false;
  }

  const nextLooksLikeContinuation =
    next.descriptionLines.length > 0 &&
    (next.descriptionLines.every((line) => isBulletLike(line) || line.length > 40) ||
      Boolean(next.heading && next.heading === previous.heading));

  const previousOpen =
    Boolean(previous.title || previous.organization) &&
    (previous.descriptionLines.length === 0 ||
      previous.descriptionLines.some((line) => !/[.!?]$/.test(line)));

  const adjacentPages =
    previous.pageEnd !== undefined &&
    next.pageStart !== undefined &&
    next.pageStart - previous.pageEnd <= 1;

  return nextLooksLikeContinuation && previousOpen && adjacentPages;
}

function finalizeDraft(
  draft: DraftRecord,
  documentId: string,
): StructuredRecord | null {
  const rawParts = [
    draft.title,
    draft.organization,
    draft.employmentType,
    draft.dateText,
    draft.location,
    draft.heading ? `Section: ${draft.heading}` : undefined,
    ...draft.descriptionLines,
    draft.skills.length > 0 ? `Skills: ${draft.skills.join(", ")}` : undefined,
  ].filter(Boolean);

  if (rawParts.length === 0) {
    return null;
  }

  const recordType = refineRecordType(draft);

  return {
    id: `${documentId}:record:${hashId(rawParts.join("|"))}`,
    documentId,
    recordType,
    title: draft.title,
    organization: draft.organization,
    dateText: draft.dateText,
    startDate: draft.startDate,
    endDate: draft.endDate,
    location: draft.location,
    heading: draft.heading,
    employmentType: draft.employmentType,
    descriptionLines: draft.descriptionLines,
    skills: draft.skills.length > 0 ? draft.skills : undefined,
    pageStart: draft.pageStart,
    pageEnd: draft.pageEnd,
    rawText: rawParts.join("\n"),
  };
}

function refineRecordType(draft: DraftRecord): StructuredRecordType {
  if (draft.recordType === "profile") {
    return "profile";
  }

  if (
    draft.employmentType &&
    /intern/i.test(draft.employmentType)
  ) {
    return "internship";
  }

  if (/intern/i.test(draft.title ?? "")) {
    return "internship";
  }

  if (draft.recordType === "skill" || (draft.skills.length > 0 && !draft.title && !draft.organization)) {
    return "skill";
  }

  const heading = (draft.heading ?? "").toLowerCase();

  // Role-like entries under a mis-detected section should still count as experience.
  if (draft.title && (draft.organization || draft.dateText)) {
    if (/education|school|academic/.test(heading) && !draft.employmentType) {
      // Keep education only when the heading is education AND there is no job signal
      // other than a title/date pair that looks academic.
      if (/\b(student|fellow|scholar|bachelor|master|degree|bs|ba|ms|phd)\b/i.test(
        `${draft.title} ${draft.organization ?? ""}`,
      )) {
        return "education";
      }
    }

    if (/project/.test(heading)) {
      return "project";
    }

    if (/policy|procedure|requirement/.test(heading)) {
      return "policy";
    }

    if (/skill/.test(heading)) {
      return "skill";
    }

    return draft.recordType === "internship" ? "internship" : "experience";
  }

  if (/education|school|academic/.test(heading)) {
    return "education";
  }

  if (/project/.test(heading)) {
    return "project";
  }

  if (/policy|procedure|requirement/.test(heading)) {
    return "policy";
  }

  if (/skill/.test(heading)) {
    return "skill";
  }

  if (draft.organization && !draft.title) {
    return "organization";
  }

  return draft.recordType;
}

function isMeaningfulRecord(record: StructuredRecord): boolean {
  const wordCount = countWords(record.rawText);

  if (wordCount < 3) {
    return false;
  }

  // Reject sidebar/person suggestion fragments
  if (/\b\d+(st|nd|rd|th)\b/.test(record.rawText) && wordCount < 20) {
    return false;
  }

  if (/\b(also follow|promoted|welcome offer|subscribe for)\b/i.test(record.rawText)) {
    return false;
  }

  if (
    record.organization &&
    (/…|\.\.\./.test(record.organization) ||
      /\bis driving\b/i.test(record.organization) ||
      /\b@\b/.test(record.organization) && record.organization.length < 40 ||
      /\|/.test(record.organization))
  ) {
    // Likely sidebar / truncated suggestion text, not a real organization field
    if (!record.dateText || record.descriptionLines.length === 0) {
      return false;
    }
    record.organization = undefined;
  }

  if (
    record.title &&
    (/\b\d+(st|nd|rd|th)\b/.test(record.title) ||
      /\b(connect|message|promoted)\b/i.test(record.title))
  ) {
    return false;
  }

  // Reject isolated name/title/date/location/org
  const hasOnlyTitle =
    Boolean(record.title) &&
    !record.organization &&
    !record.dateText &&
    record.descriptionLines.length === 0 &&
    !record.skills?.length;

  if (hasOnlyTitle && record.recordType !== "profile") {
    return false;
  }

  const hasOnlyOrg =
    Boolean(record.organization) &&
    !record.title &&
    !record.dateText &&
    record.descriptionLines.length === 0;

  if (hasOnlyOrg) {
    return false;
  }

  const hasOnlyDate =
    Boolean(record.dateText) &&
    !record.title &&
    !record.organization &&
    record.descriptionLines.length === 0;

  if (hasOnlyDate) {
    return false;
  }

  const hasOnlyLocation =
    Boolean(record.location) &&
    !record.title &&
    !record.organization &&
    record.descriptionLines.length === 0;

  if (hasOnlyLocation) {
    return false;
  }

  if (record.recordType === "section" && wordCount < 12) {
    return false;
  }

  return true;
}

function buildSectionFallbackRecords(
  document: ExtractedDocument,
): StructuredRecord[] {
  return document.sections
    .filter((section) => countWords(section.text) >= 12)
    .map((section, index) => {
      const heading = section.heading ?? detectLeadingHeading(section.text);
      const body = heading
        ? section.text.replace(new RegExp(`^${escapeRegExp(heading)}\\s*`), "")
        : section.text;

      return {
        id: `${document.documentId}:section:${index}`,
        documentId: document.documentId,
        recordType: classifyFallbackType(heading, body),
        title: heading,
        heading,
        descriptionLines: body
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean),
        pageStart: section.pageNumber,
        pageEnd: section.pageNumber,
        rawText: section.text,
      } satisfies StructuredRecord;
    });
}

function classifyFallbackType(
  heading: string | undefined,
  body: string,
): StructuredRecordType {
  const text = `${heading ?? ""}\n${body}`.toLowerCase();

  if (/\b(policy|procedure|refund|requirement)\b/.test(text)) {
    return "policy";
  }

  if (/\b(education|university|college|degree)\b/.test(text)) {
    return "education";
  }

  if (/\b(experience|employment|role|intern)\b/.test(text)) {
    return "experience";
  }

  if (/\bskills?\b/.test(text)) {
    return "skill";
  }

  return "section";
}

function detectLeadingHeading(text: string): string | undefined {
  const first = text.split("\n")[0]?.trim();

  if (!first || first.length > 80) {
    return undefined;
  }

  if (SECTION_HEADINGS.has(first.toLowerCase()) || /^[A-Z][\w\s&/-]{2,60}$/.test(first)) {
    return first;
  }

  return undefined;
}

function inferRecordType(
  section: string | undefined,
  partial: Partial<DraftRecord>,
): StructuredRecordType {
  if (partial.employmentType && /intern/i.test(partial.employmentType)) {
    return "internship";
  }

  const heading = (section ?? "").toLowerCase();

  if (/education/.test(heading)) {
    return "education";
  }

  if (/skill/.test(heading)) {
    return "skill";
  }

  if (/project/.test(heading)) {
    return "project";
  }

  if (/policy|procedure/.test(heading)) {
    return "policy";
  }

  if (/experience|leadership|involvement|activities|volunteer/.test(heading)) {
    return "experience";
  }

  if (partial.title) {
    return "role";
  }

  return "unknown";
}

function isDateLine(text: string): boolean {
  return Boolean(
    text.match(
      /\b([A-Za-z]{3,9}\s+\d{4}|\d{4})\s*[–-]\s*(Present|Current|[A-Za-z]{3,9}\s+\d{4}|\d{4})\b/i,
    ),
  );
}

function isDurationOnly(text: string): boolean {
  return /^\d+\s*(yr|yrs|year|years)?\s*\d*\s*(mo|mos|month|months)?$/i.test(
    text.trim(),
  ) || /^\d+\s*(yr|yrs|year|years)$/i.test(text.trim());
}

function isEmploymentTypeLine(text: string): boolean {
  const normalized = text.toLowerCase().trim();
  return EMPLOYMENT_TYPES.includes(normalized);
}

function isLocationLine(text: string): boolean {
  if (
    isDateLine(text) ||
    isEmploymentTypeLine(text) ||
    isBulletLine(text) ||
    isSkillsLine(text)
  ) {
    return false;
  }

  const cleaned = stripEmploymentSuffix(text).replace(/\s*·\s*.*$/, "").trim();

  if (cleaned.length < 5 || cleaned.length > 80) {
    return false;
  }

  if (/^(and|or|with|for|to|of|in|on|at|the|a|an)\b/i.test(cleaned)) {
    return false;
  }

  // Prefer City, Region[, Country] shapes with short segments.
  const parts = cleaned.split(",").map((part) => part.trim()).filter(Boolean);
  if (parts.length < 2 || parts.length > 3) {
    return false;
  }

  if (parts.some((part) => part.split(/\s+/).length > 4 || part.split(/\s+/).length < 1)) {
    return false;
  }

  if (parts.some((part) => !/^[\p{L}\s.'-]+$/u.test(part))) {
    return false;
  }

  return (
    /\b(united states|usa|u\.s\.a\.|canada|mexico|japan|remote|on-site|hybrid|california|texas|new york|florida|united kingdom|england|germany|france|china|korea|india)\b/i.test(
      cleaned,
    ) ||
    (parts.length === 3 &&
      parts.every((part) => /^[\p{Lu}]/u.test(part) && part.split(/\s+/).length <= 3))
  );
}

function isSkillsLine(text: string): boolean {
  return /^skills?\s*:/i.test(text.trim());
}

function isBulletLine(text: string): boolean {
  return /^[-*•]\s+\S+/.test(text) || /^\d+[.)]\s+\S+/.test(text);
}

function isBulletLike(text: string): boolean {
  return isBulletLine(text) || /^[-*•]/.test(text);
}

function isOrganizationLine(text: string): boolean {
  if (isDateLine(text) || isBulletLine(text) || isLocationLine(text)) {
    return false;
  }

  if (/\s[·•∙|]\s/.test(text) && EMPLOYMENT_TYPES.some((type) => new RegExp(type, "i").test(text))) {
    return true;
  }

  if (
    text.length <= 90 &&
    !/[.!?]$/.test(text) &&
    /\b(university|college|lab|laboratory|inc\.?|llc|corp|center|centre|institute|school|fraternity|sorority|council|technologies|tech)\b/i.test(
      text,
    )
  ) {
    return true;
  }

  return false;
}

function isTitleLikeLine(text: string): boolean {
  if (
    isDateLine(text) ||
    isBulletLine(text) ||
    isLocationLine(text) ||
    isEmploymentTypeLine(text) ||
    isDurationOnly(text) ||
    isSkillsLine(text) ||
    isOrganizationLine(text)
  ) {
    return false;
  }

  if (text.length < 3 || text.length > 90) {
    return false;
  }

  if (/[.!?]$/.test(text)) {
    return false;
  }

  const words = text.split(/\s+/);
  if (words.length > 12) {
    return false;
  }

  // Prefer title case / short role-like phrases
  const capitalized = words.filter((word) => /^[\p{Lu}]/u.test(word)).length;
  return capitalized / words.length >= 0.5;
}

function looksLikePersonName(text: string): boolean {
  const words = text.trim().split(/\s+/);
  return (
    words.length >= 2 &&
    words.length <= 5 &&
    words.every((word) => /^[\p{L}'.-]+$/u.test(word)) &&
    words.every((word) => /^[\p{Lu}]/u.test(word))
  );
}

function stripEmploymentSuffix(text: string): string {
  return text
    .replace(/\s*[·•∙|]\s*(Internship|Intern|Part-time|Full-time|Contract|Remote|On-site|Hybrid).*$/i, "")
    .replace(/\s*[·•∙|]\s*On-site$/i, "")
    .trim();
}

function extractEmploymentFromOrgLine(text: string): string | undefined {
  const match = text.match(
    /[·•∙|]\s*(Internship|Intern|Part-time|Full-time|Contract|Remote|On-site|Hybrid)\b/i,
  );
  return match?.[1];
}

function parseSkillsLine(text: string): string[] {
  const body = text.replace(/^skills?\s*:\s*/i, "");
  return body
    .split(",")
    .map((part) => part.replace(/\+\d+\s*skills?/i, "").trim())
    .filter((part) => part.length > 1 && !/^\+\d+/.test(part));
}

function normalizeBullet(text: string): string {
  return text.replace(/^[-*•]\s+/, "").replace(/^\d+[.)]\s+/, "").trim();
}

function formatDateLabel(value?: number): string | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (value === Number.MAX_SAFE_INTEGER) {
    return "Present";
  }

  const year = Math.floor(value / 100);
  const month = value % 100;
  const months = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];

  if (month >= 1 && month <= 12) {
    return `${months[month]} ${year}`;
  }

  return String(year);
}

function countWords(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

function hashId(value: string): string {
  let hash = 2166136261;

  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return Math.abs(hash).toString(36);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
