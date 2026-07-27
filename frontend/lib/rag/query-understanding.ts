import { tokenize } from "./token-utils";

export type QueryType =
  | "greeting"
  | "identity-overview"
  | "employment-related"
  | "internship-related"
  | "education-related"
  | "skills-related"
  | "organization-related"
  | "location-related"
  | "chronological-request"
  | "list-request"
  | "broad-summary"
  | "specific-fact"
  | "unsupported-or-unclear";

export type QueryUnderstanding = {
  originalQuestion: string;
  normalizedQuestion: string;
  queryType: QueryType;
  expandedTerms: string[];
};

const FUZZY_DICTIONARY = [
  "school",
  "hometown",
  "involvements",
  "involvement",
  "education",
  "experience",
  "experiences",
  "jobs",
  "job",
  "role",
  "roles",
  "internship",
  "internships",
  "skills",
  "skill",
  "organization",
  "organizations",
  "policy",
  "cancellation",
  "refund",
  "location",
  "summary",
];

const EXACT_MISSPELLINGS: Record<string, string> = {
  fram: "from",
  schol: "school",
  hoemtown: "hometown",
  involvments: "involvements",
  internshp: "internship",
  internshps: "internships",
};

const EXPANSIONS: Record<string, string[]> = {
  school: ["education", "university", "college", "institution", "degree", "academic", "program"],
  job: ["employment", "experience", "position", "role", "occupation", "internship", "employer"],
  jobs: ["employment", "experience", "position", "role", "occupation", "internship", "employer"],
  role: ["employment", "experience", "position", "job", "responsibilities"],
  roles: ["employment", "experience", "position", "job", "responsibilities"],
  internship: ["intern", "employment", "experience", "role", "position"],
  internships: ["intern", "employment", "experience", "role", "position"],
  skills: ["skill", "abilities", "competencies", "proficiencies", "technologies"],
  skill: ["skills", "abilities", "competencies", "proficiencies"],
  organization: ["organizations", "involvement", "activities", "membership", "chapter"],
  organizations: ["organization", "involvement", "activities", "membership", "chapter"],
  involvements: ["activities", "organizations", "leadership", "memberships", "volunteering", "roles"],
  involvement: ["activities", "organizations", "leadership", "memberships", "volunteering", "roles"],
  recent: ["latest", "current", "newest", "present", "recent", "chronological"],
  latest: ["recent", "current", "newest", "present"],
  current: ["present", "active", "latest", "recent"],
  location: ["city", "region", "based", "address", "located", "from", "hometown"],
  hometown: ["from", "origin", "born", "based", "residence"],
  refund: ["refund", "return", "payment", "cancellation", "policy"],
  cancellation: ["refund", "return", "payment", "cancellation", "policy"],
  summary: ["overview", "topics", "sections", "document"],
  who: ["profile", "overview", "identity", "about"],
};

export function understandQuery(question: string): QueryUnderstanding {
  const normalizedQuestion = normalizeQuestion(question);
  const tokens = tokenize(normalizedQuestion);
  const expandedTerms = expandTerms(tokens);

  return {
    originalQuestion: question,
    normalizedQuestion,
    queryType: classifyQuery(normalizedQuestion, tokens),
    expandedTerms,
  };
}

export function normalizeQuestion(question: string): string {
  return question
    .normalize("NFKC")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((word) => normalizePossibleMisspelling(word))
    .join(" ");
}

function classifyQuery(question: string, tokens: string[]): QueryType {
  const lower = question.toLowerCase();

  if (/^(hi|hello|hey|good morning|good afternoon|good evening)\b/.test(lower)) {
    return "greeting";
  }

  if (
    /\b(favorite food|favourite food|zodiac|blood type|social security|password)\b/.test(
      lower,
    )
  ) {
    return "unsupported-or-unclear";
  }

  if (
    /\b(hometown|birthplace|origin)\b/.test(lower) ||
    /\bwhere\b.+\bfrom\b/.test(lower) ||
    /\bfrom where\b/.test(lower) ||
    /\bborn in\b/.test(lower)
  ) {
    return "location-related";
  }

  if (
    /\b(who is|who'?s|tell me about|main subject|this person|about (him|her|them|this person))\b/.test(
      lower,
    ) ||
    /\b(what does (this|the) person do|overview of (this|the) (person|subject))\b/.test(
      lower,
    )
  ) {
    return "identity-overview";
  }

  if (/\b(internship|internships|intern)\b/.test(lower)) {
    return "internship-related";
  }

  if (/\b(skill|skills|abilities|competencies)\b/.test(lower)) {
    return "skills-related";
  }

  if (
    /\b(organization|organizations|involvement|involvements|activities|memberships?)\b/.test(
      lower,
    )
  ) {
    return "organization-related";
  }

  if (/\b(summarize|summary|overview|main topics|what is this document)\b/.test(lower)) {
    return "broad-summary";
  }

  if (/\b(recent|latest|current|newest|previous|earliest|chronological)\b/.test(lower)) {
    return "chronological-request";
  }

  if (tokens.some((token) => ["school", "education", "university", "college", "degree"].includes(token))) {
    return "education-related";
  }

  if (
    tokens.some((token) =>
      ["job", "jobs", "role", "roles", "employment", "experience", "experiences", "position"].includes(
        token,
      ),
    )
  ) {
    return "employment-related";
  }

  if (tokens.some((token) => ["location", "city", "region", "from", "address", "located", "hometown"].includes(token))) {
    return "location-related";
  }

  if (/\b(list|what are|which|entries)\b/.test(lower)) {
    return "list-request";
  }

  return tokens.length <= 1 ? "unsupported-or-unclear" : "specific-fact";
}

function expandTerms(tokens: string[]): string[] {
  const terms = new Set(tokens);

  for (const token of tokens) {
    for (const expansion of EXPANSIONS[token] ?? []) {
      terms.add(expansion);
    }
  }

  return [...terms];
}

function normalizePossibleMisspelling(word: string): string {
  const cleanWord = word.toLowerCase().replace(/[^\p{L}\p{N}-]/gu, "");

  if (EXACT_MISSPELLINGS[cleanWord]) {
    return word.replace(cleanWord, EXACT_MISSPELLINGS[cleanWord]);
  }

  // "schol" is length 5 - still allow fuzzy to "school"
  if (cleanWord.length < 4 || /^[A-Z]/.test(word)) {
    return word;
  }

  // Avoid rewriting short valid words (e.g. "rule" -> "role").
  const protectedWords = new Set([
    "rule",
    "rules",
    "rate",
    "rates",
    "form",
    "from",
    "more",
    "some",
    "come",
    "home",
  ]);

  if (protectedWords.has(cleanWord)) {
    return word;
  }

  const match = FUZZY_DICTIONARY.find((candidate) => {
    const distance = levenshtein(cleanWord, candidate);
    const maxDistance = cleanWord.length >= 7 ? 2 : 1;
    return distance > 0 && distance <= maxDistance;
  });

  return match ? word.replace(cleanWord, match) : word;
}

function levenshtein(first: string, second: string): number {
  const dp = Array.from({ length: first.length + 1 }, () =>
    Array.from({ length: second.length + 1 }, () => 0),
  );

  for (let index = 0; index <= first.length; index += 1) {
    dp[index][0] = index;
  }

  for (let index = 0; index <= second.length; index += 1) {
    dp[0][index] = index;
  }

  for (let row = 1; row <= first.length; row += 1) {
    for (let column = 1; column <= second.length; column += 1) {
      dp[row][column] = Math.min(
        dp[row - 1][column] + 1,
        dp[row][column - 1] + 1,
        dp[row - 1][column - 1] +
          (first[row - 1] === second[column - 1] ? 0 : 1),
      );
    }
  }

  return dp[first.length][second.length];
}
