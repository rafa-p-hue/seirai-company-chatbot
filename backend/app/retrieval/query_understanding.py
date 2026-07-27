"""Document-agnostic query classification and generic intent expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

from app.models.api import ChatMessage


GREETING_RE = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening)[!.,\s]*$",
    re.I,
)

HELP_RE = re.compile(
    r"^(?:"
    r"can you help(?:\s+me)?|"
    r"help(?:\s+me)?|"
    r"what can you do|"
    r"how can you help(?:\s+me)?|"
    r"what do you do|"
    r"how does this work|"
    r"what kinds? of questions can i ask|"
    r"what questions can i ask|"
    r"what can i ask"
    r")[!?.\s]*$",
    re.I,
)

SCHOOL_RE = re.compile(
    r"\b(school|college|university|institution|campus|attend(?:s|ed|ing)?)\b",
    re.I,
)
MAJOR_RE = re.compile(
    r"\b(major|degree|field of study|academic program|stud(?:y|ies|ying))\b",
    re.I,
)
GPA_RE = re.compile(
    r"\b(gpa|grade[- ]?point\s+average|grade point average)\b",
    re.I,
)
BIRTHDAY_RE = re.compile(
    r"\b(birthday|date of birth|birth\s*date|dob|born)\b",
    re.I,
)
FOOD_RE = re.compile(
    r"\b(favorite\s+food|favourite\s+food|food\s+preference|preferred\s+food)\b",
    re.I,
)
LEADERSHIP_RE = re.compile(
    r"\b(leadership|leader(?:ship)?\s+roles?|president|vice[- ]?president|"
    r"chair(?:person)?|founder|director|coordinator|educator)\b",
    re.I,
)

ROLE_EXPERIENCE_RE = re.compile(
    r"\b("
    r"role|roles|experience|experiences|work|worked|job|jobs|position|positions|"
    r"responsibility|responsibilities|involvement|involvements|activity|activities|"
    r"internship|internships|intern|research|leadership|volunteer|affiliation|"
    r"fraternity|sorority|employer|employment|accomplishments|where .+ work|founded|found"
    r")\b",
    re.I,
)

EXPERIENCE_QUERY_TYPES = {
    "experience",
    "internship",
    "research",
    "leadership",
}

SUMMARY_QUERY_TYPES = {"identity", "summary"}

MUSIC_RE = re.compile(
    r"\b(music|musician|musicians|ensemble|ensembles|band|bands|performance|"
    r"performances|choir|orchestra|play music)\b",
    re.I,
)
INSTRUMENT_RE = re.compile(
    r"\b(instrument|what .+ play|plays? the|guitar|piano|violin|drums|flute)\b",
    re.I,
)
SUMMARY_RE = re.compile(
    r"\b("
    r"what can you tell me about|tell me about|about this document|"
    r"about the (main )?subject|overview|summary|who is|who'?s"
    r")\b",
    re.I,
)
STUDY_RE = re.compile(
    r"\b(stud(?:y|ies|ying)|major|degree|field of study|academic program|education)\b",
    re.I,
)
WORK_RE = re.compile(
    r"\b(where (?:does|do|did) .+ work|where .+ works?|employer|employment|"
    r"place of work|work(?:s|ed)? (?:at|for))\b",
    re.I,
)


@dataclass
class QueryUnderstanding:
    original_question: str
    normalized_question: str
    resolved_question: str
    expanded_question: str
    query_type: str
    expanded_terms: List[str] = field(default_factory=list)
    subject_name: Optional[str] = None


# Soft ranking preferences only — never hard filters. Always include universal.
PREFERRED_TYPES = {
    "identity": ["universal", "profile", "education", "experience"],
    "summary": ["universal", "profile", "education", "experience", "research", "leadership"],
    "education": ["universal", "education", "profile"],
    "school": ["universal", "education", "profile"],
    "major": ["universal", "education", "profile"],
    "gpa": ["universal", "education", "profile"],
    "birthday": ["universal", "profile"],
    "favorite_food": ["universal", "profile", "skills"],
    "experience": ["universal", "experience", "internship", "research", "leadership"],
    "internship": ["universal", "internship", "experience"],
    "research": ["universal", "research", "experience"],
    "leadership": ["universal", "leadership", "experience"],
    "interest": ["universal", "skills", "leadership", "experience"],
    "skills": ["universal", "skills", "experience"],
    "policy": ["universal", "section"],
    "product": ["universal", "section"],
    "location": ["universal", "profile"],
    "awards": ["universal", "profile", "experience", "leadership"],
    "executive": ["universal", "experience", "profile"],
    "instrument": ["universal", "skills", "leadership", "experience"],
    "price": ["universal", "section"],
    "quantity": ["universal", "section"],
    "date": ["universal", "section", "education"],
    "accessibility": ["universal", "section"],
    "general": ["universal"],
}


def understand_query(
    question: str,
    *,
    history: Sequence[ChatMessage] | None = None,
    subject_name: Optional[str] = None,
    document_entities: Sequence[str] | None = None,
    document_name: Optional[str] = None,
    document_headings: Sequence[str] | None = None,
) -> QueryUnderstanding:
    normalized = _normalize(question)
    history_text = " ".join(
        msg.content for msg in (history or []) if msg.role in {"user", "assistant"}
    )
    subject = subject_name
    if not subject:
        from app.ingestion.entity_detect import resolve_subject_name

        subject = resolve_subject_name(
            question=normalized,
            history_text=history_text,
            entities=document_entities or [],
            document_name=document_name or "",
        )
    subject = subject or "the main subject described in the document"

    resolved = resolve_question_with_context(
        normalized, subject_name=subject, history=history
    )
    query_type = classify_query(
        resolved,
        subject_name=subject if subject_name or document_entities else None,
    )
    expanded_terms = expand_terms(resolved, query_type=query_type)
    if document_headings:
        expanded_terms = sorted(
            set(expanded_terms)
            | {
                token
                for heading in document_headings[:8]
                for token in re.findall(r"[a-z0-9]+", heading.lower())
                if len(token) > 3
            }
        )
    expanded_question = expand_short_question(
        resolved,
        query_type=query_type,
        subject_name=subject,
        history=history,
    )
    return QueryUnderstanding(
        original_question=question,
        normalized_question=normalized,
        resolved_question=resolved,
        expanded_question=expanded_question,
        query_type=query_type,
        expanded_terms=expanded_terms,
        subject_name=None
        if subject == "the main subject described in the document"
        else subject,
    )


def classify_query(question: str, *, subject_name: Optional[str] = None) -> str:
    lower = question.lower().strip().rstrip("?")
    if GREETING_RE.match(lower) or lower in {"hi", "hello", "hey"}:
        return "greeting"
    if HELP_RE.match(lower) or lower in {
        "help",
        "can you help",
        "what can you do",
        "what kinds of questions can i ask",
        "what questions can i ask",
        "what can i ask",
    }:
        return "help"
    if re.search(r"\b(password|social security|zodiac)\b", lower):
        return "unsupported"

    # Policy / handbook fact questions (before generic unsupported-style cues).
    if re.search(
        r"\b(refund|cancellation|cancel|membership|pet|pets|volunteer|"
        r"compost|sustainab|rental|rentals|alcohol|"
        r"accessib(?:le|ility)|disabilit(?:y|ies)|wheelchair|ada|"
        r"donation|donated|opening (?:date|period)|future plans?)\b",
        lower,
    ) or re.search(r"\b(policy|policies|rules?|prohibited|allowed)\b", lower):
        if re.search(
            r"\b(accessib(?:le|ility)|disabilit(?:y|ies)|wheelchair|ada)\b", lower
        ):
            return "accessibility"
        if re.search(r"\b(price|cost|fee|fees|how much|dues|\$)\b", lower):
            return "price"
        if re.search(r"\b(how many|quantity|pounds|tons|amount)\b", lower):
            return "quantity"
        if re.search(
            r"\b(when|opening|timeline|what year|what period|date)\b", lower
        ) and re.search(r"\b(open|opening|plan|future|timeline)\b", lower):
            return "date"
        return "policy"

    if GPA_RE.search(lower):
        return "gpa"
    if BIRTHDAY_RE.search(lower):
        return "birthday"
    if FOOD_RE.search(lower):
        return "favorite_food"

    # Instrument questions are distinct from general music/ensemble membership.
    if INSTRUMENT_RE.search(lower) and re.search(
        r"\b(instrument|play|plays|playing|played)\b", lower
    ):
        return "instrument"

    # Awards / accolades require explicit evidence — never invent.
    if re.search(
        r"\b(awards?|honors?|honours?|recognitions?|scholarships?|"
        r"grammy|oscar|emmy|nobel|pulitzer)\b",
        lower,
    ) and re.search(r"\b(award|honor|receive|win|won|get|got|any)\b", lower):
        return "awards"

    # Executive / CEO claims.
    if re.search(r"\b(ceo|chief executive|cfo|cto|coo)\b", lower) or re.search(
        r"\bwhat company\b", lower
    ):
        return "executive"

    # Geographic location (not institution name).
    # Accessibility questions win even when they mention "location".
    if re.search(
        r"\b(accessib(?:le|ility)|disabilit(?:y|ies)|wheelchair|ada)\b", lower
    ):
        return "accessibility"
    if re.search(
        r"\b(what state|which state|hometown|where .+ located|school located|"
        r"based in|live[s]?|city|address|where is)\b",
        lower,
    ) and not re.search(r"\b(attend|university|college)\b", lower):
        return "location"
    if re.search(r"\bwhere is .+ school located\b", lower) or re.search(
        r"\bschool .+ located\b", lower
    ):
        return "location"
    if re.search(r"\b(final |street |site )?address\b", lower):
        return "location"

    # Bare topical tokens and equivalent phrasings should share an intent.
    if lower in {"music"} or (MUSIC_RE.search(lower) and not INSTRUMENT_RE.search(lower)):
        return "interest"

    # Leadership titles even without the word "role/experience".
    if LEADERSHIP_RE.search(lower) and re.search(
        r"\b(is|was|are|were|president|vice|chair|founder|director)\b", lower
    ):
        if re.search(r"\b(founded|founder|establish(?:ed)?)\b", lower):
            return "leadership"
        if re.search(r"\b(president|vice[- ]?president|chair|director|leadership)\b", lower):
            return "leadership"

    # School (institution) vs major (academic program) — never conflate.
    school_hit = bool(SCHOOL_RE.search(lower))
    major_hit = bool(MAJOR_RE.search(lower))
    # "where is school located" already handled as location above.
    if school_hit and not major_hit:
        return "school"
    if major_hit and not school_hit:
        if re.search(r"\bminor\b", lower):
            return "education"
        return "major"
    if school_hit and major_hit:
        # "school of X" as major phrasing is rare; prefer school when attend/institution.
        if re.search(r"\b(attend|institution|university|college)\b", lower):
            return "school"
        return "major"
    if lower in {"major", "study", "studies", "his major", "her major", "their major"}:
        return "major"
    if lower in {"school", "university", "college"}:
        return "school"
    if re.search(r"\b\w+'s\s+major\b", lower):
        return "major"
    if re.search(r"\bminor\b", lower):
        return "education"

    # Broad document/person summaries.
    if lower in {
        "what can you tell me about this document",
        "tell me about the main subject",
        "tell me about this document",
        "about this document",
        "who is the main person",
        "who is the person",
    } or (
        SUMMARY_RE.search(lower)
        and not ROLE_EXPERIENCE_RE.search(lower)
        and not STUDY_RE.search(lower)
        and not MUSIC_RE.search(lower)
        and not HELP_RE.match(lower)
    ):
        if re.search(r"\b(who is|who'?s)\b", lower) or lower.startswith("tell me about"):
            return "identity"
        return "summary"

    # Work / employment questions (multiple roles).
    if WORK_RE.search(lower) or re.search(r"\bwhere (?:does|do|did).+\bwork\b", lower):
        return "experience"

    # Founded / founder / establish organization questions.
    if re.search(
        r"\b(founded|founder|what group .+ found|what organization .+ (?:found|establish)|"
        r"establish(?:ed)?)\b",
        lower,
    ):
        return "leadership"

    # Role/experience intent must win over bare-name identity cues.
    if ROLE_EXPERIENCE_RE.search(lower):
        if re.search(r"\b(internship|internships|intern)\b", lower):
            return "internship"
        if re.search(r"\b(research)\b", lower) and not STUDY_RE.search(lower):
            return "research"
        if LEADERSHIP_RE.search(lower) or re.search(
            r"\b(leadership|volunteer|involvement|involvements)\b",
            lower,
        ):
            return "leadership"
        # Generic activities stay experience unless leadership titles are asked.
        if re.search(r"\b(activity|activities|accomplishments|affiliation)\b", lower):
            return "experience"
        return "experience"

    if subject_name:
        first = subject_name.split()[0].lower()
        if lower == first or lower == subject_name.lower():
            return "identity"
        if re.search(rf"\bwho is\s+{re.escape(first)}\b", lower) or re.search(
            rf"\bwho is\s+{re.escape(subject_name.lower())}\b", lower
        ):
            return "identity"

    if re.search(
        r"\b(who is|who'?s|tell me about|about (him|her|them)|profile|overview|summary)\b",
        lower,
    ):
        return "identity"
    if re.search(
        r"\b(policy|policies|terms|conditions|privacy|compliance|leave policy)\b",
        lower,
    ):
        return "policy"
    if re.search(
        r"\b(manual|product|feature|specification|howto|how to|install|calibrat\w*|device)\b",
        lower,
    ):
        return "product"
    if STUDY_RE.search(lower):
        return "major"
    if SCHOOL_RE.search(lower):
        return "school"
    if re.search(r"\b(skill|skills|abilities)\b", lower):
        return "skills"
    if re.search(r"\b(hometown|where .+ from|location|based in|live[s]?)\b", lower):
        return "location"
    if lower in {"job", "jobs", "work", "experience", "experiences", "role", "roles"}:
        return "experience"
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", question.strip()) if len(token) > 1]
    if len(tokens) == 1 and tokens[0][0].isupper():
        return "identity"
    return "general"


def resolve_question_with_context(
    question: str,
    *,
    subject_name: str,
    history: Sequence[ChatMessage] | None = None,
) -> str:
    """Resolve pronouns using recent conversation context and detected subject."""
    text = question
    if subject_name and subject_name != "the main subject described in the document":
        if re.search(r"\b(he|his|him|she|her|they|their|them)\b", text, re.I):
            text = re.sub(r"\bhis\b", f"{subject_name}'s", text, flags=re.I)
            text = re.sub(r"\bher\b", f"{subject_name}'s", text, flags=re.I)
            text = re.sub(r"\btheir\b", f"{subject_name}'s", text, flags=re.I)
            text = re.sub(r"\bhe\b", subject_name, text, flags=re.I)
            text = re.sub(r"\bshe\b", subject_name, text, flags=re.I)
            text = re.sub(r"\bhim\b", subject_name, text, flags=re.I)
            text = re.sub(r"\bthem\b", subject_name, text, flags=re.I)
            text = re.sub(r"\bthey\b", subject_name, text, flags=re.I)
            return text

    # If history named a person and the question is short/role-oriented, bind subject.
    if history and ROLE_EXPERIENCE_RE.search(question):
        for message in reversed(list(history)):
            match = re.search(
                r"who is\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
                message.content,
                re.I,
            )
            if match:
                name = match.group(1).strip()
                return resolve_question_with_context(
                    question, subject_name=name, history=None
                )
    return text


def expand_short_question(
    question: str,
    *,
    query_type: str,
    subject_name: str,
    history: Sequence[ChatMessage] | None = None,
) -> str:
    """Generic intent expansion — never document-specific facts."""
    lower = question.lower().strip().rstrip("?")
    key = re.sub(r"[^a-z0-9\s]", "", lower)
    key = re.sub(r"\s+", " ", key).strip()

    subject_l = subject_name.lower()
    first = subject_l.split()[0] if subject_name else ""
    if key == first or key == subject_l:
        return f"Provide a brief summary of {subject_name} according to the document."

    expansions = {
        "school": f"What institutions, schools, or universities are associated with {subject_name}?",
        "what school": f"What institutions, schools, or universities are associated with {subject_name}?",
        "what school does the person attend": (
            f"What school, college, university, or institution does {subject_name} attend?"
        ),
        "major": f"What degrees or fields of study are listed for {subject_name}?",
        "what major": f"What degrees or fields of study are listed for {subject_name}?",
        "what is the persons major": f"What degrees or fields of study are listed for {subject_name}?",
        "what does the person study": f"What degrees or fields of study are listed for {subject_name}?",
        "what does he study": f"What degrees or fields of study are listed for {subject_name}?",
        "study": f"What degrees or fields of study are listed for {subject_name}?",
        "minor": f"What secondary fields of study or minors are listed for {subject_name}?",
        "gpa": f"What GPA or grade-point average is listed for {subject_name}?",
        "birthday": f"What birthday or date of birth is listed for {subject_name}?",
        "favorite food": f"What favorite food or food preference is listed for {subject_name}?",
        "role": f"What positions, responsibilities, or affiliations does {subject_name} have?",
        "roles": f"What positions, responsibilities, or affiliations does {subject_name} have?",
        "experience": f"What roles, positions, responsibilities, and experience are listed for {subject_name}?",
        "experiences": f"What roles, positions, responsibilities, and experience are listed for {subject_name}?",
        "music": f"What music, musical performance, ensemble, or band activities are described for {subject_name}?",
        "does the person play music": f"What music, musical performance, ensemble, or band activities are described for {subject_name}?",
        "is the person involved in music": f"What music, musical performance, ensemble, or band activities are described for {subject_name}?",
        "where does the person work": f"What employers, workplaces, roles, or positions are listed for {subject_name}?",
        "where does he work": f"What employers, workplaces, roles, or positions are listed for {subject_name}?",
        "what instrument does the person play": (
            f"What musical instrument does the document say {subject_name} plays?"
        ),
        "what leadership roles are listed": (
            f"What leadership titles such as president, chair, founder, director, "
            f"coordinator, or educator are listed for {subject_name}?"
        ),
        "what group did the person found": (
            f"What group, club, or organization did {subject_name} found?"
        ),
        "who is the main person": f"Who is the main person described in the document and what identifying fields are listed?",
        "who is the person": f"Who is the main person described in the document and what identifying fields are listed?",
        "tell me about the main subject": f"Provide a brief overview of {subject_name} covering identity, education, roles, research, and affiliations.",
        "what can you tell me about this document": (
            "Provide a brief overview of the main subject in this document, "
            "covering identity, education, responsibilities, research, and affiliations."
        ),
        "tell me about this document": (
            "Provide a brief overview of the main subject in this document, "
            "covering identity, education, responsibilities, research, and affiliations."
        ),
    }
    for pattern, template in list(expansions.items()):
        if key == pattern or key.endswith(pattern):
            return template
    if re.search(r"\bexperiences?\b", key) and len(key.split()) <= 8:
        return expansions["experiences"]
    if re.search(r"\broles?\b", key) and len(key.split()) <= 8:
        return expansions["roles"]
    if re.search(r"\bmusic|ensemble|band|performance\b", key):
        return expansions["music"]
    if re.search(r"\battend|school|university|college|institution\b", key) and not re.search(
        r"\bmajor|degree|field of study|stud\b", key
    ):
        return expansions["what school does the person attend"]
    if re.search(r"\bstud(?:y|ies)|major|field of study\b", key) and len(key.split()) <= 10:
        return expansions["major"]
    if re.search(r"\bwhere .+ work|employer|workplace\b", key):
        return expansions["where does the person work"]
    if re.search(r"\binstrument\b", key):
        return expansions["what instrument does the person play"]
    if re.search(r"\bgpa|grade point\b", key):
        return expansions["gpa"]
    if re.search(r"\bbirthday|date of birth|dob\b", key):
        return expansions["birthday"]
    if re.search(r"\bfood\b", key):
        return expansions["favorite food"]
    if re.search(r"\bleadership\b", key):
        return expansions["what leadership roles are listed"]
    if re.search(r"\bfound(?:ed|er)?\b", key):
        return expansions["what group did the person found"]

    if query_type in {"identity", "summary"}:
        return (
            f"Provide a brief overview of {subject_name} covering identity, education, "
            f"responsibilities and activities, research or projects, and affiliations."
        )
    if query_type == "interest":
        return expansions["music"]
    if query_type == "school":
        return expansions["what school does the person attend"]
    if query_type == "major":
        return expansions["major"]
    if query_type == "gpa":
        return expansions["gpa"]
    if query_type == "birthday":
        return expansions["birthday"]
    if query_type == "favorite_food":
        return expansions["favorite food"]
    if query_type == "education":
        if "minor" in key:
            return f"What secondary fields of study or minors are listed for {subject_name}?"
        return expansions["major"]
    if query_type == "research":
        return f"What research experience, positions, or projects are described for {subject_name}?"
    if query_type == "leadership":
        return expansions["what leadership roles are listed"]
    if query_type == "experience":
        return (
            f"What roles, positions, responsibilities, employment, internships, "
            f"research, and activities are listed for {subject_name}?"
        )
    if query_type == "internship":
        return f"What internship or temporary roles are listed for {subject_name}?"
    if query_type == "policy":
        return f"What policy details in the document answer: {question}"
    if query_type == "product":
        return f"What product or manual details in the document answer: {question}"
    return question


def expand_terms(question: str, *, query_type: str) -> List[str]:
    tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    mapping = {
        "school": ["university", "college", "campus", "institution", "education", "student"],
        "major": ["major", "field", "study", "degree", "education"],
        "minor": ["minor", "secondary", "field", "study"],
        "research": ["research", "assistant", "lab", "project", "study", "fellow"],
        "activities": ["activities", "involvements", "accomplishments", "leadership", "role"],
        "activity": ["activities", "involvements", "accomplishments", "role"],
        "who": ["name", "profile", "summary", "overview", "about"],
        "role": ["roles", "position", "positions", "title", "responsibility", "responsibilities", "affiliation", "experience"],
        "roles": ["role", "position", "positions", "responsibility", "responsibilities", "experience", "employment"],
        "experience": [
            "experiences",
            "roles",
            "positions",
            "responsibilities",
            "employment",
            "internships",
            "leadership",
            "research",
            "activities",
            "involvement",
            "job",
            "work",
        ],
        "experiences": [
            "experience",
            "roles",
            "positions",
            "responsibilities",
            "employment",
            "internships",
            "leadership",
            "research",
            "activities",
            "involvement",
        ],
        "position": ["positions", "role", "roles", "responsibility", "title", "experience"],
        "positions": ["position", "role", "roles", "responsibility", "experience"],
        "music": ["music", "musician", "performance", "ensemble", "band", "choir", "orchestra"],
        "musician": ["music", "performance", "ensemble", "band"],
        "play": ["music", "performance", "ensemble", "band"],
        "study": ["major", "degree", "field", "education", "academic", "program", "studies"],
        "studies": ["major", "degree", "field", "education", "academic", "program", "study"],
        "policy": ["policy", "procedure", "rule", "guideline", "terms"],
        "product": ["product", "feature", "manual", "specification", "guide"],
    }
    out: Set[str] = set(tokens)
    for token in list(tokens):
        out.update(mapping.get(token, []))
    if query_type in EXPERIENCE_QUERY_TYPES:
        out.update(
            {
                "roles",
                "positions",
                "responsibilities",
                "employment",
                "internships",
                "leadership",
                "research",
                "activities",
                "involvement",
                "experience",
                "job",
                "work",
            }
        )
    if query_type == "education":
        out.update(
            {
                "major",
                "minor",
                "university",
                "college",
                "school",
                "degree",
                "institution",
                "field",
                "study",
                "education",
                "academic",
                "program",
            }
        )
    if query_type == "school":
        out.update(
            {
                "school",
                "university",
                "college",
                "campus",
                "institution",
                "attend",
                "student",
                "academy",
            }
        )
    if query_type == "major":
        out.update(
            {
                "major",
                "degree",
                "field",
                "study",
                "education",
                "academic",
                "program",
            }
        )
    if query_type == "gpa":
        out.update({"gpa", "grade", "point", "average", "cumulative"})
    if query_type == "birthday":
        out.update({"birthday", "birth", "date", "dob", "born"})
    if query_type == "favorite_food":
        out.update({"food", "favorite", "favourite", "preference", "meal"})
    if query_type == "leadership":
        out.update(
            {
                "leadership",
                "president",
                "chair",
                "founder",
                "director",
                "coordinator",
                "educator",
                "officer",
                "vice",
            }
        )
    if query_type in SUMMARY_QUERY_TYPES:
        out.update(
            {
                "name",
                "profile",
                "summary",
                "overview",
                "education",
                "major",
                "research",
                "roles",
                "activities",
                "affiliation",
                "experience",
            }
        )
    if query_type == "interest":
        out.update({"music", "musician", "performance", "ensemble", "band", "choir", "orchestra"})
    if query_type == "identity":
        out.update({"name", "profile", "summary", "overview", "email", "title", "education", "major"})
    if query_type == "policy":
        out.update({"policy", "procedure", "guideline"})
    if query_type == "product":
        out.update({"product", "feature", "manual", "guide", "calibrate", "button", "mode"})
        if "calibrat" in " ".join(tokens):
            out.update({"calibrate", "calibration", "mode", "button", "press", "hold"})
    return sorted(out)


def _normalize(question: str) -> str:
    text = question.replace("\u2019", "'").strip()
    return re.sub(r"\s+", " ", text)
