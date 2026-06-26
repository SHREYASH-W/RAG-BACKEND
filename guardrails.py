"""
Guardrails — input validation, topic enforcement, prompt injection
detection, output cleaning, and hallucination flagging.
"""

import re
import math
import logging
from collections import Counter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Input Guards
# ═══════════════════════════════════════════════════════════════

class InputGuardResult:
    """Result of an input guard check."""

    __slots__ = ("passed", "reason", "sanitized_input")

    def __init__(self, passed: bool, reason: str = "",
                 sanitized_input: str = ""):
        self.passed = passed
        self.reason = reason
        self.sanitized_input = sanitized_input


# ── Patterns that suggest prompt injection ─────────────────────
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"forget\s+(everything|all|your)\s+(instructions?|rules?|training)",
        r"you\s+are\s+now\s+a",
        r"pretend\s+(you\s+are|to\s+be)",
        r"act\s+as\s+(if|though|a)",
        r"new\s+instruction[s:]",
        r"system\s*prompt",
        r"reveal\s+(your|the)\s+(system|instructions?|prompt)",
        r"what\s+(are|is)\s+your\s+(instructions?|system\s*prompt|rules?)",
        r"override\s+(your|the|all)",
        r"jailbreak",
        r"dan\s+mode",
        r"developer\s+mode",
    ]
]

# ── Legal domain keywords ──────────────────────────────────────
_LEGAL_KEYWORDS: set[str] = {
    # Acts & statutes
    "constitution", "article", "section", "act", "law", "legal", "court",
    "supreme", "high", "district", "tribunal", "judge", "justice",
    "fundamental", "rights", "directive", "principles", "parliament",
    "legislature", "amendment", "schedule", "bill", "ordinance",
    "statute", "regulation", "rule", "notification", "gazette",
    # Specific Indian acts
    "ipc", "crpc", "cpc", "bns", "bnss", "bsa",
    "bharatiya", "nagarik", "suraksha", "sanhita", "nyaya",
    "indian", "penal", "code", "criminal", "civil", "procedure",
    "evidence", "contract", "partnership", "company", "companies",
    "income", "tax", "gst", "customs", "excise",
    "cyber", "information", "technology", "it act",
    "consumer", "protection", "environment", "pollution",
    "arbitration", "conciliation", "negotiable", "instruments",
    # Legal concepts
    "bail", "fir", "arrest", "custody", "warrant", "summons",
    "cognizable", "non-cognizable", "bailable", "non-bailable",
    "appeal", "petition", "writ", "habeas", "corpus", "mandamus",
    "certiorari", "prohibition", "quo warranto",
    "plaintiff", "defendant", "accused", "prosecution",
    "conviction", "acquittal", "sentence", "penalty", "fine",
    "imprisonment", "death", "punishment", "offense", "offence",
    "complaint", "charge", "trial", "hearing", "evidence",
    "witness", "testimony", "affidavit", "oath",
    "property", "land", "succession", "inheritance", "will",
    "marriage", "divorce", "maintenance", "custody",
    "citizenship", "passport", "visa", "immigration",
    "fundamental", "duties", "preamble", "sovereignty",
    "federalism", "separation", "powers",
    "lok", "sabha", "rajya", "president", "governor",
    "chief", "minister", "prime",
    "election", "commission", "comptroller", "auditor",
    "attorney", "general", "solicitor", "advocate",
    # Common question patterns
    "punishable", "liable", "guilty", "innocent", "legal",
    "illegal", "lawful", "unlawful", "constitutional",
    "unconstitutional", "void", "voidable", "valid",
}

_MAX_INPUT_LENGTH = 1000
_MIN_INPUT_LENGTH = 3


def _entropy(text: str) -> float:
    """Shannon entropy of a string — low entropy = gibberish / repeated chars."""
    if not text:
        return 0.0
    freq = Counter(text.lower())
    length = len(text)
    return -sum(
        (c / length) * math.log2(c / length)
        for c in freq.values()
        if c > 0
    )


def check_input(text: str) -> InputGuardResult:
    """
    Run all input guards on a user query.

    Returns an InputGuardResult with .passed = True if the query is acceptable.
    """
    stripped = text.strip()
    
    # ── 0. Greetings ───────────────────────────────────────────
    clean_text = re.sub(r"[^\w\s]", "", stripped.lower()).strip()
    if clean_text in {"hi", "hello", "hey", "greetings", "namaste", "good morning", "good afternoon", "good evening"}:
        return InputGuardResult(
            False,
            "Hello! I am Nyaya AI, your Indian legal advisor. How can I assist you with Indian law today?"
        )

    # ── 1. Empty / too short ───────────────────────────────────
    if len(stripped) < _MIN_INPUT_LENGTH:
        return InputGuardResult(False, "Please enter a valid question.")

    # ── 2. Too long ────────────────────────────────────────────
    if len(stripped) > _MAX_INPUT_LENGTH:
        return InputGuardResult(
            False,
            f"Question is too long (max {_MAX_INPUT_LENGTH} characters). "
            "Please be more concise.",
        )

    # ── 3. Gibberish detection (low entropy) ───────────────────
    ent = _entropy(stripped)
    if ent < 2.0 and len(stripped) > 10:
        return InputGuardResult(
            False,
            "Your input doesn't appear to be a valid question. "
            "Please rephrase.",
        )

    # ── 4. Prompt injection ────────────────────────────────────
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            logger.warning("Prompt injection blocked: %s", stripped[:80])
            return InputGuardResult(
                False,
                "I can only answer questions about Indian law. "
                "Please ask a legal question.",
            )

    # ── 5. Topic check (keyword-based, fast) ───────────────────
    words = set(re.findall(r"[a-zA-Z]+", stripped.lower()))

    # Allow short questions (might be follow-ups like "what about article 21?")
    if len(stripped) > 30:
        overlap = words & _LEGAL_KEYWORDS
        if not overlap:
            # No legal keywords at all — likely off-topic
            return InputGuardResult(
                False,
                "I specialize in Indian law. Please ask a question about "
                "Indian legal provisions, acts, or constitutional matters.",
            )

    return InputGuardResult(True, sanitized_input=stripped)


# ═══════════════════════════════════════════════════════════════
#  Topic Guard (LLM-based, for ambiguous cases)
# ═══════════════════════════════════════════════════════════════

def is_legal_topic_llm(classification_response: str) -> bool:
    """
    Parse the LLM topic classification response.
    Returns True if the query is about Indian law.
    """
    cleaned = classification_response.strip().upper()
    return "LEGAL" in cleaned


# ═══════════════════════════════════════════════════════════════
#  Output Guards
# ═══════════════════════════════════════════════════════════════

# Patterns to strip from LLM output — removes "sources say" language
_SOURCE_REFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "According to Source 1, ..." → ""
    (re.compile(
        r"(?:According to|As (?:per|stated in|mentioned in)|Based on|From)"
        r"\s+(?:Source\s*\d+|the (?:retrieved|provided|given|above)"
        r"\s+(?:context|documents?|passages?|text|information|excerpts?))"
        r"[,\s]*",
        re.IGNORECASE,
    ), ""),

    # "Source 1 states that" → ""
    (re.compile(
        r"Source\s*\d+\s+(?:states?|says?|mentions?|indicates?|shows?|"
        r"provides?|notes?|explains?|describes?)\s+(?:that\s+)?",
        re.IGNORECASE,
    ), ""),

    # "[Source 1]" or "(Source 1)"
    (re.compile(r"[\[\(]Source\s*\d+[\]\)]", re.IGNORECASE), ""),

    # "the retrieved documents" / "the provided context"
    (re.compile(
        r"(?:the\s+)?(?:retrieved|provided|given|supplied|available)"
        r"\s+(?:documents?|context|passages?|text|information|excerpts?)"
        r"\s*(?:do(?:es)?\s+not|don'?t|indicate|show|mention|state|say|suggest)",
        re.IGNORECASE,
    ), "The available legal provisions"),

    # "Based on the context provided"
    (re.compile(
        r"(?:Based on|According to|From)\s+the\s+(?:context|information|"
        r"documents?|data)\s+(?:provided|given|available|retrieved)",
        re.IGNORECASE,
    ), "Under the applicable legal provisions"),
]


def clean_output(text: str) -> str:
    """
    Post-process LLM output to remove any references to internal
    retrieval mechanics and clean up formatting.
    """
    result = text

    for pattern, replacement in _SOURCE_REFERENCE_PATTERNS:
        result = pattern.sub(replacement, result)

    # Clean up any resulting double spaces or empty lines
    result = re.sub(r"  +", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Fix sentences that now start with lowercase after removal
    result = re.sub(
        r"(?<=\. )([a-z])",
        lambda m: m.group(1).upper(),
        result,
    )

    return result.strip()


def check_grounding(answer: str, context_chunks: list[str]) -> float:
    """
    Basic grounding check — estimate what fraction of the answer's
    key claims appear in the retrieved context.

    Returns a confidence score between 0.0 and 1.0.
    This is a heuristic, not perfect, but catches egregious hallucinations.
    """
    if not context_chunks or not answer:
        return 0.0

    # Extract legal references from the answer
    article_refs = set(re.findall(r"Article\s+(\d+[A-Z]?)", answer, re.I))
    section_refs = set(re.findall(r"Section\s+(\d+[A-Z]?)", answer, re.I))

    if not article_refs and not section_refs:
        # No specific legal references to check — can't verify
        return 0.7  # Neutral confidence

    # Check how many of those references appear in the context
    context_text = " ".join(context_chunks)
    context_articles = set(re.findall(r"Article\s+(\d+[A-Z]?)", context_text, re.I))
    context_sections = set(re.findall(r"Section\s+(\d+[A-Z]?)", context_text, re.I))

    all_refs = article_refs | section_refs
    grounded_refs = (
        (article_refs & context_articles) |
        (section_refs & context_sections)
    )

    if not all_refs:
        return 0.7

    return len(grounded_refs) / len(all_refs)
