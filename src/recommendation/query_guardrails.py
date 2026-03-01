"""
Query guardrails for recommendation requests.
Rejects nonsensical or unclear prompts before LLM/retrieval calls.
"""

from dataclasses import dataclass
import re


@dataclass
class QueryGuardrailResult:
    """Result of guardrail validation."""
    is_valid: bool
    message: str
    normalized_query: str


class QueryGuardrails:
    """Deterministic query guardrails for input quality."""

    _KNOWN_SHORT_TOKENS = {
        "ai", "ml", "hr", "qa", "ui", "ux", "sql", "aws", "gcp",
        "api", "jd", "kpi", "b2b", "b2c"
    }

    _GIBBERISH_MARKERS = {
        "asdf", "qwerty", "zxcv", "lorem", "ipsum", "blah", "random"
    }

    _VALIDATION_MESSAGE = (
        "Please provide a valid prompt or instruction. "
        "Include role/skills or a job requirement (example: "
        "'Need Java developers with teamwork and communication skills')."
    )

    _PROMPT_INJECTION_PATTERNS = [
        r"ignore (all )?(previous|prior) instructions",
        r"disregard (all )?(previous|prior) instructions",
        r"override (system|developer|safety) (prompt|instruction|instructions)",
        r"reveal (the )?(system|developer) prompt",
        r"show (me )?(the )?(system|developer) prompt",
        r"print (the )?(hidden|secret) instructions",
        r"you are now",
        r"act as .* without restrictions",
        r"bypass (guardrails|safety|filters?)",
        r"jailbreak",
        r"do anything now",
        r"developer mode",
        r"simulate .* tool output",
    ]

    _DOMAIN_KEYWORDS = {
        "hire", "hiring", "recruit", "recruitment", "candidate", "candidates",
        "role", "roles", "job", "jd", "position", "assessment", "assessments",
        "test", "tests", "skills", "skill", "developer", "engineer", "analyst",
        "manager", "sales", "communication", "collaboration", "leadership",
        "teamwork", "behavioral", "personality", "technical", "coding",
        "programming", "python", "java", "sql", "data", "problem-solving",
        "problem", "graduate", "graduates", "customer", "support", "soft", "hard"
    }

    @staticmethod
    def normalize(query: str) -> str:
        """Normalize query text without altering intent."""
        if not query:
            return ""
        query = " ".join(query.split())
        query = re.sub(r"[^\w\s\-.,!?/+()]", " ", query)
        return " ".join(query.split()).strip()

    @classmethod
    def validate(cls, query: str) -> QueryGuardrailResult:
        """Validate whether a query is clear enough for recommendations."""
        normalized = cls.normalize(query)
        if not normalized:
            return QueryGuardrailResult(False, cls._VALIDATION_MESSAGE, normalized)

        # Reject symbol-heavy inputs like "??? !!! ###".
        alpha_num_count = sum(ch.isalnum() for ch in normalized)
        if alpha_num_count < max(3, int(len(normalized) * 0.4)):
            return QueryGuardrailResult(False, cls._VALIDATION_MESSAGE, normalized)

        lowered = normalized.lower()
        tokens = re.findall(r"[a-z0-9+/#-]+", lowered)
        if not tokens:
            return QueryGuardrailResult(False, cls._VALIDATION_MESSAGE, normalized)

        # Reject known gibberish markers.
        if any(marker in lowered for marker in cls._GIBBERISH_MARKERS):
            return QueryGuardrailResult(False, cls._VALIDATION_MESSAGE, normalized)

        # Block prompt injection / instruction override attempts.
        if any(re.search(pattern, lowered) for pattern in cls._PROMPT_INJECTION_PATTERNS):
            return QueryGuardrailResult(False, cls._VALIDATION_MESSAGE, normalized)

        meaningful_tokens = []
        for token in tokens:
            if len(token) >= 3:
                meaningful_tokens.append(token)
            elif token in cls._KNOWN_SHORT_TOKENS:
                meaningful_tokens.append(token)

        # If query is too short or vague, request clarification.
        if len(meaningful_tokens) < 2:
            return QueryGuardrailResult(False, cls._VALIDATION_MESSAGE, normalized)

        # Reject repeated-token noise like "java java java java".
        unique_meaningful = set(meaningful_tokens)
        if len(meaningful_tokens) >= 4 and len(unique_meaningful) == 1:
            return QueryGuardrailResult(False, cls._VALIDATION_MESSAGE, normalized)

        # Reject inputs unrelated to hiring/assessment context.
        # Example rejected: "my name is mohit"
        if not any(token in cls._DOMAIN_KEYWORDS for token in meaningful_tokens):
            return QueryGuardrailResult(False, cls._VALIDATION_MESSAGE, normalized)

        return QueryGuardrailResult(True, "", normalized)
