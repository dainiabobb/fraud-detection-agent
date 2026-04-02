"""
Input sanitization utilities for the fraud detection agent.

Two public helpers are provided:

* sanitize_transaction  — validates and normalises an inbound transaction dict
                          before it is stored, embedded, or passed to a model.
* sanitize_llm_input    — strips prompt-injection patterns from any free-text
                          string before it is included in a model prompt.
"""

import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: tuple[str, ...] = (
    "transaction_id",
    "user_id",
    "amount",
    "timestamp",
)

_MERCHANT_NAME_MAX_LEN = 200

# Patterns that are canonically associated with prompt injection attempts.
# Kept as a compiled tuple for fast iteration.  The list intentionally stays
# conservative — overly broad filters would break legitimate merchant names.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)",
        r"(forget|disregard)\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)",
        r"\bsystem\s*:",           # "system:" role header injection
        r"\bassistant\s*:",        # "assistant:" role header injection
        r"\buser\s*:",             # "user:" role header injection
        r"<\s*/?system\s*>",       # XML-style role tags
        r"<\s*/?prompt\s*>",
        r"<\s*/?instruction\s*>",
        r"\[\s*INST\s*\]",         # Llama / Mistral instruction tokens
        r"\[/?SYS\]",
        r"###\s*(instruction|system|human|assistant)",  # Alpaca-style headers
        r"act\s+as\s+(if\s+you\s+are|a|an)\s+",        # persona hijacking
        r"you\s+are\s+now\s+(a|an)\s+",
        r"jailbreak",
        r"do\s+anything\s+now",    # DAN prompt family
    )
)

# Control characters except tab (\x09), newline (\x0a), carriage return (\x0d)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Runs of three or more whitespace characters (after control-char removal)
_EXCESS_WHITESPACE_RE = re.compile(r"[ \t]{3,}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sanitize_transaction(raw: dict) -> dict:
    """Validate and normalise a raw inbound transaction dict.

    Performs the following operations in order:
    1. Checks that all required fields are present.
    2. Strips leading/trailing whitespace from every string field.
    3. Validates that ``amount`` is a positive number.
    4. Truncates ``merchant_name`` to 200 characters.

    Args:
        raw: Unsanitized transaction payload from API Gateway / SQS.

    Returns:
        A new dict with cleaned values.  The original dict is not mutated.

    Raises:
        ValueError: If a required field is missing, ``amount`` is not a
                    positive number, or ``amount`` cannot be cast to float.
    """
    # --- required field presence check ---
    missing = [f for f in _REQUIRED_FIELDS if f not in raw or raw[f] is None]
    if missing:
        raise ValueError(f"Missing required transaction fields: {missing}")

    # Shallow copy so we never mutate the caller's dict.
    cleaned: dict[str, Any] = dict(raw)

    # --- strip strings ---
    for key, value in cleaned.items():
        if isinstance(value, str):
            cleaned[key] = value.strip()

    # --- amount validation ---
    try:
        amount = float(cleaned["amount"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Transaction amount must be numeric, got: {raw['amount']!r}"
        ) from exc

    if amount <= 0:
        raise ValueError(
            f"Transaction amount must be greater than 0, got: {amount}"
        )
    cleaned["amount"] = amount

    # --- merchant_name truncation ---
    if "merchant_name" in cleaned and isinstance(cleaned["merchant_name"], str):
        cleaned["merchant_name"] = cleaned["merchant_name"][:_MERCHANT_NAME_MAX_LEN]

    return cleaned


def sanitize_llm_input(text: str) -> str:
    """Remove prompt-injection patterns and normalise whitespace.

    This is a defence-in-depth measure applied to any user-supplied text
    before it is embedded in a model prompt.  It is not a complete security
    boundary on its own — the system prompt should also instruct the model to
    ignore out-of-context instructions.

    Operations performed in order:
    1. NFC-normalise unicode (prevent lookalike character smuggling).
    2. Strip C0/C1 control characters (keep tab, LF, CR).
    3. Remove known injection keyword patterns (replaced with a placeholder).
    4. Collapse runs of 3+ spaces/tabs to a single space.
    5. Strip leading/trailing whitespace.

    Args:
        text: Raw user-supplied string to be included in a model prompt.

    Returns:
        Cleaned string safe to embed in a prompt.
    """
    # 1. Unicode normalisation
    normalised = unicodedata.normalize("NFC", text)

    # 2. Strip unsafe control characters
    no_ctrl = _CONTROL_CHAR_RE.sub("", normalised)

    # 3. Remove injection patterns
    sanitised = no_ctrl
    for pattern in _INJECTION_PATTERNS:
        sanitised = pattern.sub("[REMOVED]", sanitised)

    # 4. Collapse excessive horizontal whitespace (preserve single newlines)
    sanitised = _EXCESS_WHITESPACE_RE.sub(" ", sanitised)

    # 5. Final trim
    return sanitised.strip()
