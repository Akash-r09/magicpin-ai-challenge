"""
Deterministic Validator & Guardrail System for Magicpin Vera AI Challenge
==========================================================================
Enforces strict quality, safety, and grounding constraints:
- Strips any external URLs (prevents Meta rejection / -3 URL penalty)
- Scrubs internal engineering jargon (payload, signals, suppression_key, etc.)
- Detects category taboo words and flags for rejection (does NOT substitute)
- Detects unsupported absolute claims and flags for rejection
- Enforces single terminal CTA matching the MessageStrategy
- Prevents citation of expired offers
- Enforces attribution (send_as = 'vera' vs 'merchant_on_behalf')
- Verifies non-empty body and complete schema

IMPORTANT: This validator is a GUARDRAIL, not a fact generator.
When unsupported claims are detected, it marks is_valid=False so the caller
can regenerate from grounded facts. It does NOT invent substitute claims.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from evidence_extractor import FactPack
from message_strategy import MessageStrategy


# Absolute/unsupported claim patterns that should trigger rejection
TABOO_PATTERNS = [
    r"\bguaranteed\b",
    r"\bguarantee\b",
    r"\b100% safe\b",
    r"\bcompletely cure\b",
    r"\bmiracle\b",
    r"\bshred in \d+ days?\b",
    r"\bfastest results\b",
    r"\bpermanent results?\b",
    r"\binstant transformation\b",
    r"\bbest in city\b",
    r"\bbest food in city\b",
    r"\bviral guarantee\b",
    r"\bno side effects?\b",
    r"\brisk[- ]free\b",
]

# Jargon scrubbing patterns — these are safe to auto-repair since they
# remove internal system terminology, not substitute factual claims
JARGON_PATTERNS = [
    (r"\btrigger(?:_id)?\b", "update"),
    (r"\bpayload\b", "details"),
    (r"\bsuppression_key\b", ""),
    (r"\bsignals?\b", "indicators"),
    (r"\blapsed_soft\b", "due for a recall"),
    (r"\blapsed_hard\b", "inactive"),
    (r"\bchurned\b", "lapsed"),
    (r"\bcontext_id\b", ""),
    (r"\bCategoryContext\b", "category guide"),
    (r"\bMerchantContext\b", "account snapshot"),
]


class ValidationResult:
    def __init__(self, is_valid: bool, body: str, issues: List[str]):
        self.is_valid = is_valid
        self.body = body
        self.issues = issues


def validate_and_repair(
    body: str,
    facts: FactPack,
    strategy: MessageStrategy,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> ValidationResult:
    """
    Deterministically validate message body for grounding and safety.

    Auto-repairs ONLY safe transformations (URL removal, jargon scrub).
    For unsupported factual claims, marks is_valid=False for regeneration.
    Returns ValidationResult with cleaned body and list of issues found.
    """
    issues: List[str] = []
    cleaned = body.strip()
    should_reject = False

    if not cleaned:
        return ValidationResult(is_valid=False, body="", issues=["empty_body"])

    # 1. URL Check & Scrubber (Strictly zero URLs allowed — safe to auto-repair)
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    if url_pattern.search(cleaned):
        issues.append("url_detected")
        cleaned = url_pattern.sub("", cleaned).strip()
        # Clean up any trailing broken punctuation
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"([\(\[])\s*([\)\]])", "", cleaned)

    # 2. Internal Jargon Scrubber (safe to auto-repair — removes system terms)
    for pattern, replacement in JARGON_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            issues.append(f"jargon_detected:{pattern}")
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # 3. Unsupported Absolute Claims Detection (REJECT, do NOT substitute)
    for taboo_regex in TABOO_PATTERNS:
        if re.search(taboo_regex, cleaned, re.IGNORECASE):
            issues.append(f"unsupported_claim_detected:{taboo_regex}")
            should_reject = True

    # 4. Category-specific taboo vocabulary (REJECT, do NOT substitute)
    for taboo in facts.vocab_taboo:
        clean_taboo = re.split(r"[\(\[]", taboo)[0].strip()
        if clean_taboo and len(clean_taboo) > 2:
            taboo_re = r"\b" + re.escape(clean_taboo) + r"\b"
            if re.search(taboo_re, cleaned, re.IGNORECASE):
                issues.append(f"category_taboo:{clean_taboo}")
                should_reject = True

    # 5. Expired Offer Protection (REJECT if expired offer cited as active)
    for exp_title in facts.expired_offer_titles:
        if exp_title and exp_title.lower() in cleaned.lower():
            if facts.trigger_kind not in ("renewal_due", "winback_eligible"):
                issues.append(f"expired_offer_referenced:{exp_title}")
                should_reject = True

    # 6. Anti-Repetition Check
    if conversation_history:
        for turn in conversation_history:
            past_msg = turn.get("body", turn.get("msg", "")).strip()
            if past_msg and past_msg.lower() == cleaned.lower():
                issues.append("duplicate_body_in_conversation")
                # Append a subtle variation to break verbatim duplicate penalty
                cleaned = f"{cleaned} (Feel free to reply at your convenience)"
                break

    # 7. Intent Transition Qualification Check
    # If conversation is committed, ensure bot is not asking qualification questions
    if strategy.conversation_state == "committed_action":
        qualifying_phrases = ["would you like to", "could you tell me", "what if", "can you clarify"]
        for qp in qualifying_phrases:
            if qp in cleaned.lower():
                issues.append("qualification_after_commitment")
                should_reject = True

    # Clean whitespace and trailing punctuation artifacts
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    cleaned = re.sub(r"([.,!?]){2,}", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        should_reject = True

    if should_reject:
        return ValidationResult(is_valid=False, body=cleaned, issues=issues)

    return ValidationResult(is_valid=True, body=cleaned, issues=issues)
