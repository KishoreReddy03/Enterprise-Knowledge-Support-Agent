"""
Input guardrail for pre-processing support tickets.

Runs BEFORE the Intake Agent to detect prompt injection, mask PII,
and enforce rate limits. Follows the FLAG approach — suspicious inputs
are flagged and auto-escalated, not blocked.

This is the first line of defense in the production pipeline.

⚠️ PROMPT INJECTION REGEX LIMITATIONS & HARDENING AUDIT
======================================================
The current pre-processing injection detection is strictly heuristic-driven and
uses pre-compiled Regular Expressions (re). This represents a standard first-line
operational check but carries severe vulnerabilities under production conditions:

1. THE SYSTEMIC LIMITATIONS OF REGEX:
   * Semantic Jailbreaks: Conversational shifts, virtual simulations, roleplay 
     ("Do Anything Now" / DAN), and nested adversarial framing will bypass literal
     word boundaries.
   * Indirect Prompt Injection: Malicious instructions hidden inside third-party 
     sources (e.g. a scraped GitHub issue containing instruction overrides like 
     "output only standard API URLs") will bypass the input guard completely, 
     contaminating downstream Retrieval & Synthesis states.
   * Multilingual Attacks: Translating adversarial overrides into non-English
     languages or mixed-cased lexical variants will skip past standard patterns.
   * Obfuscated & Encoded Payloads: Obfuscation methods such as Base64/Hex encoding,
     URL encodings, or ciphered text segments easily defeat literal strings.

2. HYBRID HARDENING ROADMAP (PRODUCTION GRADE):
   To mitigate these bypass risks, the pre-processing guardrail must transition from
   static regex rules to a multi-layered security grid:
   * Layer A - Specialized Classification: Integrate a specialized, lightweight
     guard model (e.g. Meta's Llama Guard or Prompt Guard) via local inference, 
     detecting adversarial prompt intent with semantic awareness.
   * Layer B - Vector Semantic Guard: Maintain a local vector index of historical
     adversarial injection templates. Run cosine similarity checks against raw 
     inputs to catch semantic jailbreak clusters.
   * Layer C - Dual-Pass Context Verification: Execute the injection verifier 
     on both raw incoming customer tickets (input pass) AND retrieved raw database 
     chunks inside the Retrieval Agent (context pass) to neutralize indirect injections.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.redis_client import get_redis_client

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GuardrailResult:
    """
    Result from running input guardrails.

    Attributes:
        is_safe: Whether the input passed all checks.
        sanitized_content: The ticket content after PII masking.
        warnings: List of guardrail warnings triggered.
        pii_detected: Types of PII found and masked.
        should_escalate: Whether to auto-escalate due to guardrail flags.
        query_hash: SHA256 hash of the sanitized content (for caching).
    """
    is_safe: bool = True
    sanitized_content: str = ""
    warnings: list[str] = field(default_factory=list)
    pii_detected: list[str] = field(default_factory=list)
    should_escalate: bool = False
    query_hash: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Prompt injection patterns — phrases that attempt to override system behavior
INJECTION_PATTERNS = [
    # Direct instruction override
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?|prompts?)",
    r"disregard\s+(all\s+)?(previous|your)\s+(instructions?|rules?|prompts?)",
    r"forget\s+(all\s+)?(previous|your)\s+(instructions?|rules?)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"act\s+as\s+(a|an)\s+(?!support|stripe)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*you\s+are",
    # Data exfiltration
    r"(print|output|reveal|show|tell\s+me)\s+(your|the|all)\s+(system\s+)?(prompt|instructions?|rules?|context)",
    r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
    # Encoding tricks
    r"base64\s*(decode|encode)",
    r"\\x[0-9a-fA-F]{2}",
    # Role manipulation
    r"(jailbreak|DAN|developer\s+mode|unrestricted\s+mode)",
]

# PII patterns for detection and masking
PII_PATTERNS = {
    "credit_card": {
        "pattern": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        "mask": "****-****-****-XXXX",
        "description": "Credit card number",
    },
    "ssn": {
        "pattern": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "mask": "***-**-XXXX",
        "description": "Social Security Number",
    },
    "stripe_secret_key": {
        "pattern": r"\b(sk_live_[a-zA-Z0-9]{24,})\b",
        "mask": "sk_live_****REDACTED****",
        "description": "Stripe live secret key",
    },
    "stripe_restricted_key": {
        "pattern": r"\b(rk_live_[a-zA-Z0-9]{24,})\b",
        "mask": "rk_live_****REDACTED****",
        "description": "Stripe restricted key",
    },
    "api_key_generic": {
        "pattern": r"\b(sk_test_[a-zA-Z0-9]{24,})\b",
        "mask": "sk_test_****REDACTED****",
        "description": "Stripe test secret key",
    },
    "phone": {
        "pattern": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "mask": "***-***-XXXX",
        "description": "Phone number",
    },
}

# HTML/script patterns to sanitize
SANITIZE_PATTERNS = [
    (r"<script[^>]*>.*?</script>", "", "Script tag"),
    (r"<iframe[^>]*>.*?</iframe>", "", "Iframe tag"),
    (r"<[^>]+on\w+\s*=", "<", "Event handler attribute"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT GUARD
# ═══════════════════════════════════════════════════════════════════════════════

class InputGuard:
    """
    Pre-processing guardrail for incoming support tickets.

    Runs three checks in order:
    1. Prompt injection detection
    2. PII detection and masking
    3. Rate limiting (via Redis)

    Follows the FLAG approach: suspicious inputs are flagged and
    auto-escalated to a human, not blocked outright.
    """

    def __init__(self) -> None:
        """Initialize the input guard."""
        self._redis = get_redis_client()
        self._injection_regexes = [
            re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
        ]
        self._pii_regexes = {
            name: re.compile(info["pattern"])
            for name, info in PII_PATTERNS.items()
        }
        logger.info("InputGuard initialized")

    def _check_prompt_injection(self, content: str) -> list[str]:
        """
        Check for prompt injection attempts.

        Args:
            content: Raw ticket content.

        Returns:
            List of warning strings for any detected injection patterns.
        """
        warnings = []
        content_lower = content.lower()

        for regex in self._injection_regexes:
            match = regex.search(content_lower)
            if match:
                warnings.append(
                    f"prompt_injection_detected: matched pattern near "
                    f"'{match.group()[:50]}'"
                )

        return warnings

    def _mask_pii(self, content: str) -> tuple[str, list[str]]:
        """
        Detect and mask PII in the ticket content.

        Args:
            content: Raw ticket content.

        Returns:
            Tuple of (masked_content, list of PII types found).
        """
        masked = content
        pii_found = []

        for pii_type, regex in self._pii_regexes.items():
            matches = regex.findall(masked)
            if matches:
                pii_info = PII_PATTERNS[pii_type]
                pii_found.append(pii_info["description"])
                masked = regex.sub(pii_info["mask"], masked)
                logger.info(
                    f"PII masked: {pii_info['description']} "
                    f"({len(matches)} occurrence(s))"
                )

        return masked, pii_found

    def _sanitize_html(self, content: str) -> str:
        """
        Remove potentially dangerous HTML/script content.

        Args:
            content: Ticket content.

        Returns:
            Sanitized content.
        """
        sanitized = content
        for pattern, replacement, name in SANITIZE_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE | re.DOTALL)
        return sanitized

    def _compute_hash(self, content: str) -> str:
        """
        Compute a deterministic hash of the content for caching.

        Args:
            content: Sanitized ticket content.

        Returns:
            SHA256 hex digest.
        """
        return hashlib.sha256(content.strip().lower().encode()).hexdigest()[:16]

    async def _check_rate_limit(
        self,
        customer_id: str,
    ) -> tuple[bool, int]:
        """
        Check if the customer has exceeded the rate limit.

        Args:
            customer_id: The customer identifier.

        Returns:
            Tuple of (is_allowed, current_count).
        """
        # 20 requests per minute per customer
        return await self._redis.check_rate_limit(
            identifier=customer_id,
            max_requests=20,
            window_seconds=60,
        )

    async def check(
        self,
        ticket_content: str,
        customer_id: str = "unknown",
    ) -> GuardrailResult:
        """
        Run all input guardrail checks.

        Performs injection detection, PII masking, HTML sanitization,
        and rate limiting. Returns a GuardrailResult with warnings
        and the sanitized content.

        Args:
            ticket_content: Raw customer ticket text.
            customer_id: Customer identifier for rate limiting.

        Returns:
            GuardrailResult with all check outcomes.
        """
        result = GuardrailResult()
        content = ticket_content

        # 1. Check for prompt injection
        injection_warnings = self._check_prompt_injection(content)
        if injection_warnings:
            result.warnings.extend(injection_warnings)
            result.should_escalate = True
            logger.warning(
                f"Prompt injection detected for customer {customer_id}: "
                f"{injection_warnings}"
            )

        # 2. Sanitize HTML/scripts
        content = self._sanitize_html(content)

        # 3. Mask PII
        content, pii_types = self._mask_pii(content)
        if pii_types:
            result.pii_detected = pii_types
            result.warnings.append(
                f"pii_detected_and_masked: {', '.join(pii_types)}"
            )
            logger.info(f"PII masked for customer {customer_id}: {pii_types}")

        # 4. Rate limiting
        is_allowed, count = await self._check_rate_limit(customer_id)
        if not is_allowed:
            result.warnings.append(
                f"rate_limit_exceeded: {count}/20 requests in 60s"
            )
            result.should_escalate = True
            logger.warning(f"Rate limit exceeded for {customer_id}: {count}/20")

        # Compute query hash for semantic caching
        result.sanitized_content = content
        result.query_hash = self._compute_hash(content)
        result.is_safe = len(injection_warnings) == 0

        logger.info(
            f"Input guard complete: safe={result.is_safe}, "
            f"warnings={len(result.warnings)}, "
            f"pii_masked={len(pii_types)}"
        )

        return result


# Module-level singleton
_guard: InputGuard | None = None


def get_input_guard() -> InputGuard:
    """Get or create the InputGuard singleton."""
    global _guard
    if _guard is None:
        _guard = InputGuard()
    return _guard
