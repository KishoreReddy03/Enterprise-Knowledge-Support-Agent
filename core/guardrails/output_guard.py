"""
Output guardrail for post-processing draft replies.

Runs AFTER the Drafting Agent, BEFORE the Quality Gate, to verify
response safety, faithfulness to sources, and topic adherence.

This is the last safety check before a response reaches the customer.
"""

import logging
import re
from dataclasses import dataclass, field

from core.llm_client import call_fast

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OutputGuardrailResult:
    """
    Result from running output guardrails.

    Attributes:
        is_safe: Whether the output passed all checks.
        warnings: List of guardrail warnings triggered.
        hallucination_flags: Claims not grounded in sources.
        topic_violation: Whether response goes off-topic.
        pii_leaked: Whether response contains customer PII.
        modified_reply: The reply after any safety modifications.
        citation_mismatches: List of claims with mismatched citations.
    """
    is_safe: bool = True
    warnings: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    topic_violation: bool = False
    pii_leaked: bool = False
    modified_reply: str = ""
    citation_mismatches: list[str] = field(default_factory=list)
    grounding_score: float | None = None
    """Percentage of factual claims verified as grounded (0.0–1.0).
    None when grounding verification was not run (e.g. escalated ticket).
    """


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns that should NEVER appear in customer-facing responses
FORBIDDEN_OUTPUT_PATTERNS = [
    # Internal system details
    (r"(sk_live_|sk_test_|rk_live_)[a-zA-Z0-9]{10,}", "API key leak"),
    (r"(password|secret|token)\s*[:=]\s*\S+", "Credential leak"),
    (r"(postgresql|mysql|redis)://[^\s]+", "Database connection string"),
    (r"(localhost|127\.0\.0\.1|0\.0\.0\.0):\d+", "Internal service URL"),
    # Internal references
    (r"(internal[\s-]?note|rep[\s-]?guidance|DO_NOT_SEND)", "Internal note leaked"),
    # Inappropriate content
    (r"\b(fuck|shit|damn|hell|ass)\b", "Inappropriate language"),
]

# Off-topic indicators — response should be about Stripe, not competitors' internals
OFF_TOPIC_PATTERNS = [
    r"(?:use|switch\s+to|try|recommend)\s+(?:Adyen|Square|PayPal|Braintree)\s+instead",
    r"(?:Adyen|Square|PayPal|Braintree)\s+(?:is|are)\s+better",
]

# PII patterns that should not be echoed back in responses
PII_ECHO_PATTERNS = [
    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "SSN in response"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b", "Card number in response"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT GUARD
# ═══════════════════════════════════════════════════════════════════════════════

class OutputGuard:
    """
    Post-processing guardrail for draft replies.

    Runs four checks:
    1. Forbidden pattern detection (API keys, internal notes, profanity)
    2. PII echo prevention (don't repeat customer's sensitive data)
    3. Topic guardrail (stay within Stripe domain)
    4. Hallucination check (verify claims are grounded in sources)
    """

    def __init__(self) -> None:
        """Initialize the output guard."""
        self._forbidden_regexes = [
            (re.compile(p, re.IGNORECASE), desc)
            for p, desc in FORBIDDEN_OUTPUT_PATTERNS
        ]
        self._offtopic_regexes = [
            re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS
        ]
        self._pii_echo_regexes = [
            (re.compile(p), desc)
            for p, desc in PII_ECHO_PATTERNS
        ]
        logger.info("OutputGuard initialized")

    def _check_forbidden_patterns(self, reply: str) -> list[str]:
        """
        Check for forbidden patterns in the draft reply.

        Args:
            reply: Draft reply text.

        Returns:
            List of warning strings.
        """
        warnings = []
        for regex, description in self._forbidden_regexes:
            if regex.search(reply):
                warnings.append(f"forbidden_pattern: {description}")
        return warnings

    def _check_pii_echo(self, reply: str) -> list[str]:
        """
        Check if the reply echoes back customer PII.

        Args:
            reply: Draft reply text.

        Returns:
            List of PII types found in the reply.
        """
        warnings = []
        for regex, description in self._pii_echo_regexes:
            if regex.search(reply):
                warnings.append(f"pii_echo: {description}")
        return warnings

    def _check_topic(self, reply: str) -> bool:
        """
        Check if the reply stays within the Stripe domain.

        Args:
            reply: Draft reply text.

        Returns:
            True if an off-topic violation is detected.
        """
        for regex in self._offtopic_regexes:
            if regex.search(reply):
                return True
        return False

    def _redact_forbidden(self, reply: str) -> str:
        """
        Redact any forbidden patterns from the reply.

        Args:
            reply: Draft reply text.

        Returns:
            Reply with forbidden patterns redacted.
        """
        redacted = reply

        # Redact API keys
        redacted = re.sub(
            r"(sk_live_|sk_test_|rk_live_)[a-zA-Z0-9]{10,}",
            r"\1****REDACTED****",
            redacted,
        )

        # Redact database URLs
        redacted = re.sub(
            r"(postgresql|mysql|redis)://[^\s]+",
            "[REDACTED_CONNECTION_STRING]",
            redacted,
        )

        # Redact internal URLs
        redacted = re.sub(
            r"(localhost|127\.0\.0\.1|0\.0\.0\.0):\d+",
            "[INTERNAL_URL_REDACTED]",
            redacted,
        )

        return redacted

    async def _check_hallucination(
        self,
        reply: str,
        context: str,
        retrieved_chunks: list[dict] = None,
    ) -> list[str]:
        """
        Check if the reply contains claims not grounded in sources.

        Uses the high-fidelity NLI GroundingVerifier + physical Token Gate Override
        if retrieved_chunks are provided. Otherwise, falls back to prompt fact-checking.

        Args:
            reply: Draft reply text.
            context: The synthesized context from retrieval.
            retrieved_chunks: Optional raw retrieved chunk dicts for strict token gating.

        Returns:
            List of potentially hallucinated claims.
        """
        if not reply:
            return []

        # Skip if reply is very short (likely a clarifying question)
        if len(reply.split()) < 20:
            return []

        # High-Fidelity NLI + Deterministic Token Gate Override
        if retrieved_chunks:
            try:
                from core.guardrails.grounding_verifier import GroundingVerifier
                verifier = GroundingVerifier()
                report = await verifier.verify_grounding(reply, retrieved_chunks)
                if not report.is_safe or report.grounding_score < 1.0:
                    logger.warning(
                        f"GroundingVerifier flagged {len(report.ungrounded_segments)} ungrounded claims"
                    )
                    return report.ungrounded_segments
                return []
            except Exception as e:
                logger.error(f"GroundingVerifier in OutputGuard failed, falling back to prompt check: {e}")

        # Fallback to prompt-based check
        if not context:
            return []

        try:
            prompt = f"""You are a fact-checker. Compare the REPLY against the SOURCE CONTEXT.

SOURCE CONTEXT:
{context[:3000]}

REPLY TO CHECK:
{reply[:2000]}

List any factual claims in the REPLY that are NOT supported by the SOURCE CONTEXT.
Only flag specific technical claims (API behavior, parameter names, error codes).
Do NOT flag general advice, greetings, or opinions.

Return ONLY a JSON array of unsupported claims, or an empty array if all claims are grounded:
["claim 1 not in sources", "claim 2 not in sources"]

If all claims are supported, return: []"""

            response = await call_fast(prompt, max_tokens=512)

            # Parse the response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            response = response.replace("```json", "").replace("```", "").strip()

            import json
            claims = json.loads(response)

            if isinstance(claims, list) and len(claims) > 0:
                logger.warning(
                    f"Hallucination check found {len(claims)} ungrounded claims"
                )
                return [str(c) for c in claims]

            return []

        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
            return []

    async def check(
        self,
        draft_reply: str,
        synthesized_context: str = "",
        retrieved_chunks: list[dict] = None,
        sources_cited: list[dict] = None,
    ) -> OutputGuardrailResult:
        """
        Run all output guardrail checks on a draft reply.

        Args:
            draft_reply: The AI-generated draft reply.
            synthesized_context: The source context used to generate the reply.
            retrieved_chunks: Optional raw retrieved chunk dicts for strict token gating.
            sources_cited: Optional metadata list of sources cited in the response.

        Returns:
            OutputGuardrailResult with all check outcomes.
        """
        result = OutputGuardrailResult()
        result.modified_reply = draft_reply

        if not draft_reply:
            return result

        # 1. Check forbidden patterns
        forbidden_warnings = self._check_forbidden_patterns(draft_reply)
        if forbidden_warnings:
            result.warnings.extend(forbidden_warnings)
            result.is_safe = False
            result.modified_reply = self._redact_forbidden(draft_reply)
            logger.warning(f"Forbidden patterns found: {forbidden_warnings}")

        # 2. Check PII echo
        pii_warnings = self._check_pii_echo(draft_reply)
        if pii_warnings:
            result.warnings.extend(pii_warnings)
            result.pii_leaked = True
            logger.warning(f"PII echo detected: {pii_warnings}")

        # 3. Topic guardrail
        if self._check_topic(draft_reply):
            result.topic_violation = True
            result.warnings.append(
                "topic_violation: Response recommends competitor over Stripe"
            )
            logger.warning("Off-topic response detected")

        # 4. Hallucination check (async — uses semantic/physical NLI verifier or LLM prompt fallback)
        hallucination_flags = []
        verified_segments = []

        if retrieved_chunks:
            try:
                from core.guardrails.grounding_verifier import GroundingVerifier
                verifier = GroundingVerifier()
                report = await verifier.verify_grounding(draft_reply, retrieved_chunks)
                verified_segments = report.verified_segments
                result.grounding_score = report.grounding_score  # surface to caller
                if not report.is_safe or report.grounding_score < 1.0:
                    logger.warning(
                        f"GroundingVerifier flagged {len(report.ungrounded_segments)} ungrounded claims"
                    )
                    hallucination_flags = report.ungrounded_segments
            except Exception as e:
                logger.error(f"GroundingVerifier in OutputGuard failed, falling back to prompt check: {e}")
                # Fallback to prompt-based check
                hallucination_flags = await self._check_hallucination(
                    draft_reply, synthesized_context, retrieved_chunks=None
                )
        else:
            # Fallback to prompt-based check
            hallucination_flags = await self._check_hallucination(
                draft_reply, synthesized_context, retrieved_chunks=None
            )

        if hallucination_flags:
            result.hallucination_flags = hallucination_flags
            result.warnings.append(
                f"hallucination_risk: {len(hallucination_flags)} ungrounded claim(s)"
            )
            logger.warning(
                f"Hallucination flags: {hallucination_flags}"
            )

        # 5. Citation verification
        if sources_cited and verified_segments:
            try:
                from core.guardrails.grounding_verifier import CitationAttributionVerifier
                citation_verifier = CitationAttributionVerifier()
                citation_report = citation_verifier.verify_citations(
                    reply=draft_reply,
                    sources_cited=sources_cited,
                    verified_segments=verified_segments,
                    retrieved_chunks=retrieved_chunks,
                )
                if not citation_report.is_safe:
                    for mismatch in citation_report.mismatches:
                        result.citation_mismatches.append(mismatch.reason)
                        result.warnings.append(f"citation_mismatch: {mismatch.reason}")
                if citation_report.warnings:
                    result.warnings.extend(citation_report.warnings)
            except Exception as e:
                logger.error(f"CitationAttributionVerifier failed: {e}")

        logger.info(
            f"Output guard complete: safe={result.is_safe}, "
            f"warnings={len(result.warnings)}, "
            f"hallucinations={len(result.hallucination_flags)}, "
            f"citation_mismatches={len(result.citation_mismatches)}"
        )

        return result


# Module-level singleton
_guard: OutputGuard | None = None


def get_output_guard() -> OutputGuard:
    """Get or create the OutputGuard singleton."""
    global _guard
    if _guard is None:
        _guard = OutputGuard()
    return _guard
