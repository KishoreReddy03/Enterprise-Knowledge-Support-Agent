"""
Dynamic Few-Shot Selector for the Drafting Agent.

Selects the most relevant gold-standard examples to inject into the
drafting prompt, teaching the LLM the correct tone, format, and citation
style by demonstration rather than by fine-tuning.

This is the "poor man's fine-tuning" — 80% of the benefit, 0% of the cost.
"""

import json
import logging
from pathlib import Path
from typing import Any

from core.ingestion.embedder import DocumentEmbedder

logger = logging.getLogger(__name__)

# Path to the gold-standard examples file
EXAMPLES_FILE = Path(__file__).parent.parent.parent / "data" / "few_shot_examples.json"


class FewShotSelector:
    """
    Dynamically selects the most relevant few-shot examples for a ticket.

    Uses cosine similarity between the ticket embedding and pre-computed
    example embeddings to find the best matches.
    """

    def __init__(self) -> None:
        """Load examples and pre-compute their embeddings."""
        self._embedder = DocumentEmbedder()
        self._examples: list[dict[str, Any]] = []
        self._example_embeddings: list[list[float]] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load examples on first use."""
        if self._loaded:
            return

        try:
            with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
                self._examples = json.load(f)

            # Pre-compute embeddings for all example tickets
            for example in self._examples:
                ticket_text = example.get("ticket_content", "")
                tags = example.get("tags", [])
                # Combine ticket content with tags for richer embedding
                embed_text = f"{ticket_text} {' '.join(tags)}"
                embedding = self._embedder.embed_text(embed_text)
                self._example_embeddings.append(embedding)

            self._loaded = True
            logger.info(
                f"FewShotSelector loaded {len(self._examples)} examples "
                f"from {EXAMPLES_FILE}"
            )

        except FileNotFoundError:
            logger.warning(f"Few-shot examples file not found: {EXAMPLES_FILE}")
            self._examples = []
            self._example_embeddings = []
            self._loaded = True

        except Exception as e:
            logger.error(f"Error loading few-shot examples: {e}")
            self._examples = []
            self._example_embeddings = []
            self._loaded = True

    def _cosine_similarity(
        self, vec_a: list[float], vec_b: list[float]
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = sum(a * a for a in vec_a) ** 0.5
        magnitude_b = sum(b * b for b in vec_b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    # Guardrail constants to prevent drift
    SIMILARITY_THRESHOLD = 0.45
    YEARLY_DECAY_RATE = 0.05
    MAX_AGE_PENALTY = 0.30

    def _is_structurally_consistent(self, example: dict[str, Any]) -> bool:
        """
        Validate that the example response is structurally consistent and
        adheres to correct formatting and citation styles to prevent drift.
        """
        # 1. Check basic presence of keys and types
        ticket = example.get("ticket_content")
        ideal = example.get("ideal_response")
        if not isinstance(ticket, str) or not ticket.strip():
            logger.warning(f"Example {example.get('id', '?')} rejected: empty ticket_content")
            return False
        if not isinstance(ideal, str) or not ideal.strip():
            logger.warning(f"Example {example.get('id', '?')} rejected: empty ideal_response")
            return False

        # 2. Check for common draft placeholder patterns
        placeholders = ["TODO", "FIXME", "[insert", "<insert", "[Your Name]", "<placeholder>", "{{placeholder}}"]
        for ph in placeholders:
            if ph.lower() in ideal.lower():
                logger.warning(f"Example {example.get('id', '?')} rejected: contains placeholder '{ph}'")
                return False

        # 3. Check for HTML tag style leakage (responses should be markdown/text)
        import re
        if re.search(r"<[a-z]+[^>]*>", ideal.lower()) and not re.search(r"<(express|custom|standard|connected)>", ideal.lower()):
            # Note: We allow account types in Connect docs like <express> but block HTML tags like <div>, <b> etc.
            if any(tag in ideal.lower() for tag in ["<div>", "<span>", "<p>", "<br>", "<b>", "<i>"]):
                logger.warning(f"Example {example.get('id', '?')} rejected: contains raw HTML tags")
                return False

        # 4. Check citation style structure:
        # Check balanced brackets
        open_brackets = ideal.count("[")
        close_brackets = ideal.count("]")
        if open_brackets != close_brackets:
            logger.warning(f"Example {example.get('id', '?')} rejected: mismatched brackets in response")
            return False
            
        # Check no empty brackets
        if "[]" in ideal:
            logger.warning(f"Example {example.get('id', '?')} rejected: empty brackets found")
            return False

        return True

    def select(
        self,
        ticket_content: str,
        primary_topic: str = "other",
        n: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Select the N most relevant few-shot examples for a ticket.

        Uses a hybrid approach:
        1. Topic matching (bonus for same primary_topic)
        2. Semantic similarity via embeddings
        3. Freshness, quality, and structural consistency guardrails

        Args:
            ticket_content: The customer's support ticket text.
            primary_topic: Classified topic from intake agent.
            n: Number of examples to return.

        Returns:
            List of the top-N most relevant example dicts.
        """
        self._ensure_loaded()

        if not self._examples:
            return []

        # Embed the incoming ticket
        try:
            ticket_embedding = self._embedder.embed_text(ticket_content)
        except Exception as e:
            logger.error(f"Failed to embed ticket for few-shot selection: {e}")
            return []

        # Score each example
        scored: list[tuple[float, int]] = []
        for i, (example, example_embedding) in enumerate(
            zip(self._examples, self._example_embeddings)
        ):
            # 1. Filter out inactive or deprecated examples
            if not example.get("active", True):
                logger.info(f"Skipping inactive example {example.get('id', '?')}")
                continue
            if example.get("deprecated", False):
                logger.info(f"Skipping deprecated example {example.get('id', '?')}")
                continue

            # 2. Filter out structurally or style inconsistent examples
            if not self._is_structurally_consistent(example):
                logger.info(f"Skipping structurally inconsistent example {example.get('id', '?')}")
                continue

            # 3. Base score: cosine similarity
            similarity = self._cosine_similarity(ticket_embedding, example_embedding)

            # 4. Reject examples below the minimum similarity threshold
            if similarity < self.SIMILARITY_THRESHOLD:
                logger.info(
                    f"Example {example.get('id', '?')} rejected: similarity {similarity:.3f} "
                    f"is below threshold {self.SIMILARITY_THRESHOLD}"
                )
                continue

            # 5. Topic bonus: +0.15 if primary_topic matches
            score = similarity
            example_topic = example.get("classification", {}).get(
                "primary_topic", "other"
            )
            if example_topic == primary_topic and primary_topic != "other":
                score += 0.15

            # 6. Apply age-based decay penalty if a date is present
            if "date" in example:
                try:
                    from datetime import datetime, date
                    example_date_str = example["date"]
                    example_date = datetime.strptime(example_date_str, "%Y-%m-%d").date()
                    
                    # Compute days old relative to current date
                    current_date = date.today()
                    days_old = (current_date - example_date).days
                    if days_old > 0:
                        years_old = days_old / 365.25
                        penalty = min(self.MAX_AGE_PENALTY, years_old * self.YEARLY_DECAY_RATE)
                        score -= penalty
                        logger.info(
                            f"Example {example.get('id', '?')} age penalty applied: -{penalty:.3f} "
                            f"({days_old} days old)"
                        )
                except Exception as ex:
                    logger.warning(
                        f"Failed to calculate age decay for example {example.get('id', '?')} "
                        f"with date '{example.get('date')}': {ex}"
                    )

            scored.append((score, i))

        # Sort by score descending and take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scored[:n]]

        selected = [self._examples[i] for i in top_indices]

        logger.info(
            f"Selected {len(selected)} few-shot examples: "
            f"{[ex.get('id', '?') for ex in selected]} "
            f"(topic={primary_topic})"
        )

        return selected

    def format_for_prompt(self, examples: list[dict[str, Any]]) -> str:
        """
        Format selected examples into a prompt-ready string.

        Args:
            examples: List of example dicts from select().

        Returns:
            Formatted string to inject into the drafting prompt.
        """
        if not examples:
            return ""

        sections: list[str] = []
        sections.append("═══ REFERENCE EXAMPLES ═══")
        sections.append(
            "Below are examples of how experienced support reps have handled "
            "similar tickets. Match this tone, structure, and citation style.\n"
        )

        for i, example in enumerate(examples, 1):
            classification = example.get("classification", {})
            sources = example.get("sources_used", [])

            sections.append(f"── Example {i} ──")
            sections.append(f"TICKET: {example.get('ticket_content', '')}")
            sections.append(
                f"CLASSIFICATION: complexity={classification.get('complexity', '?')}, "
                f"urgency={classification.get('urgency', '?')}, "
                f"topic={classification.get('primary_topic', '?')}"
            )
            sections.append(
                f"SOURCES USED: {', '.join(sources)}"
            )
            sections.append(
                f"IDEAL RESPONSE:\n{example.get('ideal_response', '')}"
            )
            sections.append("")  # blank line separator

        sections.append("═══ END EXAMPLES ═══\n")
        sections.append(
            "Now generate a response for the actual ticket below, "
            "following the same quality standard as the examples above."
        )

        return "\n".join(sections)


# Module-level singleton (lazy-loaded)
_selector: FewShotSelector | None = None


def get_few_shot_selector() -> FewShotSelector:
    """Get or create the FewShotSelector singleton."""
    global _selector
    if _selector is None:
        _selector = FewShotSelector()
    return _selector
