import json
import logging
from dataclasses import dataclass, field
from typing import Any
from core.llm_client import call_fast, call_strong

logger = logging.getLogger(__name__)

@dataclass
class SegmentVerification:
    segment: str
    is_factual_claim: bool
    is_grounded: bool
    source_chunk_ids: list[str] = field(default_factory=list)
    reason: str = ""

@dataclass
class GroundingReport:
    is_safe: bool
    grounding_score: float  # Percentage of factual claims that are grounded
    verified_segments: list[SegmentVerification]
    ungrounded_segments: list[str] = field(default_factory=list)

class GroundingVerifier:
    """
    Executes segment-level provenance tracking and natural language inference (NLI)
    to trace every claim in the generated answer to specific retrieved source chunks.
    """

    async def verify_grounding(
        self,
        reply: str,
        retrieved_chunks: list[dict[str, Any]],
        strict_threshold: float = 1.0
    ) -> GroundingReport:
        """
        Traces every segment in the generated reply back to the source chunks.
        Flags any ungrounded segment.
        """
        if not reply:
            return GroundingReport(is_safe=True, grounding_score=1.0, verified_segments=[])

        if not retrieved_chunks:
            # If no chunks were retrieved and the reply contains any factual claims, it's unsafe
            return GroundingReport(
                is_safe=False,
                grounding_score=0.0,
                verified_segments=[
                    SegmentVerification(
                        segment=reply,
                        is_factual_claim=True,
                        is_grounded=False,
                        reason="No source chunks retrieved to back up the reply."
                    )
                ],
                ungrounded_segments=[reply]
            )

        # Build clean string of chunks for the prompt
        chunks_str = ""
        for i, chunk in enumerate(retrieved_chunks):
            chunk_id = chunk.get("id") or chunk.get("chunk_id") or f"chunk_{i}"
            payload = chunk.get("payload") or {}
            content = payload.get("content") or chunk.get("text") or chunk.get("content") or ""
            title = payload.get("title") or chunk.get("title") or "Untitled"
            chunks_str += f"--- START CHUNK ID: {chunk_id} (Title: {title}) ---\n{content}\n--- END CHUNK ID: {chunk_id} ---\n\n"

        prompt = f"""You are an advanced Natural Language Inference (NLI) and Fact-Checking system.
Your task is to analyze a support response segment-by-segment and trace every factual claim back to its specific supporting SOURCE CHUNK ID.

REPLY TO CHECK:
{reply}

SOURCE CHUNKS:
{chunks_str}

Instructions:
1. Break down the REPLY into individual sentences/segments.
2. For each segment, decide if it makes a specific technical/factual claim (e.g. API methods, parameters, error behaviors).
3. If it makes a claim, search the SOURCE CHUNKS to see if the claim is fully supported.
4. If it is supported, map it to the corresponding source chunk ID(s).
5. If it is NOT supported, or if it asserts something not mentioned in the chunks, mark "is_grounded" as false.
6. Provide a short reason explaining your trace.

Return ONLY a valid JSON array of objects with the exact keys:
[
  {{
    "segment": "sentence from reply",
    "is_factual_claim": true/false,
    "is_grounded": true/false,
    "source_chunk_ids": ["chunk_id_1"],
    "reason": "explanation of NLI trace"
  }}
]"""

        try:
            response = await call_strong(prompt, max_tokens=1024, temperature=0.0)
            
            # Clean and parse JSON block defensively
            response = response.strip()
            raw_results = None
            
            # Try parsing inside markdown code blocks first
            import re
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if code_block_match:
                try:
                    raw_results = json.loads(code_block_match.group(1).strip())
                except Exception:
                    pass
            
            # If that failed, search for the outer bracket array [...]
            if raw_results is None:
                array_match = re.search(r'(\[\s*\{[\s\S]*\}\s*\])', response)
                if array_match:
                    try:
                        raw_results = json.loads(array_match.group(1))
                    except Exception:
                        pass
                        
            # Final fallback
            if raw_results is None:
                raw_results = json.loads(response)
            
            # Build chunk text map for rapid exact matching
            chunk_texts = {}
            for idx, chunk in enumerate(retrieved_chunks):
                cid = chunk.get("id") or chunk.get("chunk_id") or f"chunk_{idx}"
                payload = chunk.get("payload") or {}
                content = (payload.get("content") or chunk.get("text") or chunk.get("content") or "").lower()
                title = (payload.get("title") or chunk.get("title") or "").lower()
                chunk_texts[cid] = content + " " + title

            # Technical terms pattern (snake_case, camelCase, API paths, quoted strings, dash parameters)
            import re
            tech_pattern = re.compile(
                r'\b([a-zA-Z0-9]+_[a-zA-Z0-9_]+|[a-z0-9]+[A-Z][a-zA-Z0-9]*|[a-zA-Z]+-[a-zA-Z0-9-]+)\b|'  # snake_case / camelCase / dash-terms
                r'/[a-zA-Z0-9_]+(?:/[a-zA-Z0-9_]+)+|'                                                   # API paths
                r'[\'"`]([a-zA-Z0-9_-]{3,})[\'"`]'                                                      # Quoted params
            )

            verified_segments = []
            ungrounded_segments = []
            
            total_claims = 0
            grounded_claims = 0

            for r in raw_results:
                seg = SegmentVerification(
                    segment=r["segment"],
                    is_factual_claim=r["is_factual_claim"],
                    is_grounded=r["is_grounded"],
                    source_chunk_ids=r.get("source_chunk_ids", []),
                    reason=r.get("reason", "")
                )
                
                # --- DETERMINISTIC TOKEN GATE OVERRIDE ---
                # Solves the "LLM-as-judge" overestimation problem by ensuring that technical API parameters,
                # endpoints, error codes, and headers are physically present in the claimed supporting chunks.
                if seg.is_factual_claim and seg.is_grounded:
                    tech_terms = set()
                    for match in tech_pattern.finditer(seg.segment):
                        term = match.group(0)
                        if term.startswith(("'", '"', "`")) and term.endswith(("'", '"', "`")):
                            term = term[1:-1]
                        term_lower = term.lower().strip()
                        if term_lower not in ("http", "https", "true", "false", "null", "none", "stripe"):
                            tech_terms.add(term_lower)
                            
                    if tech_terms and seg.source_chunk_ids:
                        missing_terms = []
                        for term in tech_terms:
                            found = False
                            for cid in seg.source_chunk_ids:
                                chunk_content = chunk_texts.get(cid, "")
                                if term in chunk_content:
                                    found = True
                                    break
                            if not found:
                                missing_terms.append(term)
                                
                        if missing_terms:
                            logger.warning(
                                f"[TOKEN GATE OVERRIDE] Overriding LLM judge for segment: '{seg.segment}'. "
                                f"Technical terms {missing_terms} are absent in claimed chunks {seg.source_chunk_ids}."
                            )
                            seg.is_grounded = False
                            seg.reason = (
                                f"Deterministic override: Technical terms {missing_terms} "
                                f"do not exist in the claimed supporting source chunks."
                            )

                verified_segments.append(seg)

                if seg.is_factual_claim:
                    total_claims += 1
                    if seg.is_grounded:
                        grounded_claims += 1
                    else:
                        ungrounded_segments.append(seg.segment)
            
            grounding_score = grounded_claims / total_claims if total_claims > 0 else 1.0
            is_safe = grounding_score >= strict_threshold

            logger.info(
                f"Grounding verification complete: score={grounding_score:.2f}, "
                f"total_claims={total_claims}, ungrounded_segments={len(ungrounded_segments)}"
            )

            return GroundingReport(
                is_safe=is_safe,
                grounding_score=grounding_score,
                verified_segments=verified_segments,
                ungrounded_segments=ungrounded_segments
            )

        except Exception as e:
            logger.error(f"Grounding verifier failed: {e}")
            # Fail-safe: escalate if verifier breaks
            return GroundingReport(
                is_safe=False,
                grounding_score=0.0,
                verified_segments=[
                    SegmentVerification(
                        segment=reply,
                        is_factual_claim=True,
                        is_grounded=False,
                        reason=f"Verifier error: {str(e)}"
                    )
                ],
                ungrounded_segments=[reply]
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CITATION VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CitationMismatch:
    segment: str
    citation_marker: str
    expected_chunk_id: str
    actual_supporting_chunks: list[str]
    reason: str


@dataclass
class CitationVerificationReport:
    is_safe: bool
    mismatches: list[CitationMismatch]
    warnings: list[str] = field(default_factory=list)


class CitationAttributionVerifier:
    """
    Enforces citation-level attribution by verifying that each inline citation
    marker (e.g. [1], [2]) maps to a chunk ID that actually supports the claim
    made in that sentence/segment.
    """
    
    def verify_citations(
        self,
        reply: str,
        sources_cited: list[dict[str, Any]],
        verified_segments: list[Any],
        retrieved_chunks: list[dict[str, Any]] = None,
    ) -> CitationVerificationReport:
        """
        Cross-references inline citations in reply segments with NLI grounding analysis
        to verify that the exact claim citing a source is actually supported by that source.
        Uses a physical token gate override to verify that all cited technical keywords are
        present verbatim in the specific cited source chunk.
        """
        import re
        mismatches = []
        warnings = []
        
        if not reply or not sources_cited or not verified_segments:
            return CitationVerificationReport(is_safe=True, mismatches=[])
            
        # Map 1-indexed citation numbers (e.g. "1" from [1]) to chunk_ids
        citation_to_chunk = {}
        for idx, src in enumerate(sources_cited):
            chunk_id = src.get("chunk_id")
            if chunk_id:
                citation_to_chunk[str(idx + 1)] = chunk_id
                
        # Build chunk text map for rapid physical presence checks
        chunk_texts = {}
        if retrieved_chunks:
            for idx, chunk in enumerate(retrieved_chunks):
                cid = chunk.get("id") or chunk.get("chunk_id") or f"chunk_{idx}"
                payload = chunk.get("payload") or {}
                content = (payload.get("content") or chunk.get("text") or chunk.get("content") or "").lower()
                title = (payload.get("title") or chunk.get("title") or "").lower()
                chunk_texts[cid] = content + " " + title

        # Regex to find citation tags like [1], [2], [3]
        citation_pattern = re.compile(r'\[(\d+)\]')
        
        for seg in verified_segments:
            segment_text = seg.segment
            is_factual = getattr(seg, "is_factual_claim", False)
            is_grounded = getattr(seg, "is_grounded", False)
            actual_chunks = getattr(seg, "source_chunk_ids", [])
            
            # Normalize actual chunks in case they are nested or differently mapped
            normalized_actual_chunks = []
            for ac in actual_chunks:
                if isinstance(ac, dict):
                    normalized_actual_chunks.append(ac.get("chunk_id") or ac.get("id") or "")
                else:
                    normalized_actual_chunks.append(str(ac))
            
            # Find all citation markers in this segment
            markers = citation_pattern.findall(segment_text)
            if not markers:
                continue
                
            for marker in markers:
                expected_chunk_id = citation_to_chunk.get(marker)
                if not expected_chunk_id:
                    warnings.append(
                        f"out_of_bounds_citation: Response contains inline citation [{marker}] "
                        f"but only {len(sources_cited)} sources were cited in the metadata."
                    )
                    mismatches.append(
                        CitationMismatch(
                            segment=segment_text,
                            citation_marker=f"[{marker}]",
                            expected_chunk_id="UNKNOWN",
                            actual_supporting_chunks=normalized_actual_chunks,
                            reason=f"Inline citation index [{marker}] is out of bounds."
                        )
                    )
                    continue
                
                # --- DETERMINISTIC SCOPED TECHNICAL TOKEN GATE OVERRIDE ---
                if expected_chunk_id in chunk_texts:
                    chunk_content = chunk_texts[expected_chunk_id]
                    
                    # API paths, snake_case, camelCase, quoted parameters, dash parameters
                    tech_pattern = re.compile(
                        r'\b([a-zA-Z0-9]+_[a-zA-Z0-9_]+|[a-z0-9]+[A-Z][a-zA-Z0-9]*|[a-zA-Z]+-[a-zA-Z0-9-]+)\b|'
                        r'/[a-zA-Z0-9_]+(?:/[a-zA-Z0-9_]+)+|'
                        r'[\'"`]([a-zA-Z0-9_-]{3,})[\'"`]'
                    )
                    
                    tech_terms = set()
                    for match in tech_pattern.finditer(segment_text):
                        term = match.group(0)
                        if term.startswith(("'", '"', "`")) and term.endswith(("'", '"', "`")):
                            term = term[1:-1]
                        term_lower = term.lower().strip()
                        if term_lower not in ("http", "https", "true", "false", "null", "none", "stripe"):
                            tech_terms.add(term_lower)
                            
                    if tech_terms:
                        missing_terms = [
                            term for term in tech_terms
                            if term not in chunk_content
                        ]
                        if missing_terms:
                            reason = (
                                f"Deterministic citation mismatch override: Technical terms {missing_terms} "
                                f"in segment citing [{marker}] are absent in the cited chunk '{expected_chunk_id}'."
                            )
                            logger.warning(f"[TOKEN GATE OVERRIDE] {reason} | Segment: '{segment_text}'")
                            mismatches.append(
                                CitationMismatch(
                                    segment=segment_text,
                                    citation_marker=f"[{marker}]",
                                    expected_chunk_id=expected_chunk_id,
                                    actual_supporting_chunks=normalized_actual_chunks,
                                    reason=reason
                                )
                            )
                            continue

                # Verify that the expected chunk ID is actually in the supporting chunks of the segment
                if expected_chunk_id not in normalized_actual_chunks:
                    reason = (
                        f"Citation mismatch: The claim cites [{marker}] ({expected_chunk_id}) "
                        f"but NLI verifier determined this segment is supported by {normalized_actual_chunks or 'no sources'}."
                    )
                    logger.warning(f"[CITATION ENFORCEMENT] {reason} | Segment: '{segment_text}'")
                    mismatches.append(
                        CitationMismatch(
                            segment=segment_text,
                            citation_marker=f"[{marker}]",
                            expected_chunk_id=expected_chunk_id,
                            actual_supporting_chunks=normalized_actual_chunks,
                            reason=reason
                        )
                    )
                    
        is_safe = len(mismatches) == 0
        return CitationVerificationReport(
            is_safe=is_safe,
            mismatches=mismatches,
            warnings=warnings
        )
