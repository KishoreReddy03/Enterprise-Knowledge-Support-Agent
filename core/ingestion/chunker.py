"""
Semantic chunking module for the ingestion pipeline.

This module handles intelligent splitting of documents from various sources
(Stripe docs, GitHub issues, StackOverflow, changelogs) into retrieval-optimized
chunks with deterministic IDs for deduplication.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from bs4 import BeautifulSoup

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a single chunk of text optimized for retrieval.
    
    Attributes:
        text: The actual text content of the chunk.
        source_type: Origin of the content ('stripe_doc', 'github_issue', 
                     'stackoverflow', 'changelog').
        source_url: URL where the original content can be found.
        title: Title of the parent document.
        section_path: Hierarchical path of headers (e.g., ['Webhooks', 'Best Practices']).
        date: Optional date associated with the content (ISO format).
        chunk_id: Deterministic SHA256 hash (first 16 chars) for deduplication.
        metadata: Flexible dictionary for additional source-specific fields.
    """
    text: str
    source_type: str
    source_url: str
    title: str
    section_path: list[str]
    date: str | None
    chunk_id: str
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """
    Handles intelligent chunking of documents from various sources.
    
    Uses source-specific strategies to create retrieval-optimized chunks
    while maintaining semantic coherence and respecting token limits.
    """

    # Token limits for chunk sizing
    MAX_TOKENS: int = 800
    MIN_TOKENS: int = 100

    # Resolution indicator words for GitHub issues
    RESOLUTION_KEYWORDS: set[str] = {
        "fix",
        "solution",
        "workaround",
        "resolved by",
        "try this",
    }
    # Patterns to skip in GitHub comments
    SKIP_PATTERNS: set[str] = {
        "me too",
        "+1",
        "same here",
        "same issue",
    }

    def __init__(self) -> None:
        """Initialize the chunker with tiktoken encoder."""
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken encoder: {e}")
            raise

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using cl100k_base encoding.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            Number of tokens in the text.
        """
        return len(self._encoder.encode(text))

    def _generate_chunk_id(self, text: str, source_url: str) -> str:
        """
        Generate deterministic chunk ID from content and source.
        
        Args:
            text: The chunk text content.
            source_url: The source URL.
            
        Returns:
            First 16 characters of SHA256 hash.
        """
        content = f"{text}{source_url}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _strip_html(self, text: str) -> str:
        """
        Remove HTML tags from text using BeautifulSoup.
        
        Args:
            text: Text potentially containing HTML.
            
        Returns:
            Clean text with HTML tags removed.
        """
        if not text:
            return ""
        try:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ")
        except Exception as e:
            logger.warning(f"HTML stripping failed, returning original text: {e}")
            return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Text with potentially irregular whitespace.
            
        Returns:
            Text with normalized whitespace.
        """
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        return text.strip()

    def _clean_text(self, text: str) -> str:
        """
        Apply all text cleaning operations.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text with HTML stripped and whitespace normalized.
        """
        text = self._strip_html(text)
        text = self._normalize_whitespace(text)
        return text

    def _split_on_paragraphs(self, text: str, max_tokens: int) -> list[str]:
        """
        Split text on paragraph boundaries to fit within token limit.
        
        Args:
            text: Text to split.
            max_tokens: Maximum tokens per resulting chunk.
            
        Returns:
            List of text segments, each within token limit.
        """
        paragraphs = re.split(r"\n\n+", text)
        segments: list[str] = []
        current_segment: list[str] = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_tokens = self._count_tokens(paragraph)

            # If single paragraph exceeds max, split on sentences
            if para_tokens > max_tokens:
                if current_segment:
                    segments.append("\n\n".join(current_segment))
                    current_segment = []
                    current_tokens = 0

                # Split large paragraph on sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    sentence_tokens = self._count_tokens(sentence)
                    if current_tokens + sentence_tokens > max_tokens and current_segment:
                        segments.append(" ".join(current_segment))
                        current_segment = []
                        current_tokens = 0
                    current_segment.append(sentence)
                    current_tokens += sentence_tokens
            elif current_tokens + para_tokens > max_tokens:
                # Start new segment
                if current_segment:
                    segments.append("\n\n".join(current_segment))
                current_segment = [paragraph]
                current_tokens = para_tokens
            else:
                current_segment.append(paragraph)
                current_tokens += para_tokens

        # Don't forget the last segment
        if current_segment:
            segments.append("\n\n".join(current_segment))

        return segments

    def _parse_markdown_sections(
        self, text: str
    ) -> list[tuple[list[str], str]]:
        """
        Parse markdown text into sections based on headers.
        
        Args:
            text: Markdown text to parse.
            
        Returns:
            List of tuples: (section_path, section_content).
        """
        # Match markdown headers (## or ###)
        header_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
        
        sections: list[tuple[list[str], str]] = []
        header_stack: list[tuple[int, str]] = []  # (level, title)
        
        # Split on headers while keeping the headers
        parts = header_pattern.split(text)
        
        # parts format: [pre-content, #-marks, title, content, #-marks, title, content, ...]
        if not parts:
            return [([], text)]

        # Handle content before first header
        pre_header_content = parts[0].strip()
        if pre_header_content:
            sections.append(([], pre_header_content))

        # Process header-content pairs
        i = 1
        while i < len(parts) - 2:
            header_marks = parts[i]
            header_title = parts[i + 1].strip()
            content = parts[i + 2].strip() if i + 2 < len(parts) else ""

            level = len(header_marks)  # 2 for ##, 3 for ###

            # Update header stack
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, header_title))

            # Build section path from stack
            section_path = [h[1] for h in header_stack]

            if content:
                sections.append((section_path, content))

            i += 3

        return sections

    def _is_skip_comment(self, comment_body: str) -> bool:
        """
        Check if a GitHub comment should be skipped.
        
        Args:
            comment_body: The comment text.
            
        Returns:
            True if the comment matches skip patterns.
        """
        lower_body = comment_body.lower().strip()
        
        # Check for exact matches or very short "me too" style comments
        if len(lower_body) < 20:
            for pattern in self.SKIP_PATTERNS:
                if pattern in lower_body:
                    return True
        return False

    def _is_resolution_comment(self, comment_body: str) -> bool:
        """
        Detect if a GitHub comment contains a resolution.
        
        Args:
            comment_body: The comment text.
            
        Returns:
            True if the comment appears to contain a resolution.
        """
        lower_body = comment_body.lower()

        # Check for code blocks
        if "```" in comment_body or re.search(r"`[^`]+`", comment_body):
            return True

        # Check for resolution keywords
        for keyword in self.RESOLUTION_KEYWORDS:
            if keyword in lower_body:
                return True

        return False

    def chunk(
        self,
        text: str,
        source_type: str,
        source_url: str,
        title: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """
        Chunk text from stripe_doc or changelog sources.
        
        Args:
            text: The raw text content to chunk.
            source_type: Must be 'stripe_doc' or 'changelog'.
            source_url: URL of the source document.
            title: Title of the document.
            metadata: Optional additional metadata.
            
        Returns:
            List of Chunk objects optimized for retrieval.
            
        Raises:
            ValueError: If source_type is not supported by this method.
        """
        if source_type not in ("stripe_doc", "changelog"):
            raise ValueError(
                f"chunk() only handles 'stripe_doc' and 'changelog'. "
                f"Got '{source_type}'. Use chunk_document() for automatic dispatch."
            )

        if metadata is None:
            metadata = {}

        text = self._clean_text(text)
        chunks: list[Chunk] = []

        if source_type == "stripe_doc":
            chunks = self._chunk_stripe_doc(text, source_url, title, metadata)
        elif source_type == "changelog":
            chunks = self._chunk_changelog(text, source_url, title, metadata)

        logger.info(
            f"Chunked {source_type} document '{title}' into {len(chunks)} chunks"
        )
        return chunks

    def _chunk_stripe_doc(
        self,
        text: str,
        source_url: str,
        title: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """
        Chunk Stripe documentation using markdown header structure.
        
        Args:
            text: Cleaned document text.
            source_url: Document URL.
            title: Document title.
            metadata: Additional metadata.
            
        Returns:
            List of Chunk objects.
        """
        sections = self._parse_markdown_sections(text)
        chunks: list[Chunk] = []
        pending_merge: tuple[list[str], str] | None = None

        for section_path, content in sections:
            # Include title in section path if not already present
            full_path = [title] + section_path if section_path else [title]
            
            # Clean the content
            content = self._clean_text(content)
            if not content:
                continue

            token_count = self._count_tokens(content)

            # Handle pending merge from previous small section
            if pending_merge is not None:
                merged_content = f"{pending_merge[1]}\n\n{content}"
                merged_tokens = self._count_tokens(merged_content)

                if merged_tokens <= self.MAX_TOKENS:
                    # Use the current section's path for merged content
                    content = merged_content
                    token_count = merged_tokens
                    pending_merge = None
                else:
                    # Previous section stays too small, create chunk anyway
                    prev_path, prev_content = pending_merge
                    chunks.append(
                        Chunk(
                            text=prev_content,
                            source_type="stripe_doc",
                            source_url=source_url,
                            title=title,
                            section_path=prev_path,
                            date=metadata.get("date"),
                            chunk_id=self._generate_chunk_id(prev_content, source_url),
                            metadata=metadata.copy(),
                        )
                    )
                    pending_merge = None

            # Section too small - queue for merge
            if token_count < self.MIN_TOKENS:
                pending_merge = (full_path, content)
                continue

            # Section too large - split on paragraphs
            if token_count > self.MAX_TOKENS:
                segments = self._split_on_paragraphs(content, self.MAX_TOKENS)
                for i, segment in enumerate(segments):
                    segment_path = full_path.copy()
                    if len(segments) > 1:
                        segment_path.append(f"Part {i + 1}")
                    
                    chunks.append(
                        Chunk(
                            text=segment,
                            source_type="stripe_doc",
                            source_url=source_url,
                            title=title,
                            section_path=segment_path,
                            date=metadata.get("date"),
                            chunk_id=self._generate_chunk_id(segment, source_url),
                            metadata=metadata.copy(),
                        )
                    )
            else:
                # Section is within bounds
                chunks.append(
                    Chunk(
                        text=content,
                        source_type="stripe_doc",
                        source_url=source_url,
                        title=title,
                        section_path=full_path,
                        date=metadata.get("date"),
                        chunk_id=self._generate_chunk_id(content, source_url),
                        metadata=metadata.copy(),
                    )
                )

        # Handle any remaining pending merge
        if pending_merge is not None:
            prev_path, prev_content = pending_merge
            chunks.append(
                Chunk(
                    text=prev_content,
                    source_type="stripe_doc",
                    source_url=source_url,
                    title=title,
                    section_path=prev_path,
                    date=metadata.get("date"),
                    chunk_id=self._generate_chunk_id(prev_content, source_url),
                    metadata=metadata.copy(),
                )
            )

        return chunks

    def _chunk_changelog(
        self,
        text: str,
        source_url: str,
        title: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """
        Chunk changelog entries, keeping each entry intact.
        
        Args:
            text: Changelog text (entries separated by date headers or blank lines).
            source_url: Changelog URL.
            title: Changelog title.
            metadata: Additional metadata.
            
        Returns:
            List of Chunk objects, one per changelog entry.
        """
        # Split on date-like patterns or horizontal rules
        # Common patterns: "## 2024-01-15", "### January 15, 2024", "---"
        entry_pattern = re.compile(
            r"(?:^|\n)(?=#{1,3}\s*\d{4}[-/]?\d{0,2}[-/]?\d{0,2}|"
            r"#{1,3}\s*(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)|"
            r"---+\n)",
            re.IGNORECASE,
        )

        entries = entry_pattern.split(text)
        chunks: list[Chunk] = []

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            # Try to extract date from entry
            date_match = re.search(
                r"(\d{4}[-/]\d{2}[-/]\d{2}|"
                r"(?:January|February|March|April|May|June|July|"
                r"August|September|October|November|December)\s+\d{1,2},?\s*\d{4})",
                entry,
                re.IGNORECASE,
            )
            entry_date = date_match.group(1) if date_match else metadata.get("date")

            # Never split a single changelog entry
            chunk_metadata = metadata.copy()
            chunk_metadata["entry_date"] = entry_date

            chunks.append(
                Chunk(
                    text=entry,
                    source_type="changelog",
                    source_url=source_url,
                    title=title,
                    section_path=[title],
                    date=entry_date,
                    chunk_id=self._generate_chunk_id(entry, source_url),
                    metadata=chunk_metadata,
                )
            )

        return chunks

    def chunk_github_issue(self, issue_data: dict[str, Any]) -> list[Chunk]:
        """
        Chunk a GitHub issue with its comments.
        
        Args:
            issue_data: Dictionary with structure:
                {
                    "title": str,
                    "body": str,
                    "url": str,
                    "labels": list[str],
                    "comments": [{"body": str, "is_resolution": bool}]
                }
                
        Returns:
            List of Chunk objects:
            - One chunk for title + body
            - Separate chunks for resolution comments
        """
        title = issue_data.get("title", "")
        body = issue_data.get("body", "")
        url = issue_data.get("url", "")
        labels = issue_data.get("labels", [])
        comments = issue_data.get("comments", [])

        chunks: list[Chunk] = []

        # Chunk 1: Issue title + body
        issue_text = self._clean_text(f"{title}\n\n{body}")
        if issue_text.strip():
            chunks.append(
                Chunk(
                    text=issue_text,
                    source_type="github_issue",
                    source_url=url,
                    title=title,
                    section_path=[title, "Issue"],
                    date=None,
                    chunk_id=self._generate_chunk_id(issue_text, url),
                    metadata={
                        "labels": labels,
                        "is_resolution": False,
                    },
                )
            )

        # Process comments
        for i, comment in enumerate(comments):
            comment_body = comment.get("body", "")
            is_resolution = comment.get("is_resolution", False)

            # Clean comment
            comment_body = self._clean_text(comment_body)
            if not comment_body:
                continue

            # Skip "me too" style comments
            if self._is_skip_comment(comment_body):
                logger.debug(f"Skipping low-value comment {i} in issue {url}")
                continue

            # Auto-detect resolution if not explicitly marked
            if not is_resolution:
                is_resolution = self._is_resolution_comment(comment_body)

            # Only create chunks for resolution comments
            if is_resolution:
                chunks.append(
                    Chunk(
                        text=comment_body,
                        source_type="github_issue",
                        source_url=url,
                        title=title,
                        section_path=[title, f"Resolution Comment {i + 1}"],
                        date=None,
                        chunk_id=self._generate_chunk_id(comment_body, url),
                        metadata={
                            "labels": labels,
                            "is_resolution": True,
                            "comment_index": i,
                        },
                    )
                )

        logger.info(
            f"Chunked GitHub issue '{title}' into {len(chunks)} chunks "
            f"({sum(1 for c in chunks if c.metadata.get('is_resolution'))} resolutions)"
        )
        return chunks

    def chunk_stackoverflow(self, question_data: dict[str, Any]) -> list[Chunk]:
        """
        Chunk a StackOverflow question with its answers.
        
        Args:
            question_data: Dictionary with structure:
                {
                    "title": str,
                    "body": str,
                    "url": str,
                    "score": int,
                    "accepted_answer": {"body": str, "score": int} | None,
                    "other_answers": [{"body": str, "score": int}]
                }
                
        Returns:
            List of Chunk objects:
            - One chunk for question title + body
            - One chunk for accepted answer (if exists)
            - Chunks for other answers with score > 5
        """
        title = question_data.get("title", "")
        body = question_data.get("body", "")
        url = question_data.get("url", "")
        question_score = question_data.get("score", 0)
        accepted_answer = question_data.get("accepted_answer")
        other_answers = question_data.get("other_answers", [])

        chunks: list[Chunk] = []

        # Chunk 1: Question title + body
        question_text = self._clean_text(f"{title}\n\n{body}")
        if question_text.strip():
            chunks.append(
                Chunk(
                    text=question_text,
                    source_type="stackoverflow",
                    source_url=url,
                    title=title,
                    section_path=[title, "Question"],
                    date=None,
                    chunk_id=self._generate_chunk_id(question_text, url),
                    metadata={
                        "question_score": question_score,
                        "is_question": True,
                        "is_answer": False,
                        "is_accepted": False,
                    },
                )
            )

        # Chunk 2: Accepted answer (if exists)
        if accepted_answer:
            answer_body = self._clean_text(accepted_answer.get("body", ""))
            answer_score = accepted_answer.get("score", 0)

            if answer_body:
                chunks.append(
                    Chunk(
                        text=answer_body,
                        source_type="stackoverflow",
                        source_url=url,
                        title=title,
                        section_path=[title, "Accepted Answer"],
                        date=None,
                        chunk_id=self._generate_chunk_id(answer_body, url),
                        metadata={
                            "answer_score": answer_score,
                            "is_question": False,
                            "is_answer": True,
                            "is_accepted": True,
                        },
                    )
                )

        # Other answers with score > 5
        for i, answer in enumerate(other_answers):
            answer_score = answer.get("score", 0)

            # Skip low-scored answers
            if answer_score <= 0:
                continue

            # Only include answers with score > 5
            if answer_score <= 5:
                continue

            answer_body = self._clean_text(answer.get("body", ""))
            if not answer_body:
                continue

            chunks.append(
                Chunk(
                    text=answer_body,
                    source_type="stackoverflow",
                    source_url=url,
                    title=title,
                    section_path=[title, f"Answer {i + 1} (Score: {answer_score})"],
                    date=None,
                    chunk_id=self._generate_chunk_id(answer_body, url),
                    metadata={
                        "answer_score": answer_score,
                        "is_question": False,
                        "is_answer": True,
                        "is_accepted": False,
                    },
                )
            )

        logger.info(
            f"Chunked StackOverflow question '{title}' into {len(chunks)} chunks"
        )
        return chunks

    def chunk_document(
        self,
        source_type: str,
        data: dict[str, Any] | str,
        **kwargs: Any,
    ) -> list[Chunk]:
        """
        Unified entry point for chunking any document type.
        
        Automatically dispatches to the appropriate chunking method based on source_type.
        
        Args:
            source_type: Type of source ('stripe_doc', 'changelog', 
                        'github_issue', 'stackoverflow').
            data: Either a string (for stripe_doc/changelog) or a dict
                  (for github_issue/stackoverflow).
            **kwargs: Additional arguments passed to chunk() for text sources.
                      Required for text sources: source_url, title.
                      Optional: metadata.
                      
        Returns:
            List of Chunk objects.
            
        Raises:
            ValueError: If source_type is unknown or required args are missing.
        """
        if source_type == "github_issue":
            if not isinstance(data, dict):
                raise ValueError(
                    "github_issue requires dict data with title, body, url, comments"
                )
            return self.chunk_github_issue(data)

        elif source_type == "stackoverflow":
            if not isinstance(data, dict):
                raise ValueError(
                    "stackoverflow requires dict data with title, body, url, answers"
                )
            return self.chunk_stackoverflow(data)

        elif source_type in ("stripe_doc", "changelog"):
            if not isinstance(data, str):
                raise ValueError(
                    f"{source_type} requires string data (the document text)"
                )
            
            # Validate required kwargs
            if "source_url" not in kwargs:
                raise ValueError(f"source_url is required for {source_type}")
            if "title" not in kwargs:
                raise ValueError(f"title is required for {source_type}")
            
            return self.chunk(
                text=data,
                source_type=source_type,
                source_url=kwargs["source_url"],
                title=kwargs["title"],
                metadata=kwargs.get("metadata"),
            )

        else:
            raise ValueError(
                f"Unknown source_type: '{source_type}'. "
                f"Supported types: stripe_doc, changelog, github_issue, stackoverflow"
            )


# Module-level instance for convenience
chunker = SemanticChunker()
