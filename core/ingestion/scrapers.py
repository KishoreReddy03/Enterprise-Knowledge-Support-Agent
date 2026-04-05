"""
Web scraping and API fetching module for data ingestion.

Handles scraping Stripe documentation, changelog, GitHub issues,
and StackOverflow questions using Crawl4AI and REST APIs.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import NoExtractionStrategy

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """
    Represents a raw document fetched from an external source.
    
    This is the intermediate format between scraping and chunking.
    Contains the full document content before being split into chunks.
    
    Attributes:
        content: The raw text/markdown content of the document.
        url: Source URL of the document.
        title: Document title.
        source_type: Type of source ('stripe_doc', 'github_issue', 
                     'stackoverflow', 'changelog').
        date: Optional date associated with the document (ISO format).
        metadata: Flexible dictionary for source-specific fields.
    """
    content: str
    url: str
    title: str
    source_type: str
    date: str | None
    metadata: dict = field(default_factory=dict)


class StripeDataScraper:
    """
    Scraper for Stripe documentation and related data sources.
    
    Handles web scraping of Stripe docs and changelog using Crawl4AI,
    and fetches structured data from GitHub and StackOverflow APIs.
    """

    # Default sections to scrape from Stripe docs
    DEFAULT_DOC_SECTIONS: list[str] = [
        "webhooks",
        "errors",
        "payments",
        "billing",
        "connect",
        "radar",
    ]

    # Default GitHub repos to fetch issues from
    DEFAULT_GITHUB_REPOS: list[str] = [
        "stripe/stripe-python",
        "stripe/stripe-node",
        "stripe/stripe-cli",
    ]

    # Default StackOverflow tags
    DEFAULT_SO_TAGS: list[str] = [
        "stripe-payments",
        "stripe-api",
        "stripe-connect",
    ]

    # Rate limiting constants
    STRIPE_DELAY_SECONDS: float = 2.0
    STRIPE_TIMEOUT_SECONDS: int = 30
    SO_MAX_REQUESTS_PER_SECOND: int = 30

    def __init__(self) -> None:
        """Initialize the scraper."""
        self._github_token: str | None = getattr(settings, "GITHUB_TOKEN", None)

    async def scrape_stripe_docs(
        self,
        sections: list[str] | None = None,
    ) -> list[RawDocument]:
        """
        Scrape Stripe documentation sections.
        
        Args:
            sections: List of doc sections to scrape. Defaults to common sections
                      like webhooks, errors, payments, billing, connect, radar.
                      
        Returns:
            List of RawDocument objects, one per successfully scraped page.
        """
        if sections is None:
            sections = self.DEFAULT_DOC_SECTIONS

        documents: list[RawDocument] = []

        async with AsyncWebCrawler() as crawler:
            for section in sections:
                url = f"https://stripe.com/docs/{section}"
                logger.info(f"Scraping Stripe docs: {url}")

                try:
                    result = await asyncio.wait_for(
                        crawler.arun(
                            url=url,
                            word_count_threshold=50,
                            extraction_strategy=NoExtractionStrategy(),
                            bypass_cache=True,
                        ),
                        timeout=self.STRIPE_TIMEOUT_SECONDS,
                    )

                    if result.success and result.markdown:
                        # Extract title from markdown or use section name
                        title = self._extract_title_from_markdown(result.markdown)
                        if not title:
                            title = section.replace("-", " ").title()

                        documents.append(
                            RawDocument(
                                content=result.markdown,
                                url=url,
                                title=title,
                                source_type="stripe_doc",
                                date=None,
                                metadata={
                                    "section": section,
                                    "scraped_at": datetime.utcnow().isoformat(),
                                },
                            )
                        )
                        logger.info(f"Successfully scraped: {url}")
                    else:
                        logger.warning(
                            f"Failed to scrape {url}: "
                            f"{result.error_message if hasattr(result, 'error_message') else 'No content'}"
                        )

                except asyncio.TimeoutError:
                    logger.error(f"Timeout scraping {url}")
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")

                # Rate limiting
                await asyncio.sleep(self.STRIPE_DELAY_SECONDS)

        logger.info(f"Scraped {len(documents)} Stripe doc pages")
        return documents

    async def scrape_stripe_changelog(
        self,
        max_entries: int = 200,
    ) -> list[RawDocument]:
        """
        Scrape Stripe changelog entries.
        
        Args:
            max_entries: Maximum number of changelog entries to return.
            
        Returns:
            List of RawDocument objects, one per changelog entry.
        """
        url = "https://stripe.com/docs/changelog"
        documents: list[RawDocument] = []

        logger.info(f"Scraping Stripe changelog: {url}")

        async with AsyncWebCrawler() as crawler:
            try:
                result = await asyncio.wait_for(
                    crawler.arun(
                        url=url,
                        word_count_threshold=20,
                        extraction_strategy=NoExtractionStrategy(),
                        bypass_cache=True,
                    ),
                    timeout=self.STRIPE_TIMEOUT_SECONDS * 2,  # Changelog may be larger
                )

                if result.success and result.markdown:
                    entries = self._parse_changelog_entries(result.markdown)

                    for entry in entries[:max_entries]:
                        documents.append(
                            RawDocument(
                                content=entry["content"],
                                url=url,
                                title=entry["title"],
                                source_type="changelog",
                                date=entry["date"],
                                metadata={
                                    "scraped_at": datetime.utcnow().isoformat(),
                                },
                            )
                        )

                    logger.info(f"Parsed {len(documents)} changelog entries")
                else:
                    logger.warning(f"Failed to scrape changelog: no content")

            except asyncio.TimeoutError:
                logger.error(f"Timeout scraping changelog")
            except Exception as e:
                logger.error(f"Error scraping changelog: {e}")

        return documents

    def _extract_title_from_markdown(self, markdown: str) -> str | None:
        """
        Extract title from first H1 header in markdown.
        
        Args:
            markdown: Markdown content.
            
        Returns:
            Title string or None if not found.
        """
        match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    def _parse_changelog_entries(
        self,
        markdown: str,
    ) -> list[dict[str, str | None]]:
        """
        Parse changelog markdown into individual entries.
        
        Args:
            markdown: Full changelog markdown content.
            
        Returns:
            List of dicts with 'title', 'content', and 'date' keys.
        """
        entries: list[dict[str, str | None]] = []

        # Split on date headers (common formats: ## 2024-01-15, ## January 15, 2024)
        date_pattern = re.compile(
            r"^##\s*(\d{4}[-/]\d{2}[-/]\d{2}|"
            r"(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+\d{1,2},?\s*\d{4})",
            re.MULTILINE | re.IGNORECASE,
        )

        parts = date_pattern.split(markdown)

        # parts: [pre-content, date1, content1, date2, content2, ...]
        i = 1
        while i < len(parts) - 1:
            date_str = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""

            if content:
                # Use first line as title, or generate from date
                lines = content.split("\n", 1)
                title = lines[0].strip("#").strip() if lines else f"Changelog {date_str}"

                entries.append({
                    "title": title,
                    "content": f"## {date_str}\n\n{content}",
                    "date": date_str,
                })

            i += 2

        return entries

    async def fetch_github_issues(
        self,
        repos: list[str] | None = None,
        max_per_repo: int = 500,
        state: str = "closed",
    ) -> list[RawDocument]:
        """
        Fetch issues from GitHub repositories.
        
        Args:
            repos: List of repos in 'owner/name' format. Defaults to Stripe SDKs.
            max_per_repo: Maximum issues to fetch per repository.
            state: Issue state filter ('open', 'closed', 'all').
            
        Returns:
            List of RawDocument objects with issue data in metadata.
        """
        if repos is None:
            repos = self.DEFAULT_GITHUB_REPOS

        documents: list[RawDocument] = []
        headers: dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "StripeDataScraper/1.0",
        }

        if self._github_token:
            headers["Authorization"] = f"token {self._github_token}"
            logger.info("Using authenticated GitHub API access")
        else:
            logger.warning(
                "No GITHUB_TOKEN configured. Using unauthenticated access (60 req/hour limit)"
            )

        async with aiohttp.ClientSession(headers=headers) as session:
            for repo in repos:
                logger.info(f"Fetching issues from {repo}")
                repo_docs = await self._fetch_repo_issues(
                    session, repo, max_per_repo, state
                )
                documents.extend(repo_docs)

        logger.info(f"Fetched {len(documents)} GitHub issues total")
        return documents

    async def _fetch_repo_issues(
        self,
        session: aiohttp.ClientSession,
        repo: str,
        max_issues: int,
        state: str,
    ) -> list[RawDocument]:
        """
        Fetch issues from a single GitHub repository.
        
        Args:
            session: aiohttp session with headers configured.
            repo: Repository in 'owner/name' format.
            max_issues: Maximum issues to fetch.
            state: Issue state filter.
            
        Returns:
            List of RawDocument objects.
        """
        documents: list[RawDocument] = []
        page = 1
        per_page = 100
        fetched = 0

        while fetched < max_issues:
            url = (
                f"https://api.github.com/repos/{repo}/issues"
                f"?state={state}&per_page={per_page}&page={page}"
            )

            try:
                async with session.get(url) as response:
                    # Check rate limit
                    remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
                    if remaining < 10:
                        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                        wait_seconds = max(0, reset_time - int(datetime.utcnow().timestamp()))
                        if wait_seconds > 0:
                            logger.warning(
                                f"GitHub rate limit low ({remaining}). Waiting {wait_seconds}s"
                            )
                            await asyncio.sleep(wait_seconds + 1)

                    if response.status != 200:
                        logger.error(
                            f"GitHub API error for {repo}: {response.status}"
                        )
                        break

                    issues = await response.json()

                    if not issues:
                        break

                    for issue in issues:
                        # Skip pull requests (they appear in issues API)
                        if "pull_request" in issue:
                            continue

                        # Filter: only issues with at least 1 comment
                        if issue.get("comments", 0) < 1:
                            continue

                        # Fetch comments
                        comments = await self._fetch_issue_comments(
                            session, issue["comments_url"]
                        )

                        # Build document
                        doc = self._build_github_issue_document(issue, comments, repo)
                        if doc:
                            documents.append(doc)
                            fetched += 1

                            if fetched >= max_issues:
                                break

                    page += 1

            except aiohttp.ClientError as e:
                logger.error(f"Network error fetching {repo} issues: {e}")
                break
            except Exception as e:
                logger.error(f"Error fetching {repo} issues: {e}")
                break

        return documents

    async def _fetch_issue_comments(
        self,
        session: aiohttp.ClientSession,
        comments_url: str,
    ) -> list[dict[str, Any]]:
        """
        Fetch all comments for a GitHub issue.
        
        Args:
            session: aiohttp session.
            comments_url: GitHub API URL for issue comments.
            
        Returns:
            List of comment dictionaries.
        """
        try:
            async with session.get(comments_url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch comments: {e}")
        return []

    def _build_github_issue_document(
        self,
        issue: dict[str, Any],
        comments: list[dict[str, Any]],
        repo: str,
    ) -> RawDocument | None:
        """
        Build a RawDocument from GitHub issue data.
        
        Args:
            issue: Issue data from GitHub API.
            comments: List of comment data.
            repo: Repository name.
            
        Returns:
            RawDocument or None if issue is invalid.
        """
        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        url = issue.get("html_url", "")
        labels = [label.get("name", "") for label in issue.get("labels", [])]

        if not title:
            return None

        # Process comments
        processed_comments: list[dict[str, Any]] = []
        for comment in comments:
            comment_body = comment.get("body", "") or ""
            if not comment_body:
                continue

            # Detect resolution comments
            is_resolution = self._is_resolution_comment(comment_body)
            processed_comments.append({
                "body": comment_body,
                "is_resolution": is_resolution,
            })

        # Store structured data in metadata for chunker
        return RawDocument(
            content=f"{title}\n\n{body}",  # Basic content for display
            url=url,
            title=title,
            source_type="github_issue",
            date=issue.get("created_at"),
            metadata={
                "repo": repo,
                "labels": labels,
                "issue_number": issue.get("number"),
                "state": issue.get("state"),
                "comments": processed_comments,
                "body": body,  # For chunker
            },
        )

    def _is_resolution_comment(self, comment_body: str) -> bool:
        """
        Detect if a GitHub comment contains a resolution.
        
        Args:
            comment_body: Comment text.
            
        Returns:
            True if comment appears to contain a resolution.
        """
        lower_body = comment_body.lower()

        # Check for code blocks
        if "```" in comment_body or re.search(r"`[^`]+`", comment_body):
            return True

        # Check for resolution keywords
        resolution_keywords = ["fix", "solution", "workaround", "resolved by", "try this"]
        for keyword in resolution_keywords:
            if keyword in lower_body:
                return True

        return False

    async def fetch_stackoverflow_questions(
        self,
        tags: list[str] | None = None,
        max_questions: int = 1000,
    ) -> list[RawDocument]:
        """
        Fetch StackOverflow questions with accepted answers.
        
        Args:
            tags: List of SO tags to filter by. Defaults to Stripe-related tags.
            max_questions: Maximum questions to fetch.
            
        Returns:
            List of RawDocument objects with Q&A data in metadata.
        """
        if tags is None:
            tags = self.DEFAULT_SO_TAGS

        documents: list[RawDocument] = []
        page = 1
        page_size = 100
        fetched = 0

        # Rate limiting: track requests
        request_times: list[float] = []

        async with aiohttp.ClientSession() as session:
            while fetched < max_questions:
                # Rate limit: max 30 requests per second
                await self._so_rate_limit(request_times)

                params = {
                    "tagged": ";".join(tags),
                    "site": "stackoverflow",
                    "filter": "withbody",
                    "sort": "votes",
                    "order": "desc",
                    "pagesize": page_size,
                    "page": page,
                }

                url = "https://api.stackexchange.com/2.3/questions"

                try:
                    async with session.get(url, params=params) as response:
                        request_times.append(asyncio.get_event_loop().time())

                        if response.status != 200:
                            logger.error(f"StackOverflow API error: {response.status}")
                            break

                        data = await response.json()

                        if "items" not in data or not data["items"]:
                            break

                        for question in data["items"]:
                            # Only process questions with accepted answers
                            if not question.get("is_answered") or not question.get("accepted_answer_id"):
                                continue

                            # Fetch accepted answer
                            answer = await self._fetch_so_answer(
                                session, question["accepted_answer_id"], request_times
                            )

                            # Fetch other high-scored answers
                            other_answers = await self._fetch_so_other_answers(
                                session, question["question_id"], 
                                question["accepted_answer_id"], request_times
                            )

                            doc = self._build_so_document(question, answer, other_answers)
                            if doc:
                                documents.append(doc)
                                fetched += 1

                                if fetched >= max_questions:
                                    break

                        # Check if more pages available
                        if not data.get("has_more", False):
                            break

                        page += 1

                except aiohttp.ClientError as e:
                    logger.error(f"Network error fetching StackOverflow: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error fetching StackOverflow: {e}")
                    break

        logger.info(f"Fetched {len(documents)} StackOverflow questions")
        return documents

    async def _so_rate_limit(self, request_times: list[float]) -> None:
        """
        Enforce StackOverflow rate limit (30 req/s).
        
        Args:
            request_times: List of timestamps of recent requests.
        """
        now = asyncio.get_event_loop().time()

        # Remove requests older than 1 second
        request_times[:] = [t for t in request_times if now - t < 1.0]

        if len(request_times) >= self.SO_MAX_REQUESTS_PER_SECOND:
            # Wait until oldest request is > 1 second old
            oldest = min(request_times)
            wait_time = 1.0 - (now - oldest) + 0.05
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    async def _fetch_so_answer(
        self,
        session: aiohttp.ClientSession,
        answer_id: int,
        request_times: list[float],
    ) -> dict[str, Any] | None:
        """
        Fetch a single StackOverflow answer by ID.
        
        Args:
            session: aiohttp session.
            answer_id: StackOverflow answer ID.
            request_times: List for rate limiting.
            
        Returns:
            Answer data dict or None.
        """
        await self._so_rate_limit(request_times)

        url = f"https://api.stackexchange.com/2.3/answers/{answer_id}"
        params = {
            "site": "stackoverflow",
            "filter": "withbody",
        }

        try:
            async with session.get(url, params=params) as response:
                request_times.append(asyncio.get_event_loop().time())

                if response.status == 200:
                    data = await response.json()
                    if data.get("items"):
                        return data["items"][0]
        except Exception as e:
            logger.warning(f"Failed to fetch answer {answer_id}: {e}")

        return None

    async def _fetch_so_other_answers(
        self,
        session: aiohttp.ClientSession,
        question_id: int,
        accepted_answer_id: int,
        request_times: list[float],
    ) -> list[dict[str, Any]]:
        """
        Fetch other answers for a question (excluding accepted).
        
        Args:
            session: aiohttp session.
            question_id: StackOverflow question ID.
            accepted_answer_id: ID of accepted answer to exclude.
            request_times: List for rate limiting.
            
        Returns:
            List of answer data dicts with score > 5.
        """
        await self._so_rate_limit(request_times)

        url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
        params = {
            "site": "stackoverflow",
            "filter": "withbody",
            "sort": "votes",
            "order": "desc",
        }

        answers: list[dict[str, Any]] = []

        try:
            async with session.get(url, params=params) as response:
                request_times.append(asyncio.get_event_loop().time())

                if response.status == 200:
                    data = await response.json()
                    for answer in data.get("items", []):
                        # Skip accepted answer and low-scored answers
                        if answer.get("answer_id") == accepted_answer_id:
                            continue
                        if answer.get("score", 0) <= 5:
                            continue
                        answers.append(answer)

        except Exception as e:
            logger.warning(f"Failed to fetch other answers for Q{question_id}: {e}")

        return answers

    def _build_so_document(
        self,
        question: dict[str, Any],
        accepted_answer: dict[str, Any] | None,
        other_answers: list[dict[str, Any]],
    ) -> RawDocument | None:
        """
        Build a RawDocument from StackOverflow Q&A data.
        
        Args:
            question: Question data from SO API.
            accepted_answer: Accepted answer data.
            other_answers: List of other high-scored answers.
            
        Returns:
            RawDocument or None if data is invalid.
        """
        title = question.get("title", "")
        body = self._strip_html(question.get("body", ""))
        url = question.get("link", "")
        score = question.get("score", 0)

        if not title or not body:
            return None

        # Build accepted answer dict
        accepted_answer_data: dict[str, Any] | None = None
        if accepted_answer:
            accepted_answer_data = {
                "body": self._strip_html(accepted_answer.get("body", "")),
                "score": accepted_answer.get("score", 0),
            }

        # Build other answers list
        other_answers_data: list[dict[str, Any]] = []
        for answer in other_answers:
            other_answers_data.append({
                "body": self._strip_html(answer.get("body", "")),
                "score": answer.get("score", 0),
            })

        # Store structured data for chunker
        return RawDocument(
            content=f"{title}\n\n{body}",
            url=url,
            title=title,
            source_type="stackoverflow",
            date=None,
            metadata={
                "question_id": question.get("question_id"),
                "score": score,
                "tags": question.get("tags", []),
                "body": body,
                "accepted_answer": accepted_answer_data,
                "other_answers": other_answers_data,
            },
        )

    def _strip_html(self, text: str) -> str:
        """
        Strip HTML tags from text.
        
        Args:
            text: Text potentially containing HTML.
            
        Returns:
            Plain text with HTML removed.
        """
        if not text:
            return ""
        try:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ").strip()
        except Exception:
            return text


# Module-level instance for convenience
scraper = StripeDataScraper()
