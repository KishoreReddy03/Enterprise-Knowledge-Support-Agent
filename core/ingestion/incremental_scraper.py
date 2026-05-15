"""
Incremental scraper that detects and syncs only new/updated items.

Instead of re-scraping all 15k+ docs daily, this:
1. Checks last sync time
2. Fetches only NEW releases/issues/answers since last sync
3. Updates existing items if modified
4. Skips unchanged data
"""

import logging
from datetime import datetime
from typing import Optional

import aiohttp

from core.ingestion.scrapers import StripeDataScraper, RawDocument
from core.ingestion.sync_tracker import tracker

logger = logging.getLogger(__name__)


class IncrementalScraper:
    """
    Wrapper around StripeDataScraper that implements incremental syncing.
    
    Tracks what's been scraped and only fetches new/updated items.
    """
    
    def __init__(self):
        """Initialize incremental scraper."""
        self.scraper = StripeDataScraper()
    
    async def sync_all_sources(self) -> dict:
        """
        Sync all sources incrementally based on sync interval.
        
        Returns:
            Dict with sync results per source
        """
        results = {
            "stripe_docs": None,
            "changelog": None,
            "github": None,
            "stackoverflow": None,
        }
        
        # Stripe docs - sync every 24 hours
        if tracker.should_sync("stripe_docs", min_interval_hours=24):
            logger.info("Syncing Stripe docs (OpenAPI spec)...")
            tracker.mark_sync_start("stripe_docs")
            docs = await self.scraper.scrape_stripe_docs()
            tracker.mark_sync_complete(
                "stripe_docs",
                items_count=len(docs),
                last_modified=datetime.utcnow()
            )
            results["stripe_docs"] = docs
        else:
            logger.info("Stripe docs recently synced, skipping")
        
        # Changelog - sync every 24 hours
        if tracker.should_sync("changelog", min_interval_hours=24):
            logger.info("Syncing Stripe changelog...")
            tracker.mark_sync_start("changelog")
            changelog = await self.scraper.scrape_stripe_changelog()
            tracker.mark_sync_complete(
                "changelog",
                items_count=len(changelog),
                last_modified=datetime.utcnow()
            )
            results["changelog"] = changelog
        else:
            logger.info("Changelog recently synced, skipping")
        
        # GitHub - sync every 12 hours (more frequently, as issues update often)
        if tracker.should_sync("github", min_interval_hours=12):
            logger.info("Syncing GitHub issues...")
            tracker.mark_sync_start("github")
            issues = await self._sync_github_incremental()
            results["github"] = issues
        else:
            logger.info("GitHub recently synced, skipping")
        
        # StackOverflow - sync every 24 hours
        if tracker.should_sync("stackoverflow", min_interval_hours=24):
            logger.info("Syncing StackOverflow Q&As...")
            tracker.mark_sync_start("stackoverflow")
            qa = await self.scraper.fetch_stackoverflow_questions(max_questions=500)
            tracker.mark_sync_complete(
                "stackoverflow",
                items_count=len(qa),
                last_modified=datetime.utcnow()
            )
            results["stackoverflow"] = qa
        else:
            logger.info("StackOverflow recently synced, skipping")
        
        # Print sync report
        print("\n" + tracker.get_sync_report())
        
        return results
    
    async def _sync_github_incremental(self) -> list[RawDocument]:
        """
        Sync GitHub issues, detecting only new/updated ones.
        
        Uses last_modified timestamp to fetch only recent issues.
        
        Returns:
            List of RawDocument objects with new/updated issues
        """
        repos = ["stripe/stripe-python", "stripe/stripe-node", "stripe/stripe-cli"]
        all_issues = []
        
        for repo in repos:
            last_modified = tracker.get_repo_last_modified(repo)
            
            if last_modified:
                logger.info(
                    f"Fetching {repo} issues updated after {last_modified.date()}"
                )
                # Fetch with filter for recently updated issues
                issues = await self._fetch_recent_issues(repo, last_modified)
            else:
                logger.info(f"First sync for {repo}, fetching all recent issues")
                # First time: fetch last 200 issues
                issues = await self.scraper.fetch_github_issues(
                    repos=[repo],
                    max_per_repo=200
                )
            
            # Track sync for this repo
            if issues:
                latest_timestamp = max(
                    datetime.fromisoformat(issue.date.replace('Z', '+00:00'))
                    for issue in issues
                    if issue.date
                )
                tracker.mark_repo_sync_complete(
                    repo,
                    items_count=len(issues),
                    last_modified=latest_timestamp
                )
            else:
                tracker.mark_repo_sync_complete(repo, items_count=0)
            
            all_issues.extend(issues)
        
        tracker.save()
        return all_issues
    
    async def _fetch_recent_issues(
        self, 
        repo: str, 
        since: datetime
    ) -> list[RawDocument]:
        """
        Fetch GitHub issues updated since a specific timestamp.
        
        Args:
            repo: Repository name
            since: Only fetch issues updated after this time
            
        Returns:
            List of RawDocument objects
        """
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "StripeDataScraper/1.0",
        }
        
        if self.scraper._github_token:
            headers["Authorization"] = f"token {self.scraper._github_token}"
        
        documents: list[RawDocument] = []
        page = 1
        since_str = since.isoformat()
        
        async with aiohttp.ClientSession(headers=headers) as session:
            while True:
                url = (
                    f"https://api.github.com/repos/{repo}/issues"
                    f"?state=closed&since={since_str}&sort=updated&order=desc"
                    f"&per_page=100&page={page}"
                )
                
                try:
                    async with session.get(url) as response:
                        if response.status != 200:
                            logger.error(f"GitHub API error: {response.status}")
                            break
                        
                        issues = await response.json()
                        if not issues:
                            break
                        
                        for issue in issues:
                            if "pull_request" in issue:
                                continue
                            
                            if issue.get("comments", 0) < 1:
                                continue
                            
                            comments = await self.scraper._fetch_issue_comments(
                                session, issue["comments_url"]
                            )
                            
                            doc = self.scraper._build_github_issue_document(
                                issue, comments, repo
                            )
                            if doc:
                                documents.append(doc)
                        
                        if len(issues) < 100:
                            break
                        
                        page += 1
                
                except Exception as e:
                    logger.error(f"Error fetching {repo} issues: {e}")
                    break
        
        logger.info(f"Fetched {len(documents)} updated issues from {repo}")
        return documents


# Global instance
incremental_scraper = IncrementalScraper()
