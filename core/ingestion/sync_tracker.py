"""
Sync tracking module for incremental data ingestion.

Tracks last sync times and URLs to detect new/updated items across sources.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SyncTracker:
    """
    Tracks sync status per source to enable incremental scraping.
    
    Instead of re-scraping everything daily, this tracks:
    - Last successful sync time per source
    - Last modified timestamps for items
    - Already-scraped URLs/IDs to avoid duplicates
    """
    
    SYNC_FILE = Path(".") / "data" / "sync_status.json"
    
    def __init__(self):
        """Initialize sync tracker, creating file if needed."""
        self.SYNC_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._load_status()
    
    def _load_status(self) -> None:
        """Load sync status from file."""
        if self.SYNC_FILE.exists():
            try:
                with open(self.SYNC_FILE, "r") as f:
                    self.status = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load sync status: {e}. Starting fresh.")
                self.status = self._default_status()
        else:
            self.status = self._default_status()
    
    def _default_status(self) -> dict:
        """Create default sync status structure."""
        return {
            "stripe_docs": {
                "last_sync": None,
                "last_modified": None,
                "items_count": 0,
            },
            "changelog": {
                "last_sync": None,
                "last_modified": None,
                "items_count": 0,
            },
            "github": {
                "last_sync": None,
                "repos": {
                    "stripe/stripe-python": {"last_modified": None, "items_count": 0},
                    "stripe/stripe-node": {"last_modified": None, "items_count": 0},
                    "stripe/stripe-cli": {"last_modified": None, "items_count": 0},
                },
            },
            "stackoverflow": {
                "last_sync": None,
                "last_modified": None,
                "items_count": 0,
            },
        }
    
    def save(self) -> None:
        """Save sync status to file."""
        try:
            with open(self.SYNC_FILE, "w") as f:
                json.dump(self.status, f, indent=2)
            logger.info(f"Sync status saved to {self.SYNC_FILE}")
        except Exception as e:
            logger.error(f"Failed to save sync status: {e}")
    
    def get_last_sync(self, source: str) -> Optional[datetime]:
        """
        Get last sync timestamp for a source.
        
        Args:
            source: Source name ('stripe_docs', 'changelog', 'github', 'stackoverflow')
            
        Returns:
            datetime of last sync, or None if never synced
        """
        last_sync_str = self.status.get(source, {}).get("last_sync")
        if last_sync_str:
            return datetime.fromisoformat(last_sync_str)
        return None
    
    def mark_sync_start(self, source: str) -> None:
        """Mark when sync started (for tracking duration)."""
        if source not in self.status:
            self.status[source] = self._default_status()[source]
        self.status[source]["_sync_start"] = datetime.utcnow().isoformat()
    
    def mark_sync_complete(
        self, 
        source: str, 
        items_count: int = 0,
        last_modified: Optional[datetime] = None
    ) -> None:
        """
        Mark sync as complete with metadata.
        
        Args:
            source: Source name
            items_count: Number of items scraped
            last_modified: Latest modification time from source
        """
        if source not in self.status:
            self.status[source] = self._default_status()[source]
        
        self.status[source]["last_sync"] = datetime.utcnow().isoformat()
        self.status[source]["items_count"] = items_count
        
        if last_modified:
            self.status[source]["last_modified"] = last_modified.isoformat()
        
        logger.info(
            f"Marked {source} sync complete: {items_count} items, "
            f"last_modified: {last_modified}"
        )
        self.save()
    
    def mark_repo_sync_complete(
        self,
        repo: str,
        items_count: int = 0,
        last_modified: Optional[datetime] = None
    ) -> None:
        """
        Mark GitHub repo sync as complete.
        
        Args:
            repo: Repository name (e.g., 'stripe/stripe-python')
            items_count: Number of issues/PRs scraped
            last_modified: Latest issue timestamp
        """
        if "github" not in self.status:
            self.status["github"] = {"last_sync": None, "repos": {}}
        
        if repo not in self.status["github"]["repos"]:
            self.status["github"]["repos"][repo] = {
                "last_modified": None,
                "items_count": 0
            }
        
        self.status["github"]["repos"][repo]["items_count"] = items_count
        if last_modified:
            self.status["github"]["repos"][repo]["last_modified"] = last_modified.isoformat()
        
        self.status["github"]["last_sync"] = datetime.utcnow().isoformat()
        logger.info(f"Marked {repo} sync complete: {items_count} items")
        self.save()
    
    def get_repo_last_modified(self, repo: str) -> Optional[datetime]:
        """Get last modified timestamp for a GitHub repo."""
        repo_data = self.status.get("github", {}).get("repos", {}).get(repo, {})
        last_mod = repo_data.get("last_modified")
        if last_mod:
            return datetime.fromisoformat(last_mod)
        return None
    
    def should_sync(
        self, 
        source: str,
        min_interval_hours: int = 24
    ) -> bool:
        """
        Check if source should be synced based on interval.
        
        Args:
            source: Source name
            min_interval_hours: Minimum hours between syncs
            
        Returns:
            True if should sync, False if recently synced
        """
        last_sync = self.get_last_sync(source)
        if not last_sync:
            logger.info(f"{source} never synced, should sync now")
            return True
        
        elapsed = datetime.utcnow() - last_sync
        should = elapsed > timedelta(hours=min_interval_hours)
        
        if should:
            logger.info(
                f"{source} last synced {elapsed.total_seconds()/3600:.1f} hours ago, should sync"
            )
        else:
            logger.info(
                f"{source} synced recently ({elapsed.total_seconds()/3600:.1f}h ago), skipping"
            )
        return should
    
    def get_sync_report(self) -> str:
        """Get human-readable sync report."""
        lines = ["=== SYNC STATUS REPORT ==="]
        
        for source, data in self.status.items():
            if source == "github":
                lines.append(f"\n{source}:")
                last_sync = data.get("last_sync")
                if last_sync:
                    lines.append(f"  Last sync: {last_sync}")
                
                for repo, repo_data in data.get("repos", {}).items():
                    items = repo_data.get("items_count", 0)
                    modified = repo_data.get("last_modified", "never")
                    lines.append(f"  {repo}: {items} items, last_modified: {modified}")
            else:
                lines.append(f"\n{source}:")
                lines.append(f"  Last sync: {data.get('last_sync', 'never')}")
                lines.append(f"  Items: {data.get('items_count', 0)}")
                lines.append(f"  Last modified: {data.get('last_modified', 'unknown')}")
        
        return "\n".join(lines)


# Global instance
tracker = SyncTracker()
