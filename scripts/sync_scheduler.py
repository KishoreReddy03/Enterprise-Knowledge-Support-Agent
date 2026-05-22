"""
Scheduler for periodic incremental data syncing.

Runs incremental sync at configured intervals to keep knowledge base fresh.
Can be run as a background service or cron job.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from core.ingestion.incremental_scraper import incremental_scraper
from core.ingestion.scheduler import orchestrator
from core.ingestion.sync_tracker import tracker

logger = logging.getLogger(__name__)


class SyncScheduler:
    """
    Schedules and runs periodic data syncs.
    
    Can run standalone as a service or be triggered by external schedulers.
    """
    
    def __init__(self, check_interval_minutes: int = 60):
        """
        Initialize scheduler.
        
        Args:
            check_interval_minutes: How often to check if sync is needed
        """
        self.check_interval = check_interval_minutes * 60  # Convert to seconds
        self.running = False
    
    async def run_once(self) -> dict:
        """
        Run incremental sync once.
        
        Fetches new data from all sources and updates the vector store.
        
        Returns:
            Dict with sync results and statistics
        """
        logger.info("=" * 60)
        logger.info("Starting incremental data sync")
        logger.info(f"Current time: {datetime.utcnow().isoformat()}")
        logger.info("=" * 60)
        
        try:
            # Step 1: Scrape new data incrementally
            logger.info("\n[STEP 1] Scraping new/updated data from sources...")
            scraped_data = await incremental_scraper.sync_all_sources()
            
            # Step 2: Chunk and insert into vector store
            logger.info("\n[STEP 2] Processing and indexing new documents...")
            
            # Count total items scraped
            total_items = sum(
                len(items) if items else 0 
                for items in scraped_data.values()
            )
            
            if total_items == 0:
                logger.info("No new data to sync (all sources recently updated)")
                return {
                    "success": True,
                    "total_new_items": 0,
                    "sources_synced": 0,
                }
            
            # Run full ingestion pipeline on scraped data
            # (This would need to be enhanced to accept incremental data)
            logger.info(f"Processing {total_items} new items through pipeline...")
            
            report = await orchestrator.run_full_ingestion(
                sources=["github", "stackoverflow", "docs", "changelog"]
            )
            
            logger.info("\n" + "=" * 60)
            logger.info("SYNC COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total new items: {total_items}")
            logger.info(f"Total chunks created: {report.total_chunks}")
            logger.info(f"Vector store inserts: {report.total_documents}")
            
            return {
                "success": True,
                "total_new_items": total_items,
                "total_chunks": report.total_chunks,
                "sources_synced": sum(1 for s in scraped_data.values() if s),
            }
        
        except Exception as e:
            logger.error(f"Sync failed with error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    async def run_continuous(self) -> None:
        """
        Run scheduler continuously, checking periodically if sync is needed.
        
        This is useful for running as a background service.
        Press Ctrl+C to stop.
        """
        self.running = True
        logger.info(
            f"Starting continuous sync scheduler "
            f"(checking every {self.check_interval}s)"
        )
        
        try:
            while self.running:
                try:
                    # Run one sync cycle
                    result = await self.run_once()
                    
                    if result["success"]:
                        logger.info(f"Sync successful: {result}")
                    else:
                        logger.error(f"Sync failed: {result['error']}")
                    
                    # Wait before next check
                    logger.info(
                        f"Next sync check in {self.check_interval}s "
                        f"({self.check_interval//60} minutes)"
                    )
                    await asyncio.sleep(self.check_interval)
                
                except KeyboardInterrupt:
                    logger.info("Sync interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in sync loop: {e}", exc_info=True)
                    await asyncio.sleep(60)  # Wait 1 min before retry
        
        finally:
            self.running = False
            logger.info("Sync scheduler stopped")
    
    def stop(self) -> None:
        """Stop the continuous scheduler."""
        self.running = False
        logger.info("Stop signal sent to scheduler")


# Global instance
scheduler = SyncScheduler(check_interval_minutes=60)


async def main():
    """Main entry point for running sync scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental data sync scheduler")
    parser.add_argument(
        "--mode",
        choices=["once", "continuous"],
        default="once",
        help="Run once or continuously"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in minutes (for continuous mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "once":
        logger.info("Running sync once...")
        result = await scheduler.run_once()
        sys.exit(0 if result["success"] else 1)
    else:
        scheduler.check_interval = args.interval * 60
        await scheduler.run_continuous()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    asyncio.run(main())
