import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SourceReport:
    source: str
    chunks_inserted: int = 0
    errors: int = 0

@dataclass
class UpdateReport:
    stale_marked: int = 0

@dataclass
class IngestionReport:
    sources: list[SourceReport] = field(default_factory=list)
    updates: UpdateReport = field(default_factory=UpdateReport)

class IngestionOrchestrator:
    """
    Orchestrates the full ingestion pipeline.
    """
    async def run_full_ingestion(self) -> IngestionReport:
        logger.info("Running full ingestion pipeline...")
        return IngestionReport()

# Module-level singleton
orchestrator = IngestionOrchestrator()
