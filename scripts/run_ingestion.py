import asyncio
import sys
sys.path.insert(0, ".")

from core.ingestion.scheduler import orchestrator

async def main():
    # GitHub + StackOverflow use REST APIs.
    # Stripe docs and changelog now use the OpenAPI GitHub repository APIs.
    report = await orchestrator.run_full_ingestion(
        sources=["github", "stackoverflow", "docs", "changelog"]
    )

    print("\n=== INGESTION REPORT ===")
    print(f"Success: {report.success}")
    print(f"Total documents: {report.total_documents}")
    print(f"Total chunks: {report.total_chunks}")

    for source in report.sources:
        print(f"\nSource: {source.source}")
        print(f"  Success: {source.success}")
        print(f"  Documents: {source.documents_scraped}")
        print(f"  Chunks: {source.chunks_created}")
        print(f"  Inserted: {source.chunks_inserted}")
        print(f"  Updated: {source.chunks_updated}")
        print(f"  Failed: {source.chunks_failed}")
        if source.error:
            print(f"  Error: {source.error}")

asyncio.run(main())