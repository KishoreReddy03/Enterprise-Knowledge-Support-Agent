import asyncio
from core.ingestion.scheduler import orchestrator

async def main():
    report = await orchestrator.run_full_ingestion(
        sources=["docs"]
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