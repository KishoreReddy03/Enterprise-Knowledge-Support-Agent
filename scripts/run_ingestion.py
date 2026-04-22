import asyncio
from core.ingestion.scrapers import scraper

async def main():
    docs = await scraper.scrape_stripe_docs(sections=["payments", "webhooks"])
    print(f"Stripe docs fetched: {len(docs)}")

    issues = await scraper.fetch_github_issues(max_per_repo=20)
    print(f"GitHub issues fetched: {len(issues)}")

    so = await scraper.fetch_stackoverflow_questions(max_questions=20)
    print(f"StackOverflow questions fetched: {len(so)}")

asyncio.run(main())