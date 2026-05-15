import sys
import os
import asyncio
import logging

# Add project root to path so we can import core/config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingestion.chunker import SemanticChunker
from core.ingestion.embedder import DocumentEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data
SAMPLE_DOCS = [
    {
        "title": "Stripe Payments Overview",
        "content": "Stripe is a suite of APIs powering online payment processing and commerce solutions for internet businesses of all sizes. Accept payments and manage your business online.",
        "url": "https://stripe.com/docs/payments",
        "source_type": "stripe_docs"
    },
    {
        "title": "Stripe GitHub Issue #123",
        "content": "Fixed a bug in stripe-python where webhooks would fail to verify signatures on some Windows environments due to encoding issues.",
        "url": "https://github.com/stripe/stripe-python/issues/123",
        "source_type": "stripe_github_issues"
    },
    {
        "title": "Stack Overflow: Stripe Webhook Error",
        "content": "Ensure you are using the raw request body when verifying Stripe webhook signatures. If you use a JSON parser before verification, it will fail.",
        "url": "https://stackoverflow.com/questions/456",
        "source_type": "stripe_stackoverflow"
    }
]

async def seed():
    chunker_obj = SemanticChunker()
    embedder_obj = DocumentEmbedder()
    
    for doc in SAMPLE_DOCS:
        logger.info(f"Ingesting: {doc['title']}")
        chunks = chunker_obj.chunk_document(doc)
        result = embedder_obj.upsert_chunks(chunks, doc["source_type"])
        logger.info(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(seed())
