import logging
import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RawDocument:
    title: str
    content: str
    url: str
    source_type: str
    metadata: dict = field(default_factory=dict)

class BaseScraper:
    async def fetch(self, url: str) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

class StripeDataScraper(BaseScraper):
    async def scrape_docs(self, url: str) -> RawDocument:
        html = await self.fetch(url)
        soup = BeautifulSoup(html, "html.parser")
        
        title = soup.title.string if soup.title else "Stripe Docs"
        content = soup.get_text(separator="\n", strip=True)
        
        return RawDocument(
            title=title,
            content=content,
            url=url,
            source_type="stripe_docs"
        )

# Module-level singleton
scraper = StripeDataScraper()
