import logging
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class BaseScraper:
    async def fetch(self, url: str) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

class StripeDocsScraper(BaseScraper):
    async def scrape(self, url: str) -> dict:
        html = await self.fetch(url)
        soup = BeautifulSoup(html, "html.parser")
        
        # Simple extraction logic
        title = soup.title.string if soup.title else "Stripe Docs"
        content = soup.get_text(separator="\n", strip=True)
        
        return {
            "title": title,
            "content": content,
            "url": url,
            "source_type": "stripe_docs"
        }

class GitHubIssueScraper(BaseScraper):
    async def scrape(self, repo: str, issue_number: int) -> dict:
        url = f"https://github.com/{repo}/issues/{issue_number}"
        html = await self.fetch(url)
        soup = BeautifulSoup(html, "html.parser")
        
        title = soup.find("span", class_="js-issue-title").get_text(strip=True)
        content = soup.find("td", class_="comment-body").get_text(separator="\n", strip=True)
        
        return {
            "title": f"GitHub Issue: {title}",
            "content": content,
            "url": url,
            "source_type": "stripe_github_issues",
            "metadata": {"repo": repo, "issue_number": issue_number}
        }
