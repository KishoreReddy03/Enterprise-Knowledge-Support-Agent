import httpx
from config import settings



r = httpx.put(
    f"{settings.QDRANT_URL}/collections/stripe_docs/points",
    headers={
        "api-key": settings.QDRANT_API_KEY,
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=60,
)

print(r.status_code)
print(r.text)