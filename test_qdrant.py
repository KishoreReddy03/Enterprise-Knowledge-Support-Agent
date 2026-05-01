import httpx
from config import settings

payload = {
    "points": [
        {
            "id": 1,
            "vector": [0.1] * 384,
            "payload": {"test": "ok"},
        }
    ]
}

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