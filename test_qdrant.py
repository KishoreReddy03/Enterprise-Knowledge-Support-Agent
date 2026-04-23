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



print(r.status_code)
print(r.text)