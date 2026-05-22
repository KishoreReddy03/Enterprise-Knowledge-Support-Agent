import logging
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)

# Lazy initialization to prevent import-time connection errors
_client: AsyncOpenAI | None = None

def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL
        )
    return _client

async def call_fast(prompt: str, max_tokens: int = 512, temperature: float = 0.0, system: str | None = None) -> str:
    """
    Call the fast LLM (e.g., Llama-3.1-8B-instant) for high-speed tasks.
    """
    try:
        client = get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model=settings.LLM_FAST_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Error calling fast LLM: {e}")
        raise

async def call_strong(prompt: str, max_tokens: int = 2048, temperature: float = 0.0, system: str | None = None) -> str:
    """
    Call the strong LLM (e.g., Llama-3.3-70B-versatile) for complex reasoning tasks.
    """
    try:
        client = get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model=settings.LLM_STRONG_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Error calling strong LLM: {e}")
        raise
