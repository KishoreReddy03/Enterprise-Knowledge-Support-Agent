import logging
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)

# Lazy initialization to prevent import-time connection errors
_client: AsyncOpenAI | None = None
_gemini_client: AsyncOpenAI | None = None

def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL
        )
    return _client

def get_gemini_client() -> AsyncOpenAI | None:
    global _gemini_client
    if _gemini_client is None and settings.GEMINI_API_KEY:
        _gemini_client = AsyncOpenAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.GEMINI_BASE_URL
        )
    return _gemini_client

async def call_fast(prompt: str, max_tokens: int = 512, temperature: float = 0.0, system: str | None = None) -> str:
    """
    Call the fast LLM (e.g., Llama-3.1-8B-instant) with fallback to Gemini.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        client = get_client()
        response = await client.chat.completions.create(
            model=settings.LLM_FAST_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=settings.LLM_GROQ_TIMEOUT,
        )
        return response.choices[0].message.content or ""
    except Exception as groq_err:
        logger.warning(
            f"Groq call failed in call_fast (Timeout/Error). Error: {groq_err}. "
            f"Attempting fallback to Gemini..."
        )
        
        gemini_client = get_gemini_client()
        if not gemini_client:
            logger.error("Gemini API key is not configured. Fallback skipped.")
            raise groq_err
            
        try:
            response = await gemini_client.chat.completions.create(
                model=settings.LLM_FALLBACK_FAST_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=settings.LLM_GEMINI_TIMEOUT,
            )
            logger.info("Successfully fell back to Gemini in call_fast.")
            return response.choices[0].message.content or ""
        except Exception as gemini_err:
            logger.error(
                f"Gemini fallback call also failed in call_fast. Error: {gemini_err}"
            )
            raise gemini_err from groq_err

async def call_strong(prompt: str, max_tokens: int = 2048, temperature: float = 0.0, system: str | None = None) -> str:
    """
    Call the strong LLM (e.g., Llama-3.3-70B-versatile) with fallback to Gemini.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        client = get_client()
        response = await client.chat.completions.create(
            model=settings.LLM_STRONG_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=settings.LLM_GROQ_TIMEOUT,
        )
        return response.choices[0].message.content or ""
    except Exception as groq_err:
        logger.warning(
            f"Groq call failed in call_strong (Timeout/Error). Error: {groq_err}. "
            f"Attempting fallback to Gemini..."
        )
        
        gemini_client = get_gemini_client()
        if not gemini_client:
            logger.error("Gemini API key is not configured. Fallback skipped.")
            raise groq_err
            
        try:
            response = await gemini_client.chat.completions.create(
                model=settings.LLM_FALLBACK_STRONG_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=settings.LLM_GEMINI_TIMEOUT,
            )
            logger.info("Successfully fell back to Gemini in call_strong.")
            return response.choices[0].message.content or ""
        except Exception as gemini_err:
            logger.error(
                f"Gemini fallback call also failed in call_strong. Error: {gemini_err}"
            )
            raise gemini_err from groq_err
