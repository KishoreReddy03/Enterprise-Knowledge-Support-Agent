"""
LLM client for Stripe Support Agent.

Uses OpenRouter API with OpenAI-compatible client.
Provides unified interface for fast and strong LLM operations.
"""

import logging

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI-compatible client for OpenRouter
client = AsyncOpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
    default_headers={
        "HTTP-Referer": "https://github.com/KishoreReddy03/Enterprise-Knowledge-Support-Agent",
        "X-Title": "Enterprise Support Agent",
    },
)


async def call_fast(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    json_mode: bool = False,
) -> str:
    """
    Call fast LLM model for lightweight operations.
    
    Used for ticket classification, sufficiency checks, quality gates.
    
    Args:
        prompt: The prompt to send to the LLM.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0-2).
        json_mode: Whether to request JSON-only output.
        
    Returns:
        The LLM response text.
        
    Raises:
        Exception: If API call fails.
    """
    try:
        response = await client.chat.completions.create(
            model=settings.LLM_FAST_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Fast LLM call failed: {e}", exc_info=True)
        raise


async def call_strong(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    json_mode: bool = False,
) -> str:
    """
    Call strong LLM model for complex operations.
    
    Used for response generation, contradiction detection.
    
    Args:
        prompt: The prompt to send to the LLM.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0-2).
        json_mode: Whether to request JSON-only output.
        
    Returns:
        The LLM response text.
        
    Raises:
        Exception: If API call fails.
    """
    try:
        response = await client.chat.completions.create(
            model=settings.LLM_STRONG_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Strong LLM call failed: {e}", exc_info=True)
        raise


async def call(
    prompt: str,
    model: str = "fast",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    json_mode: bool = False,
) -> str:
    """
    Generic LLM call with model selection.
    
    Args:
        prompt: The prompt to send.
        model: "fast" or "strong".
        max_tokens: Maximum tokens.
        temperature: Sampling temperature.
        json_mode: Whether JSON-only output is requested.
        
    Returns:
        The LLM response text.
    """
    if model == "fast":
        return await call_fast(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
    elif model == "strong":
        return await call_strong(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
    else:
        raise ValueError(f"Unknown model: {model}. Use 'fast' or 'strong'.")


async def test_connection() -> None:
    """
    Test LLM connection with a simple prompt.
    
    Raises:
        Exception: If connection fails.
    """
    try:
        logger.info("Testing LLM connection...")
        
        response = await call_fast(
            prompt="Say 'Hello from OpenRouter' in exactly 5 words or less.",
            max_tokens=64,
            temperature=0.0,
        )
        
        logger.info(f"✓ LLM test successful. Response: {response[:100]}")
        print(f"✓ LLM test successful\n  Model: {settings.LLM_FAST_MODEL}\n  Response: {response}")
        
    except Exception as e:
        logger.error(f"✗ LLM test failed: {e}")
        print(f"✗ LLM test failed: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    
    asyncio.run(test_connection())
