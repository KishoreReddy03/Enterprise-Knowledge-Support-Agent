"""
Test script: LLM Fallback Mechanism with Latency Budgets
Verifies that:
1. Groq is the primary provider and succeeds under ordinary circumstances.
2. If Groq fails or times out, it gracefully falls back to Gemini if GEMINI_API_KEY is configured.
3. Fallback respects the 5-second total latency ceiling (2.5s for Groq, 2.0s for Gemini).
4. If Gemini also fails, errors are raised and chained appropriately.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.llm_client import call_fast, call_strong


class MockResponse:
    def __init__(self, content):
        self.choices = [MagicMock()]
        self.choices[0].message.content = content


async def run_fallback_tests():
    print("\n" + "=" * 80)
    print("STARTING LLM FALLBACK & LATENCY BUDGET TESTS")
    print("=" * 80)

    # Backup original settings
    orig_gemini_key = settings.GEMINI_API_KEY
    orig_groq_timeout = settings.LLM_GROQ_TIMEOUT
    orig_gemini_timeout = settings.LLM_GEMINI_TIMEOUT

    try:
        # Set dummy key for testing fallback path
        settings.GEMINI_API_KEY = "dummy_gemini_key"
        settings.LLM_GROQ_TIMEOUT = 2.5
        settings.LLM_GEMINI_TIMEOUT = 2.0

        # Create mocks for API clients
        mock_groq_client = MagicMock()
        mock_groq_completions = AsyncMock()
        mock_groq_client.chat.completions.create = mock_groq_completions

        mock_gemini_client = MagicMock()
        mock_gemini_completions = AsyncMock()
        mock_gemini_client.chat.completions.create = mock_gemini_completions

        # ═══════════════════════════════════════════════════════════════════════
        # TEST CASE 1: Successful primary Groq call
        # ═══════════════════════════════════════════════════════════════════════
        print("\n[CASE 1] Testing successful primary call (Groq succeeds)...")
        mock_groq_completions.reset_mock()
        mock_gemini_completions.reset_mock()

        mock_groq_completions.return_value = MockResponse("Response from Groq")

        with patch("core.llm_client.get_client", return_value=mock_groq_client), \
             patch("core.llm_client.get_gemini_client", return_value=mock_gemini_client):
            
            result = await call_fast("Test prompt", system="System instruction")
            
            assert result == "Response from Groq", f"Expected 'Response from Groq', got {result}"
            # Verify Groq was called with correct parameters including timeout
            mock_groq_completions.assert_called_once()
            called_args, called_kwargs = mock_groq_completions.call_args
            assert called_kwargs["timeout"] == 2.5
            assert called_kwargs["model"] == settings.LLM_FAST_MODEL
            
            # Verify Gemini was NOT called
            mock_gemini_completions.assert_not_called()
            print("         [OK] Primary call succeeded and respected Groq timeout.")

        # ═══════════════════════════════════════════════════════════════════════
        # TEST CASE 2: Groq times out / fails, successfully falls back to Gemini
        # ═══════════════════════════════════════════════════════════════════════
        print("\n[CASE 2] Testing Groq timeout/failure fallback to Gemini...")
        mock_groq_completions.reset_mock()
        mock_gemini_completions.reset_mock()

        # Groq raises exception (e.g. TimeoutError or APIError)
        mock_groq_completions.side_effect = Exception("Groq Service Unavailable or Timeout")
        mock_gemini_completions.return_value = MockResponse("Response from Gemini")

        with patch("core.llm_client.get_client", return_value=mock_groq_client), \
             patch("core.llm_client.get_gemini_client", return_value=mock_gemini_client):
            
            result = await call_fast("Test prompt")
            
            assert result == "Response from Gemini", f"Expected 'Response from Gemini', got {result}"
            # Verify Groq was tried
            mock_groq_completions.assert_called_once()
            # Verify Gemini fallback was called with correct parameters including timeout
            mock_gemini_completions.assert_called_once()
            gemini_args, gemini_kwargs = mock_gemini_completions.call_args
            assert gemini_kwargs["timeout"] == 2.0
            assert gemini_kwargs["model"] == settings.LLM_FALLBACK_FAST_MODEL
            print("         [OK] Groq failure triggered fallback to Gemini with proper timeout.")

        # ═══════════════════════════════════════════════════════════════════════
        # TEST CASE 3: Groq fails, Gemini is not configured
        # ═══════════════════════════════════════════════════════════════════════
        print("\n[CASE 3] Testing Groq failure when Gemini fallback is unconfigured...")
        mock_groq_completions.reset_mock()
        mock_gemini_completions.reset_mock()

        settings.GEMINI_API_KEY = None  # Unconfigured
        mock_groq_completions.side_effect = Exception("Groq Rate Limit")

        with patch("core.llm_client.get_client", return_value=mock_groq_client), \
             patch("core.llm_client.get_gemini_client", return_value=None):
            
            try:
                await call_strong("Complex task")
                assert False, "Should have raised an exception"
            except Exception as e:
                assert "Groq Rate Limit" in str(e), f"Expected Groq Rate Limit error, got {e}"
                print("         [OK] Exception propagated correctly when fallback is unconfigured.")

        # ═══════════════════════════════════════════════════════════════════════
        # TEST CASE 4: Both Groq and Gemini fail (propagates chained errors)
        # ═══════════════════════════════════════════════════════════════════════
        print("\n[CASE 4] Testing double failure (both Groq and Gemini fail)...")
        settings.GEMINI_API_KEY = "dummy_gemini_key"
        mock_groq_completions.reset_mock()
        mock_gemini_completions.reset_mock()

        mock_groq_completions.side_effect = Exception("Groq Timeout")
        mock_gemini_completions.side_effect = Exception("Gemini Overloaded")

        with patch("core.llm_client.get_client", return_value=mock_groq_client), \
             patch("core.llm_client.get_gemini_client", return_value=mock_gemini_client):
            
            try:
                await call_strong("Complex task")
                assert False, "Should have raised an exception"
            except Exception as e:
                assert "Gemini Overloaded" in str(e), f"Expected Gemini Overloaded error, got {e}"
                assert e.__cause__ is not None, "Original Groq exception should be chained via `raise ... from ...`"
                assert "Groq Timeout" in str(e.__cause__), f"Expected chained Groq Timeout, got {e.__cause__}"
                print("         [OK] Error chaining correctly preserves original Groq and Gemini exceptions.")

    finally:
        # Restore settings
        settings.GEMINI_API_KEY = orig_gemini_key
        settings.LLM_GROQ_TIMEOUT = orig_groq_timeout
        settings.LLM_GEMINI_TIMEOUT = orig_gemini_timeout

    print("\n" + "=" * 80)
    print("ALL LLM FALLBACK & LATENCY BUDGET TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_fallback_tests())
