"""
Configuration module for Stripe Support Agent.

Uses pydantic-settings to load and validate environment variables from .env file.
All required variables will fail with clear error messages if missing.
"""

import logging
import sys
from typing import ClassVar

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Required variables will raise a validation error with a clear message if missing.
    Optional variables have sensible defaults for development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # API Keys - All required, no defaults
    GROQ_API_KEY: str = Field(
        ...,
        description="Groq API key for LLM access",
    )
    
    # Neon Postgres connection (for pgvector operations)
    NEON_DB_URL: str = Field(
        ...,
        description="Neon Postgres connection string for pgvector operations. Get from Neon dashboard -> Connection Details. Format: postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require",
    )
    
    # Upstash Redis - All required
    UPSTASH_REDIS_REST_URL: str = Field(
        ...,
        description="Upstash Redis REST URL",
    )
    UPSTASH_REDIS_REST_TOKEN: str = Field(
        ...,
        description="Upstash Redis REST token",
    )
    
    # Langfuse - Required keys with optional host
    LANGFUSE_PUBLIC_KEY: str = Field(
        ...,
        description="Langfuse public key for tracing",
    )
    LANGFUSE_SECRET_KEY: str = Field(
        ...,
        description="Langfuse secret key for tracing",
    )
    LANGFUSE_HOST: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL",
    )
    
    # Environment
    ENVIRONMENT: str = Field(
        default="development",
        description="Runtime environment (development, staging, production)",
    )
    
    # Model configuration
    LLM_FAST_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Fast LLM model for lightweight operations (Groq)",
    )
    LLM_STRONG_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Strong LLM model for complex operations (Groq)",
    )
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings (384 dimensions)",
    )
    GROQ_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq API base URL",
    )
    
    # Gemini Fallback configuration
    GEMINI_API_KEY: str | None = Field(
        default=None,
        description="Gemini API key for fallback operations",
    )
    GEMINI_BASE_URL: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        description="Gemini API base URL for OpenAI compatibility",
    )
    LLM_FALLBACK_FAST_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Gemini fallback model for fast tasks",
    )
    LLM_FALLBACK_STRONG_MODEL: str = Field(
        default="gemini-2.5-pro",
        description="Gemini fallback model for strong tasks",
    )
    LLM_GROQ_TIMEOUT: float = Field(
        default=2.5,
        description="Timeout for Groq calls in seconds",
    )
    LLM_GEMINI_TIMEOUT: float = Field(
        default=2.0,
        description="Timeout for Gemini fallback calls in seconds",
    )
    
    # Thresholds
    CONFIDENCE_HIGH_THRESHOLD: float = Field(
        default=0.90,
        description="Threshold for high confidence responses (auto-send eligible)",
    )
    CONFIDENCE_MEDIUM_THRESHOLD: float = Field(
        default=0.75,
        description="Threshold for medium confidence responses (review recommended)",
    )
    CONFIDENCE_LOW_THRESHOLD: float = Field(
        default=0.60,
        description="Threshold for low confidence responses (escalation likely)",
    )
    RETRIEVAL_SIMILARITY_THRESHOLD: float = Field(
        default=0.2,
        description="Minimum similarity score for retrieved documents (TEMPORARILY LOWERED FOR DEBUG)",
    )
    MAX_AGENT_RETRIES: int = Field(
        default=2,
        description="Maximum retry attempts for agent operations",
    )
    
    # Optional API keys
    GITHUB_TOKEN: str | None = Field(
        default=None,
        description="GitHub personal access token for API access (optional, increases rate limit)",
    )

    # Sensitive fields that should be masked in logs
    _SENSITIVE_FIELDS: ClassVar[set[str]] = {
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "NEON_DB_URL",
        "UPSTASH_REDIS_REST_TOKEN",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "GITHUB_TOKEN",
    }

    

def get_settings() -> Settings:
    """
    Load and validate settings from environment.
    
    Returns:
        Settings: Validated settings instance.
        
    Raises:
        SystemExit: If required environment variables are missing.
    """
    try:
        return Settings()
    except ValidationError as e:
        logger.error("Configuration validation failed!")
        for error in e.errors():
            field = error["loc"][0] if error["loc"] else "unknown"
            msg = error["msg"]
            logger.error(f"  Missing or invalid: {field} - {msg}")
        logger.error(
            "Please check your .env file and ensure all required variables are set. "
            "See .env.example for reference."
        )
        sys.exit(1)


# Global settings instance - imported by other modules
settings = get_settings()


if __name__ == "__main__":
    settings.display_config()
