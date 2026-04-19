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
    OPENROUTER_API_KEY: str = Field(
        ...,
        description="OpenRouter API key for LLM access",
    )
    
    # Supabase - All required
    SUPABASE_URL: str = Field(
        ...,
        description="Supabase project URL",
    )
    SUPABASE_ANON_KEY: str = Field(
        ...,
        description="Supabase anonymous/public key",
    )
    SUPABASE_SERVICE_KEY: str = Field(
        ...,
        description="Supabase service role key for admin operations",
    )
    
    # Qdrant - All required
    QDRANT_URL: str = Field(
        ...,
        description="Qdrant vector database URL",
    )
    QDRANT_API_KEY: str = Field(
        ...,
        description="Qdrant API key",
    )
    
    # Upstash Redis - All required
    UPSTASH_REDIS_URL: str = Field(
        ...,
        description="Upstash Redis REST URL",
    )
    UPSTASH_REDIS_TOKEN: str = Field(
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
        default="meta-llama/llama-3.1-8b-instruct:free",
        description="Fast LLM model for lightweight operations",
    )
    LLM_STRONG_MODEL: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free",
        description="Strong LLM model for complex operations",
    )
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    
    # Qdrant collections
    QDRANT_DOCS_COLLECTION: str = Field(
        default="stripe_docs",
        description="Qdrant collection name for Stripe documentation",
    )
    QDRANT_ISSUES_COLLECTION: str = Field(
        default="stripe_github_issues",
        description="Qdrant collection name for GitHub issues",
    )
    QDRANT_STACKOVERFLOW_COLLECTION: str = Field(
        default="stripe_stackoverflow",
        description="Qdrant collection name for StackOverflow posts",
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
        default=0.75,
        description="Minimum similarity score for retrieved documents",
    )
    MAX_AGENT_RETRIES: int = Field(
        default=2,
        description="Maximum retry attempts for agent operations",
    )
    
    
    GITHUB_TOKEN: str | None = Field(
        default=None,
        description="GitHub personal access token for API access (optional, increases rate limit)",
    )

    # Sensitive fields that should be masked in logs
    _SENSITIVE_FIELDS: ClassVar[set[str]] = {
        "OPENROUTER_API_KEY",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_KEY",
        "QDRANT_API_KEY",
        "UPSTASH_REDIS_TOKEN",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "GITHUB_TOKEN",
    }

    def display_config(self) -> None:
        """
        Display all configuration values with sensitive fields masked.
        
        Logs each configuration key-value pair, replacing sensitive values with '***'.
        """
        logger.info("=== Stripe Support Agent Configuration ===")
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if field_name in self._SENSITIVE_FIELDS:
                display_value = "***"
            else:
                display_value = value
            logger.info(f"{field_name}: {display_value}")
        logger.info("=== End Configuration ===")


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
