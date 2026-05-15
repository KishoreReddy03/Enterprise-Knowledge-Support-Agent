import logging
from typing import ClassVar, Literal
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application settings and environment configuration.
    
    Loads values from environment variables or .env file.
    Includes validation for required API keys and connection strings.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # API Keys - All required
    GROQ_API_KEY: str = Field(..., description="Groq API key for LLM access")
    
    # Neon Postgres connection (for pgvector operations)
    NEON_DB_URL: str = Field(
        ...,
        description="Neon Postgres connection string. Format: postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require",
    )
    
    # Upstash Redis (required)
    UPSTASH_REDIS_REST_URL: str = Field(..., description="Upstash Redis REST URL")
    UPSTASH_REDIS_REST_TOKEN: str = Field(..., description="Upstash Redis REST token")
    
    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str = Field(..., description="Langfuse public key")
    LANGFUSE_SECRET_KEY: str = Field(..., description="Langfuse secret key")
    LANGFUSE_HOST: str = Field("https://cloud.langfuse.com", description="Langfuse host")

    # Environment
    ENVIRONMENT: Literal["development", "production", "test"] = "development"
    
    # LLM Models
    LLM_FAST_MODEL: str = "llama-3.1-8b-instant"
    LLM_STRONG_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    
    # Embedding Model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Thresholds
    CONFIDENCE_HIGH_THRESHOLD: float = 0.90
    CONFIDENCE_MEDIUM_THRESHOLD: float = 0.75
    CONFIDENCE_LOW_THRESHOLD: float = 0.60
    RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.30
    MAX_AGENT_RETRIES: int = 2
    
    # Optional
    GITHUB_TOKEN: str | None = None

    # Sensitive fields for masking
    _SENSITIVE_FIELDS: ClassVar[set[str]] = {
        "GROQ_API_KEY",
        "NEON_DB_URL",
        "UPSTASH_REDIS_REST_TOKEN",
        "LANGFUSE_SECRET_KEY",
        "GITHUB_TOKEN",
    }

    def display_config(self) -> None:
        """Display all configuration values with sensitive fields masked."""
        logger.info("=== Stripe Support Agent Configuration ===")
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            display_value = "***" if field_name in self._SENSITIVE_FIELDS else value
            logger.info(f"{field_name}: {display_value}")
        logger.info("=== End Configuration ===")

def get_settings() -> Settings:
    """Load and validate settings from environment."""
    try:
        return Settings()
    except ValidationError as e:
        logger.error("Configuration validation failed!")
        for error in e.errors():
            logger.error(f"  {error['loc']}: {error['msg']}")
        import sys
        sys.exit(1)

settings = get_settings()
