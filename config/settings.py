# WHY THIS EXISTS: Single typed config class for all environment variables.

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── SARVAM ──────────────────────────────────────────
    sarvam_api_key: str = Field(..., description="Sarvam AI API key")
    sarvam_base_url: str = Field(
        default="https://api.sarvam.ai/v1",
        description="Sarvam API base URL"
    )
    sarvam_model_30b: str = Field(
        default="sarvam-m",
        description="Sarvam 30B model string"
    )
    
    # Cost tracking in INR — update from dashboard.sarvam.ai
    sarvam_cost_per_1k_input_tokens_inr: float = 0.50
    sarvam_cost_per_1k_output_tokens_inr: float = 0.66

    # ── SUPABASE ─────────────────────────────────────────
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_key: str = Field(..., description="Supabase service role key")
    database_url: str = Field(..., description="PostgreSQL connection string")

    # ── LANGSMITH ────────────────────────────────────────
    langsmith_api_key: str = Field(..., description="LangSmith API key")
    langsmith_project: str = Field(default="smartdocs")
    langchain_tracing_v2: bool = Field(default=True)

    # ── EXTERNAL SERVICES ────────────────────────────────
    tavily_api_key: str = Field(..., description="Tavily API key for CRAG fallback")
    redis_url: str = Field(default="redis://localhost:6379")

    # ── APP ──────────────────────────────────────────────
    app_secret_key: str = Field(..., description="App secret key")
    environment: str = Field(default="development")

    # ── HUGGINGFACE ──────────────────────────────────────
    hf_token: str = Field(default="", description="HuggingFace token")

    # ── EMBEDDING MODEL ──────────────────────────────────
    embedding_model: str = Field(default="intfloat/multilingual-e5-large")
    embedding_device: str = Field(default="cuda")
    embedding_batch_size: int = Field(default=32)

    # ── RETRIEVAL (overridden by retrieval_config.yaml) ──
    dense_weight: float = Field(default=0.7)
    sparse_weight: float = Field(default=0.3)
    reranker_proceed_threshold: float = Field(default=0.7)
    reranker_crag_threshold: float = Field(default=0.3)
    top_k_retrieval: int = Field(default=20)
    top_k_reranked: int = Field(default=5)
    cache_similarity_threshold: float = Field(default=0.95)
    language_confidence_threshold: float = Field(default=0.85)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Call get_settings() anywhere in the codebase.
    The .env file is read exactly once on first call.
    """
    return Settings()