"""Единая точка чтения настроек. Больше нигде в коде нет os.environ."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Absolute path keeps scripts working whether they are launched from the
        # repository root or from backend/.
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GigaChat models
    llm_router_model: str = ""
    llm_model_routing_enabled: bool = True
    llm_model_routing_policy: dict[str, Literal["small", "main"]] = Field(
        default_factory=lambda: {
            "router.route_classification": "small",
            "nutrition.food_translation": "small",
            "*": "main",
        }
    )
    agent_baseline_version: str = "baseline-v1"

    # GigaChat
    gigachat_auth_key: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-2"

    # Retries are used only around idempotent LLM and read operations.
    safe_retry_max_attempts: int = Field(default=3, ge=1, le=10)
    safe_retry_base_delay_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    safe_retry_max_delay_seconds: float = Field(default=4.0, ge=0.0, le=120.0)
    safe_retry_jitter_ratio: float = Field(default=0.25, ge=0.0, le=1.0)

    # Shared GigaChat circuit breaker state lives in Redis across all workers.
    llm_circuit_breaker_enabled: bool = True
    llm_circuit_breaker_failure_threshold: int = Field(default=5, ge=1, le=100)
    llm_circuit_breaker_recovery_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=3_600.0,
    )
    llm_circuit_breaker_half_open_lease_seconds: float = Field(
        default=210.0,
        ge=1.0,
        le=3_600.0,
    )
    llm_circuit_breaker_state_ttl_seconds: int = Field(
        default=3_600,
        ge=60,
        le=86_400,
    )

    # Supabase
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_jwt_audience: str = "authenticated"

    # HTTP API
    api_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Redis-backed background jobs
    redis_url: str = "redis://127.0.0.1:6379/0"
    agent_job_queue: str = "athena-agent"
    agent_job_ttl_seconds: int = 3_600

    # Retrieval-augmented generation
    rag_enabled: bool = True
    rag_retrieval_limit: int = 6
    rag_min_similarity: float = 0.55
    rag_context_max_chars: int = 12_000

    app_env: str = "dev"
    test_user_id: str = "4c58346d-801f-4241-a349-02a2736361f0"

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.api_cors_origins.split(",")
            if origin.strip()
        ]

    @field_validator("llm_model_routing_policy")
    @classmethod
    def validate_model_routing_policy(
        cls,
        policy: dict[str, Literal["small", "main"]],
    ) -> dict[str, Literal["small", "main"]]:
        """Accept exact or one-segment wildcard rules only."""
        for rule in policy:
            if rule == "*":
                continue
            parts = rule.split(".")
            if len(parts) != 2 or any(not cls._valid_model_route_segment(part) for part in parts):
                raise ValueError(
                    "LLM model route keys must be '*', 'node.purpose', "
                    "'node.*', or '*.purpose'"
                )
        return policy

    @staticmethod
    def _valid_model_route_segment(segment: str) -> bool:
        if segment == "*":
            return True
        normalized = segment.replace("_", "").replace("-", "")
        return bool(normalized) and normalized.isalnum() and segment == segment.lower()


settings = Settings()
