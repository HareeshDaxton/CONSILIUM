"""Application settings (ARCHITECTURE.md §8).

Loaded from environment variables (optionally a local ``.env`` file).
Secrets have NO defaults — the process must fail fast if they are missing.
All model IDs are dated snapshots from config, never floating aliases
(AGENTS.md §6); changing a model/prompt/threshold is a versioned change.
"""

from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dated snapshots, never floating aliases (AGENTS.md §6 / SKILLS.md llm-client-and-fakes).
DEFAULT_MODEL_SNAPSHOT = "gpt-4o-mini-2024-07-18"


class Settings(BaseSettings):
    """Environment-driven settings. Env var names are the field names uppercased."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # ----- LLM (OpenAI) -----
    openai_api_key: SecretStr  # required secret — no default
    llm_model_specialist: str = DEFAULT_MODEL_SNAPSHOT
    llm_model_director: str = DEFAULT_MODEL_SNAPSHOT
    llm_model_critic: str = DEFAULT_MODEL_SNAPSHOT
    llm_temperature_specialist: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_temperature_director: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_temperature_critic: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_case_token_budget: int = Field(default=100_000, gt=0)
    critic_max_rounds: int = Field(default=3, ge=1)  # AGENTS.md §4: bounded audit loop

    # ----- Persistence / storage -----
    database_url: str  # required — no default
    object_store_uri: str = "file://./data/objects"

    # ----- Vision (remote endpoint — I-13/I-14) -----
    vision_client_mode: Literal["http", "fake"] = "fake"
    vision_endpoint_url: str | None = None
    vision_endpoint_token: SecretStr | None = None
    vision_pinned_revision: str | None = None
    vision_timeout_s: float = Field(default=120.0, gt=0)

    # ----- Observability (§18; optional in dev) -----
    langfuse_host: str | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    otel_endpoint: str | None = None

    # ----- Auth (§17) -----
    jwt_secret: SecretStr  # required secret — no default
    jwt_algorithm: str = "HS256"

    @model_validator(mode="after")
    def _check_vision_mode(self) -> Self:
        if self.vision_client_mode == "http":
            missing = [
                name
                for name, value in (
                    ("VISION_ENDPOINT_URL", self.vision_endpoint_url),
                    ("VISION_ENDPOINT_TOKEN", self.vision_endpoint_token),
                    ("VISION_PINNED_REVISION", self.vision_pinned_revision),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"vision_client_mode='http' requires {', '.join(missing)} "
                    "(fail closed — AGENTS.md §8.0)"
                )
        return self

    @model_validator(mode="after")
    def _check_critic_differs_from_director(self) -> Self:
        """Critic must differ from the Director in model OR generation setting.

        Self-critique by the same configuration is weak evidence (AGENTS.md §6).
        """
        if (
            self.llm_model_critic == self.llm_model_director
            and self.llm_temperature_critic == self.llm_temperature_director
        ):
            raise ValueError(
                "critic must differ from director in model or temperature "
                "(AGENTS.md §6: self-critique by the same configuration is forbidden)"
            )
        return self
