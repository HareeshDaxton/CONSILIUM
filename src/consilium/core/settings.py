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
    # 32 bytes (or base64 of 32 bytes) for AES-256-GCM column encryption
    # (clinical_notes.raw_redacted_text_enc). None only where the column is
    # never written/read (pure unit tests); serving must set it (I-6).
    field_encryption_key: SecretStr | None = None

    # ----- Vision (remote endpoint — I-13/I-14) -----
    vision_client_mode: Literal["http", "fake"] = "fake"
    vision_endpoint_url: str | None = None
    vision_endpoint_token: SecretStr | None = None
    vision_pinned_revision: str | None = None
    vision_timeout_s: float = Field(default=120.0, gt=0)

    # ----- Guardrails model rail (§14) -----
    # "nemo" in any real deployment; "passthrough" only for the local compose
    # stack without the optional nemoguardrails dependency. Deterministic
    # input/output checks run in BOTH modes.
    rails_engine: Literal["nemo", "passthrough"] = "nemo"

    # ----- Observability (§18; optional in dev) -----
    langfuse_host: str | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    otel_endpoint: str | None = None

    # ----- Auth (§17) -----
    jwt_secret: SecretStr  # required secret — no default (HS* signing key)
    jwt_algorithm: Literal["HS256", "HS384", "HS512", "RS256", "ES256"] = "HS256"
    # OIDC-ready: with RS256/ES256 the IdP's public key verifies tokens and
    # issuer/audience are checked when set.
    jwt_public_key: SecretStr | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    api_docs_enabled: bool = False  # /docs off unless explicitly enabled (non-prod)

    # ----- Google sign-in (review UI) -----
    # A verified Google ID token is exchanged for a CONSILIUM session token
    # (HS* signed with JWT_SECRET). OFF unless GOOGLE_CLIENT_ID is set; the
    # client ID is public (it is embedded in the web page). Roles come ONLY from
    # these comma-separated email allowlists — a verified Google account in
    # neither list is refused (fail closed).
    google_client_id: str | None = None
    google_clinicians: str = ""
    google_submitters: str = ""
    session_ttl_minutes: int = Field(default=480, ge=5, le=1440)

    # ----- Worker (§5.2 Postgres-backed job queue) -----
    worker_poll_interval_s: float = Field(default=1.0, gt=0)
    worker_max_attempts: int = Field(default=3, ge=1)

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
    def _check_google_sessions_signable(self) -> Self:
        if self.google_client_id and not self.jwt_algorithm.startswith("HS"):
            raise ValueError(
                "Google sign-in issues HS*-signed session tokens; set JWT_ALGORITHM=HS256 "
                "(or leave GOOGLE_CLIENT_ID unset when tokens come from an external IdP)"
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
