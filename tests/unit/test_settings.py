"""Unit tests for core/settings.py (ARCHITECTURE.md §8)."""

import pytest
from pydantic import ValidationError

from consilium.core.settings import Settings

REQUIRED_ENV = {
    "OPENAI_API_KEY": "sk-test-synthetic",
    "DATABASE_URL": "postgresql+asyncpg://consilium:consilium@localhost:5432/consilium",
    "JWT_SECRET": "test-jwt-secret",
}


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **{**REQUIRED_ENV, **overrides})  # type: ignore[arg-type]


class TestRequiredSecrets:
    def test_loads_with_required_env(self) -> None:
        s = make_settings()
        assert s.openai_api_key.get_secret_value() == "sk-test-synthetic"
        assert s.jwt_secret.get_secret_value() == "test-jwt-secret"
        assert s.database_url.startswith("postgresql+asyncpg://")

    @pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "DATABASE_URL", "JWT_SECRET"])
    def test_missing_secret_fails_fast(self, missing: str) -> None:
        env = {k: v for k, v in REQUIRED_ENV.items() if k != missing}
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **env)  # type: ignore[arg-type]


class TestDefaults:
    def test_defaults(self) -> None:
        s = make_settings()
        assert s.critic_max_rounds == 3  # AGENTS.md §4
        assert s.vision_timeout_s == 120.0  # cold-start budget, §8.0
        assert s.vision_client_mode == "fake"  # dev default; no network
        assert s.object_store_uri == "file://./data/objects"
        assert "-" in s.llm_model_specialist  # dated snapshot, not a floating alias

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("CRITIC_MAX_ROUNDS", "5")
        assert Settings(_env_file=None).critic_max_rounds == 5


class TestVisionMode:
    def test_http_mode_requires_endpoint_config(self) -> None:
        with pytest.raises(ValidationError, match="VISION_ENDPOINT_URL"):
            make_settings(vision_client_mode="http")

    def test_http_mode_ok_when_fully_configured(self) -> None:
        s = make_settings(
            vision_client_mode="http",
            vision_endpoint_url="https://example.endpoints.huggingface.cloud",
            vision_endpoint_token="hf_test",
            vision_pinned_revision="sha256:abc/run-001",
        )
        assert s.vision_pinned_revision == "sha256:abc/run-001"


class TestCriticDiffersFromDirector:
    """AGENTS.md §6: self-critique by the same configuration is forbidden."""

    def test_same_model_same_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError, match="critic must differ"):
            make_settings(llm_temperature_critic=0.2, llm_temperature_director=0.2)

    def test_same_model_different_temperature_ok(self) -> None:
        s = make_settings(llm_temperature_director=0.2, llm_temperature_critic=0.0)
        assert s.llm_model_critic == s.llm_model_director

    def test_different_model_ok(self) -> None:
        s = make_settings(
            llm_model_critic="gpt-4o-2024-08-06",
            llm_temperature_critic=0.2,
            llm_temperature_director=0.2,
        )
        assert s.llm_model_critic == "gpt-4o-2024-08-06"
