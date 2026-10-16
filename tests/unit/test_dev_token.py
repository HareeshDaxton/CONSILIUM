"""Section 12.1 — the dev token minter produces tokens the API accepts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from consilium.api.auth import ApiRole, decode_token
from consilium.core.settings import Settings

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev_token.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dev_token", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_minted_token_verifies_with_the_same_secret(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("JWT_SECRET", settings.jwt_secret.get_secret_value())
    for var in ("JWT_ALGORITHM", "JWT_ISSUER", "JWT_AUDIENCE"):
        monkeypatch.delenv(var, raising=False)
    assert _module().main(["--sub", "clin-1", "--role", "clinician", "--role", "clinician"]) == 0
    identity = decode_token(capsys.readouterr().out.strip(), settings)
    assert identity.sub == "clin-1" and identity.roles == frozenset({ApiRole.CLINICIAN})


def test_default_secret_is_the_compose_dev_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("JWT_SECRET", raising=False)
    module = _module()
    assert module.main(["--sub", "sub-1", "--role", "submitter"]) == 0
    out = capsys.readouterr()
    assert "DEV secret" in out.err
    dev = Settings(
        _env_file=None,
        OPENAI_API_KEY="x",
        DATABASE_URL="x",
        JWT_SECRET=module.COMPOSE_DEV_SECRET,
    )
    assert decode_token(out.out.strip(), dev).roles == frozenset({ApiRole.SUBMITTER})
