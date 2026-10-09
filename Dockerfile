# CONSILIUM runtime image — slim CPU, NO torch (I-13: the API/worker never run
# inference in-process; vision is remote behind VisionClient).
# One image serves all roles (api / worker / fake-vision); the compose file
# picks the command per service (ARCHITECTURE.md §2.3).

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first for build-cache reuse.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code.
COPY src/ src/
COPY scripts/ scripts/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Default role: api. worker/fake-vision override `command` in docker-compose.yml.
CMD ["uvicorn", "consilium.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
