# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable --no-cache

FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 anomavision \
    && useradd --uid 10001 --gid anomavision --create-home --shell /bin/bash anomavision

WORKDIR /app

COPY --from=builder --chown=anomavision:anomavision /app/.venv /app/.venv
COPY --chown=anomavision:anomavision configs ./configs
COPY --chown=anomavision:anomavision src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    GRADIO_ANALYTICS_ENABLED="False" \
    ANOMAVISION_BUNDLES_DIR="/app/artifacts/bundles"

USER anomavision

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "anomavision.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
