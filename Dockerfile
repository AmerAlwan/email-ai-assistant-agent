# syntax=docker/dockerfile:1

# uv's official image ships both uv and Python — no separate install needed.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Place the virtual environment outside /app so the bind-mount of ./agent:/app
# doesn't shadow it at runtime.
ENV UV_PROJECT_ENVIRONMENT=/venv

# Compile .pyc files and use copy link mode for faster layer reuse.
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# ── Dependency layer ─────────────────────────────────────────────────────────
# Copy only the dependency manifests first.  Docker caches this layer
# independently, so `uv sync` only re-runs when pyproject.toml / uv.lock change
# — not on every source-code edit.
COPY pyproject.toml ./
# uv.lock is optional on first bootstrap; copy it when it exists.
COPY uv.lock* ./

# BuildKit cache mount keeps downloaded wheels between builds for near-instant
# reinstalls when only a subset of deps change.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# ── Source layer ──────────────────────────────────────────────────────────────
# In development the source is bind-mounted, so this layer is only used for
# production builds or the very first boot before the mount is active.
COPY . .

RUN chmod +x /app/start.sh

EXPOSE 8000

# Start both the FastAPI HTTP server and the LiveKit agent worker.
CMD ["/app/start.sh"]
