# Multi-stage build.
#
# The single-stage version shipped build-essential, the full dev dependency set
# (pytest, ruff, mypy) and ran as root. Compilers and test tooling in a
# production image are attack surface that serves no runtime purpose.

# ── Builder ───────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# build-essential and libpq-dev are needed to compile wheels, and only here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install into a virtualenv so the runtime stage can copy one self-contained
# directory rather than picking through site-packages.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# No [dev]: pytest/ruff/mypy are CI concerns, not runtime ones.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ── Runtime ───────────────────────────────────────
FROM python:3.11-slim AS runtime

# tesseract is a runtime dependency of the OCR path. libpq5 is the client
# library only — the -dev headers stay in the builder.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run unprivileged. A container escape from a root process is a host
# compromise; from this one it is not.
RUN useradd --create-home --uid 10001 ragscope

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HuggingFace caches the cross-encoder here; mount a volume to avoid
    # re-downloading ~88MB on every container start.
    HF_HOME=/home/ragscope/.cache/huggingface

WORKDIR /app
COPY --chown=ragscope:ragscope . .

# Uploads are written at runtime, so the directory must be writable by the
# unprivileged user even when no volume is mounted.
RUN mkdir -p /app/uploads /home/ragscope/.cache/huggingface \
    && chown -R ragscope:ragscope /app/uploads /home/ragscope/.cache

USER ragscope

EXPOSE 8000

# Readiness (not liveness) so orchestrators wait for Postgres and Redis before
# routing traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/readyz || exit 1

# No --reload. Workers are set via the compose/platform layer, because the
# right count depends on the host: each worker holds its own cross-encoder and
# BM25 index in memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
