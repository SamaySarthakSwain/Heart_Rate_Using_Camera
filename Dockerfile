# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 – Dependency installer (cached layer)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS deps

WORKDIR /install


COPY requirements.txt .

# Install into a prefix so we can copy the whole tree in the next stage
RUN pip install --no-cache-dir --prefix=/install/pkg -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – Lean runtime image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Re-install only the runtime OS libraries (libgl1 replaces libgl1-mesa-glx in Debian Bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from the deps stage
COPY --from=deps /install/pkg /usr/local

# Copy only the source that is needed at runtime
# (experiments/, datasets/, notebooks/, tests/, docs/ are excluded via .dockerignore)
COPY our_project/ ./our_project/
COPY website/     ./website/

# Make Python find `our_project` as a top-level package
ENV PYTHONPATH=/app

# Render injects PORT at runtime; default to 10000 (Render's default)
ENV PORT=10000

# Expose the port Render will route traffic to
EXPOSE 10000

# Healthcheck so Render (and Docker) can verify the service is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# Run uvicorn; PORT is read from the environment variable injected by Render
CMD ["sh", "-c", "uvicorn our_project.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info"]
