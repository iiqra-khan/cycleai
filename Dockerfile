# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── Env vars ──────────────────────────────────────────────────────────────────
# HF_TOKEN and OPENAI_API_KEY are set in Render dashboard.
# Enable "Available in build" for HF_TOKEN so train_and_save.py can use it.
ENV PYTHONUNBUFFERED=1
ENV HF_DATASET=iiqra/cycleai-data

# ── Train models at build time ────────────────────────────────────────────────
RUN python train_and_save.py

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start FastAPI ─────────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]