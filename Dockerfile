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

# ── Runtime env (injected safely by Render at container start) ────────────────
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start: train first, then serve ───────────────────────────────────────────
# HF_TOKEN, HF_DATASET, OPENAI_API_KEY are available here from Render dashboard
CMD ["sh", "-c", "python train_and_save.py && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]