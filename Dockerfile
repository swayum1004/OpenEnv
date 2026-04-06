FROM python:3.11-slim

# Metadata for HuggingFace Spaces
LABEL maintainer="farhan"
LABEL space_sdk="docker"
LABEL tags="openenv,email,triage,support,nlp"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY email_triage_env.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "server.py"]