FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.cache/huggingface \
    HF_HUB_CACHE=/data/.cache/huggingface/hub \
    TORCH_HOME=/data/.cache/torch \
    LINGUABRIDGE_TEMP_DIR=/data/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements-core.txt requirements-deploy.txt ./
RUN python -m pip install -r requirements-deploy.txt

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/.cache/huggingface /data/.cache/torch /data/tmp \
    && chown -R appuser:appuser /app /data

COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 7860

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false"]
