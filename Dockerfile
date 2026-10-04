# CostGuard proxy image: slim Python 3.12, core dependencies only.
#   - no mlx / mlx-lm        (Apple-silicon only)
#   - no llmlingua / torch   (~2 GB; too big for free hosts; the default heuristic compressor needs neither)
# fastembed ONNX models are downloaded at BUILD time, so a cold start (e.g. Render waking up) never hits Hugging Face.
#
#   docker build -t costguard .
#   docker run --rm -p 8000:8000 costguard                                         # mock backend, no keys
#   docker run --rm -p 8000:8000 -e COSTGUARD_BACKEND=anthropic \
#       -e COSTGUARD_ANTHROPIC_API_KEY costguard                                    # key from your shell, never baked in
#   docker build --target dashboard -t costguard-dashboard .                        # Streamlit dashboard image

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/app/models/fastembed \
    HF_HOME=/app/models/hf \
    COSTGUARD_BACKEND=mock \
    PORT=8000

WORKDIR /app

# 1) dependencies (own layer: only pyproject.toml invalidates it). Read from pyproject so there is one source of truth;
#    anthropic + python-dotenv are used by core (Anthropic backend, .env loading) but not yet listed there.
COPY pyproject.toml ./
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; \
print('\n'.join(d + ['anthropic', 'python-dotenv']))" > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt
# optional extras, e.g. `--build-arg EXTRA_PIP=langfuse` to enable async Langfuse tracing (off unless keys are set)
ARG EXTRA_PIP=""
RUN if [ -n "$EXTRA_PIP" ]; then pip install $EXTRA_PIP; fi

# 2) bake the embedding + rerank models into the image (cache tier, router classifier, context reranker)
ARG EMBED_MODELS="BAAI/bge-small-en-v1.5"
ARG RERANK_MODELS="Xenova/ms-marco-MiniLM-L-6-v2"
RUN python -c "import os; \
from fastembed import TextEmbedding; \
[TextEmbedding(m, cache_dir=os.environ['FASTEMBED_CACHE_PATH']) for m in os.environ['EMBED_MODELS'].split(',') if m]; \
print('embedding models ready:', os.environ['EMBED_MODELS'])" \
 && python -c "import os; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
[TextCrossEncoder(m, cache_dir=os.environ['FASTEMBED_CACHE_PATH']) for m in os.environ['RERANK_MODELS'].split(',') if m]; \
print('rerank models ready:', os.environ['RERANK_MODELS'])"

# 3) application code (configs + package; eval data, logs and secrets are excluded by .dockerignore)
COPY costguard ./costguard
COPY configs ./configs
COPY eval/__init__.py ./eval/__init__.py

RUN useradd --create-home --uid 10001 costguard \
 && mkdir -p /app/data/runtime \
 && chown -R costguard:costguard /app/data /app/models
USER costguard

# ---------------------------------------------------------------- dashboard (docker build --target dashboard)
FROM base AS dashboard
USER root
RUN pip install streamlit
COPY dashboard ./dashboard
USER costguard
EXPOSE 8501
CMD exec streamlit run dashboard/app.py --server.headless true --server.address 0.0.0.0 \
    --server.port ${PORT:-8501} --browser.gatherUsageStats false


# ---------------------------------------------------------------- proxy (last stage = default target of `docker build .`)
FROM base AS proxy
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)" || exit 1
# shell form so ${PORT} (set by Render/Railway) expands; exec so uvicorn is PID 1 and receives SIGTERM
CMD exec uvicorn costguard.server:get_app --factory --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips="*"
