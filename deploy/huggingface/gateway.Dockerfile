# Hugging Face Space (Docker SDK) image for the CostGuard gateway. Same build as the repo Dockerfile's proxy target,
# but runs as uid 1000 and listens on 7860, as Spaces require. Config comes from Space variables and secrets:
#   COSTGUARD_BACKEND, COSTGUARD_SPEND_CAP_USD   (variables)
#   COSTGUARD_API_KEYS, COSTGUARD_ADMIN_TOKEN, COSTGUARD_ANTHROPIC_API_KEY   (secrets, never baked into the image)
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/home/user/app/models/fastembed \
    HF_HOME=/home/user/app/models/hf \
    COSTGUARD_BACKEND=mock \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user pyproject.toml ./
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; print('\n'.join(d))" \
      > /tmp/requirements.txt \
 && pip install --user -r /tmp/requirements.txt

# bake the embedding + rerank models into the image, so a cold start never downloads them
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir=os.environ['FASTEMBED_CACHE_PATH'])" \
 && python -c "import os; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir=os.environ['FASTEMBED_CACHE_PATH'])"

COPY --chown=user costguard ./costguard
COPY --chown=user configs ./configs
COPY --chown=user eval/__init__.py ./eval/__init__.py
RUN mkdir -p data/runtime

EXPOSE 7860
CMD exec uvicorn costguard.server:get_app --factory --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips="*"
