# Hugging Face Space (Docker SDK) image for the CostGuard Streamlit dashboard. It reads the committed results in
# eval/results/*.json and a request log shipped with the Space (COSTGUARD_DB); COSTGUARD_URL points the live drift
# panel at the gateway Space. No models and no secrets are needed.
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COSTGUARD_DB=/home/user/app/eval/results/ab_anthropic.sqlite \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user pyproject.toml ./
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; print('\n'.join(d))" \
      > /tmp/requirements.txt \
 && pip install --user -r /tmp/requirements.txt streamlit altair

COPY --chown=user costguard ./costguard
COPY --chown=user configs ./configs
COPY --chown=user dashboard ./dashboard
COPY --chown=user eval ./eval

EXPOSE 7860
CMD exec streamlit run dashboard/app.py --server.port ${PORT:-7860} --server.address 0.0.0.0 --server.headless true \
    --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false
