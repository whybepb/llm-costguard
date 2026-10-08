#!/usr/bin/env bash
# Start the CostGuard demo: Redis Stack, Prometheus and Grafana in Docker, the app on the host.
#   Demo page   http://localhost:8000
#   Grafana     http://localhost:3000   (dashboard "LLM CostGuard")
#   Prometheus  http://localhost:9090
#   Langfuse    http://localhost:3001   (per-request traces; login in .env, only if LANGFUSE_* is set)
#   RedisInsight http://localhost:8001
# Stop: Ctrl-C for the app, then `docker compose --profile langfuse down` (add -v to also wipe cache,
# metrics history and traces).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "missing .env with ANTHROPIC_API_KEY (see .env.example)"; exit 1; }
set -a; . ./.env; set +a
if [ -n "${LANGFUSE_PUBLIC_KEY:-}" ]; then docker compose --profile langfuse up -d; else docker compose up -d; fi
until docker compose exec -T redis redis-cli ping >/dev/null 2>&1; do sleep 1; done
export HF_HUB_OFFLINE=1 REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
exec .venv/bin/uvicorn service.app:app --host 127.0.0.1 --port 8000
