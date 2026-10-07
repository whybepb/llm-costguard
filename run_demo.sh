#!/usr/bin/env bash
# Start the CostGuard demo: Redis Stack, Prometheus and Grafana in Docker, the app on the host.
#   Demo page   http://localhost:8000
#   Grafana     http://localhost:3000   (dashboard "LLM CostGuard")
#   Prometheus  http://localhost:9090
#   RedisInsight http://localhost:8001
# Stop: Ctrl-C for the app, then `docker compose down` (add -v to also wipe cache and metrics history).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "missing .env with ANTHROPIC_API_KEY"; exit 1; }
set -a; . ./.env; set +a
docker compose up -d
until docker compose exec -T redis redis-cli ping >/dev/null 2>&1; do sleep 1; done
export HF_HUB_OFFLINE=1 REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
exec .venv/bin/uvicorn service.app:app --host 127.0.0.1 --port 8000
