"""Deploy the gateway and the dashboard as two Hugging Face Spaces (Docker SDK).

    hf auth login                                   # once, with a WRITE token (or: python -c "from huggingface_hub import login; login()")
    python deploy/huggingface/push_spaces.py --backend mock
    python deploy/huggingface/push_spaces.py --backend anthropic --spend-cap 3   # key from COSTGUARD_ANTHROPIC_API_KEY / .env

What it does:
  1. stages only the files each image needs (no .env, no logs, no cassettes) plus a Space README;
  2. creates or updates <owner>/<name>-gateway and <owner>/<name>-dashboard;
  3. sets Space variables (backend, spend cap, gateway URL) and secrets (caller keys, admin token, provider key).
Generated caller keys and the admin token are written to --secrets-out (outside the repo), never printed.
"""
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

GATEWAY_README = """---
title: LLM CostGuard Gateway
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: OpenAI-compatible gateway that cuts LLM cost, eval-gated
---

# LLM CostGuard: live gateway

OpenAI-compatible endpoint: `POST /v1/chat/completions` with `Authorization: Bearer <caller key>`.
Each response carries `x-costguard-*` headers (cache status, similarity, route, cost, baseline cost, savings).

- `GET /health`: status and the live component at each stage
- `GET /v1/stats`: hit rate, savings and latency percentiles
- `GET /metrics`: Prometheus `costguard_*` series
- `GET /v1/drift`: PSI drift against a frozen baseline

Backend: `{backend}`. Code, docs and measured results: https://github.com/whybepb/llm-costguard
"""

DASHBOARD_README = """---
title: LLM CostGuard Dashboard
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Savings, cache, router and quality dashboard for CostGuard
---

# LLM CostGuard: observability dashboard

Streamlit dashboard over the measured runs: savings waterfall, A/B arms with confidence intervals, cache hit and
false-hit rates, router gate, latency, drift and a request explorer. The request log is the real Claude
(Sonnet 5.5 / Haiku 4.5) A/B run; the local Qwen run is selectable in the sidebar. Live drift reads the gateway at
`{gateway_url}`.

Code, docs and measured results: https://github.com/whybepb/llm-costguard
"""


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def stage_gateway(tmp: Path, backend: str) -> Path:
    d = tmp / "gateway"
    for rel in ("pyproject.toml", "costguard", "configs", "eval/__init__.py"):
        _copy(ROOT / rel, d / rel)
    shutil.copy2(HERE / "gateway.Dockerfile", d / "Dockerfile")
    (d / "README.md").write_text(GATEWAY_README.format(backend=backend))
    return d


def stage_dashboard(tmp: Path, gateway_url: str) -> Path:
    d = tmp / "dashboard"
    for rel in ("pyproject.toml", "costguard", "configs", "dashboard", "eval/__init__.py"):
        _copy(ROOT / rel, d / rel)
    res = ROOT / "eval" / "results"
    for p in list(res.glob("*.json")) + [res / "ab_anthropic.sqlite", res / "ab_mlx.sqlite"]:
        if p.exists():
            _copy(p, d / "eval" / "results" / p.name)
    shutil.copy2(HERE / "dashboard.Dockerfile", d / "Dockerfile")
    (d / "README.md").write_text(DASHBOARD_README.format(gateway_url=gateway_url))
    return d


def leak_check(folder: Path) -> None:
    """Refuse to upload if a staged file holds a key-shaped string or a local home path."""
    bad = []
    for p in folder.rglob("*"):
        if p.is_file():
            data = p.read_bytes()
            if b"sk-ant-" in data.replace(b"sk-ant-test", b"") or b"/Users/" in data:
                bad.append(str(p.relative_to(folder)))
    if bad:
        sys.exit(f"refusing to upload, possible secret or local path in: {bad}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", default=None, help="HF user or org (default: the logged-in user)")
    ap.add_argument("--name", default="llm-costguard", help="Space name prefix")
    ap.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--spend-cap", type=float, default=3.0, help="hard USD cap for real provider calls")
    ap.add_argument("--tenant", default="default", help="tenant the generated caller key maps to (default: balanced, may override mode)")
    ap.add_argument("--secrets-out", default=str(ROOT.parent / "review" / "deploy_secrets.md"),
                    help="where to write the generated caller key and admin token (keep it outside the repo)")
    ap.add_argument("--only", choices=["gateway", "dashboard"], default=None)
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi
    api = HfApi()
    owner = args.owner or api.whoami()["name"]
    gw_id, db_id = f"{owner}/{args.name}-gateway", f"{owner}/{args.name}-dashboard"
    gw_url = f"https://{owner.lower()}-{args.name}-gateway.hf.space".replace("_", "-")

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        if args.only in (None, "gateway"):
            key = None
            if args.backend == "anthropic":
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env", override=False)
                key = os.environ.get("COSTGUARD_ANTHROPIC_API_KEY")
                if not key:
                    sys.exit("set COSTGUARD_ANTHROPIC_API_KEY (shell or .env) for --backend anthropic")
            folder = stage_gateway(tmp, args.backend)
            leak_check(folder)
            api.create_repo(gw_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
            caller_key = "cg-demo-" + secrets.token_urlsafe(18)
            admin = secrets.token_urlsafe(24)
            api.add_space_variable(gw_id, "COSTGUARD_BACKEND", args.backend)
            api.add_space_variable(gw_id, "COSTGUARD_SPEND_CAP_USD", str(args.spend_cap))
            api.add_space_secret(gw_id, "COSTGUARD_API_KEYS", f"{caller_key}:{args.tenant}")
            api.add_space_secret(gw_id, "COSTGUARD_ADMIN_TOKEN", admin)
            if key:
                api.add_space_secret(gw_id, "COSTGUARD_ANTHROPIC_API_KEY", key)
            api.upload_folder(folder_path=str(folder), repo_id=gw_id, repo_type="space",
                              commit_message=f"Deploy CostGuard gateway ({args.backend})")
            out = Path(args.secrets_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"# CostGuard live deployment (private; do not commit)\n\n"
                           f"- Gateway: {gw_url}  (OpenAI base_url: {gw_url}/v1)\n"
                           f"- Backend: {args.backend}, spend cap ${args.spend_cap:g} per container lifetime\n"
                           f"- Caller key (tenant {args.tenant}): `{caller_key}`\n"
                           f"- Admin token (POST /v1/drift/baseline): `{admin}`\n")
            os.chmod(out, 0o600)
            print(f"gateway: https://huggingface.co/spaces/{gw_id}  ->  {gw_url}")
            print(f"caller key and admin token written to {out}")
        if args.only in (None, "dashboard"):
            folder = stage_dashboard(tmp, gw_url)
            leak_check(folder)
            api.create_repo(db_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
            api.add_space_variable(db_id, "COSTGUARD_URL", gw_url)
            api.upload_folder(folder_path=str(folder), repo_id=db_id, repo_type="space",
                              commit_message="Deploy CostGuard dashboard")
            db_url = f"https://{owner.lower()}-{args.name}-dashboard.hf.space".replace("_", "-")
            print(f"dashboard: https://huggingface.co/spaces/{db_id}  ->  {db_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
