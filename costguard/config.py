"""Settings (from environment) + versioned policy (from configs/policy.yaml)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    backend: str = "mock"                     # mock | mlx | openai | gemini
    policy_path: Path = ROOT / "configs" / "policy.yaml"
    prices_path: Path = ROOT / "configs" / "prices.yaml"
    db_path: Path = ROOT / "data" / "runtime" / "costguard.sqlite"
    cassette: Optional[Path] = None           # record/replay upstream calls (eval + CI)
    cassette_mode: str = "auto"               # auto | record | replay
    mock_latency_ms: float = 0.0
    semantic_backend: str = "memory"          # memory | qdrant
    qdrant_url: Optional[str] = None
    langfuse: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        try:  # repo-local .env (gitignored) for keys; never overrides real env vars
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            pass
        e = os.environ.get
        s = cls()
        s.backend = e("COSTGUARD_BACKEND", s.backend)
        if e("COSTGUARD_POLICY"):
            s.policy_path = Path(e("COSTGUARD_POLICY"))
        if e("COSTGUARD_DB"):
            s.db_path = Path(e("COSTGUARD_DB"))
        if e("COSTGUARD_CASSETTE"):
            s.cassette = Path(e("COSTGUARD_CASSETTE"))
        s.cassette_mode = e("COSTGUARD_CASSETTE_MODE", s.cassette_mode)
        s.mock_latency_ms = float(e("COSTGUARD_MOCK_LATENCY_MS", s.mock_latency_ms))
        s.semantic_backend = e("COSTGUARD_SEMANTIC_BACKEND", s.semantic_backend)
        s.qdrant_url = e("QDRANT_URL")
        s.langfuse = bool(e("LANGFUSE_PUBLIC_KEY"))
        return s


@dataclass
class ModePolicy:
    exact_cache: bool = False
    semantic_cache: bool = False
    tau: float = 0.95
    context: bool = False
    context_budget_tokens: int = 2000
    context_min_score: Optional[float] = None
    compression: bool = False
    compression_rate: float = 0.5
    compression_min_tokens: int = 400
    router: bool = False
    router_policy: str = "gated"


@dataclass
class Policy:
    raw: dict[str, Any]
    default_mode: str
    kb_version: str
    system_prompt: str
    default_max_tokens: int
    backends: dict[str, dict[str, str]]
    billing: dict[str, str]
    cache: dict[str, Any]
    modes: dict[str, ModePolicy] = field(default_factory=dict)
    config_hash: str = ""

    def mode(self, name: Optional[str]) -> tuple[str, ModePolicy]:
        n = name or self.default_mode
        if n not in self.modes:
            raise ValueError(f"unknown mode {n!r}; known: {sorted(self.modes)}")
        return n, self.modes[n]

    def model_id(self, backend: str, alias: str) -> str:
        return self.backends[backend][alias]

    def billing_for(self, backend: str) -> dict[str, str]:
        """Real APIs bill at their own prices; local/mock backends at the list price of the model they stand in for."""
        return {**self.billing, **self.raw.get("billing_by_backend", {}).get(backend, {})}


def load_policy(path: Path) -> Policy:
    raw = yaml.safe_load(Path(path).read_text())
    sp_path = ROOT / raw["system_prompt_file"]
    system_prompt = sp_path.read_text().strip()
    modes = {name: ModePolicy(**{k: v for k, v in m.items()}) for name, m in raw["modes"].items()}
    canon = json.dumps({"policy": raw, "system_prompt": system_prompt}, sort_keys=True)
    h = hashlib.sha256(canon.encode()).hexdigest()[:12]
    return Policy(raw=raw, default_mode=raw["default_mode"], kb_version=raw["kb_version"], system_prompt=system_prompt,
                  default_max_tokens=int(raw.get("default_max_tokens", 256)), backends=raw["backends"],
                  billing=raw["billing"], cache=raw.get("cache", {}), modes=modes, config_hash=h)
