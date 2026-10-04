"""Stage 5: model downshift router (strong -> cheap), behind an offline eval gate.

Decision order for one request (first rule that fires wins; every decision carries a human-readable reason):

  1. requested alias is "cheap"                       -> cheap   "requested-cheap"      (honour the client)
  2. any hardness signal (features.py)                 -> strong  "hard:<signal>"        (all policies)
  3. policy "aggressive"                               -> cheap   "aggressive:easy"
  4. policy "gated":
       no / unreadable / dry-run gate file             -> strong  "gated:no-gate-file" | "gated:bad-gate-file" | "gated:dry-run-gate"
       category given but not one of the 8             -> strong  "gated:unknown-category"
       no category and the classifier is unsure        -> strong  "gated:unclassified"
         (unsure = below the similarity floor, margin to the runner-up < 0.03, or keyword-only inference)
       category not in the gate file                   -> strong  "gated:<cat>-not-evaluated"
       gate says allow                                 -> cheap   "gated:<cat>-allowed"
         ... with "rollout": "shadow"                 -> strong  "gated:<cat>-shadow"   (would have downshifted)
         ... with "rollout": "canary:0.05"            -> cheap   "gated:<cat>-canary" for a stable 5% of queries,
                                                         strong  "gated:<cat>-holdout" for the rest
       gate says block                                 -> strong  "gated:<cat>-blocked"

Anything unexpected therefore ends on strong. The gate file (configs/router_gate.json, written by
`python -m eval.gate_router`) is re-read whenever its mtime changes, so rollback is "edit or restore the file":
no restart and no deploy.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..config import ROOT, Policy, Settings
from ..schemas import RouteDecision, RouteInput
from . import features
from .classifier import CATEGORIES, GATED_MIN_MARGIN, CategoryClassifier

log = logging.getLogger("costguard.router")

DEFAULT_GATE_PATH = ROOT / "configs" / "router_gate.json"
POLICIES = ("gated", "aggressive")


class GateFile:
    """configs/router_gate.json, reloaded on mtime change. `data` is None when missing or unreadable."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._mtime: Optional[float] = None
        self._lock = threading.Lock()
        self.data: Optional[dict] = None
        self.error: Optional[str] = None

    def get(self) -> Optional[dict]:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self.data, self.error, self._mtime = None, "missing", None
            return None
        if mtime != self._mtime:
            with self._lock:
                if mtime != self._mtime:
                    try:
                        d = json.loads(self.path.read_text())
                        if not isinstance(d, dict) or not isinstance(d.get("categories"), dict):
                            raise ValueError("expected an object with a 'categories' object")
                        self.data, self.error = d, None
                    except Exception as e:
                        log.warning("router gate file %s unreadable (%s); staying on strong", self.path, e)
                        self.data, self.error = None, f"bad: {e}"
                    self._mtime = mtime
        return self.data

    def allowed(self) -> list[str]:
        d = self.get() or {}
        if d.get("dry_run"):
            return []
        return sorted(c for c, v in d.get("categories", {}).items() if isinstance(v, dict) and v.get("allow") is True)


class GatedRouter:
    """Satisfies `costguard.interfaces.Router`. `last_ms` is the wall time of the last `route()` call; `last` holds
    the full explanation (category, its source, hardness reasons) for debugging and the dashboard."""

    def __init__(self, gate_path: Path = DEFAULT_GATE_PATH, classifier: Optional[CategoryClassifier] = None,
                 hardness: Optional[features.HardnessConfig] = None, expect: Optional[dict] = None,
                 min_margin: float = GATED_MIN_MARGIN):
        self.gate = GateFile(gate_path)
        self.min_margin = min_margin        # inferred categories must clear this margin before the gate is consulted
        self.classifier = classifier or CategoryClassifier()
        self.hardness = hardness or features.HardnessConfig()
        self.expect = expect or {}          # {"backend":..., "models": {...}} of the serving engine, for a mismatch warning
        self.last_ms: float = 0.0
        self.last: dict[str, Any] = {}
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ main entry
    def route(self, inp: RouteInput, policy: str) -> RouteDecision:
        t0 = time.perf_counter()
        info: dict[str, Any] = {"policy": policy}
        try:
            alias, reason = self._decide(inp, policy, info)
        finally:
            self.last_ms = round((time.perf_counter() - t0) * 1000, 3)
        info.update(alias=alias, reason=reason, ms=self.last_ms)
        self.last = info
        return RouteDecision(alias=alias, reason=reason, category=info.get("category"),
                             signals=list(info.get("hard_reasons") or []))

    def _decide(self, inp: RouteInput, policy: str, info: dict) -> tuple[str, str]:
        if inp.requested_alias == "cheap":
            return "cheap", "requested-cheap"
        hard = features.from_route_input(inp, self.hardness)
        info["hard"], info["hard_reasons"] = hard.hard, hard.reasons
        if hard.hard:
            return "strong", f"hard:{hard.primary}"
        if policy == "aggressive":
            return "cheap", "aggressive:easy"
        if policy != "gated":
            return "strong", f"unknown-policy:{policy}"

        gate = self.gate.get()
        if gate is None:
            return "strong", "gated:no-gate-file" if self.gate.error in (None, "missing") else "gated:bad-gate-file"
        if not self._check_gate_matches(gate):
            return "strong", "gated:gate-for-other-models"
        if gate.get("dry_run"):
            return "strong", "gated:dry-run-gate"

        cat, source = self._category(inp, info)
        if cat is None:
            return "strong", "gated:unknown-category" if source == "request" else "gated:unclassified"
        entry = gate["categories"].get(cat)
        if not isinstance(entry, dict):
            return "strong", f"gated:{cat}-not-evaluated"
        if entry.get("allow") is not True:
            return "strong", f"gated:{cat}-blocked"
        rollout = str(entry.get("rollout") or "full").strip().lower()
        if rollout == "full":
            return "cheap", f"gated:{cat}-allowed"
        if rollout == "shadow":
            return "strong", f"gated:{cat}-shadow"
        if rollout.startswith("canary:"):
            share = _float(rollout.split(":", 1)[1])
            return ("cheap", f"gated:{cat}-canary") if _bucket(inp.query) < share else ("strong", f"gated:{cat}-holdout")
        return "strong", f"gated:{cat}-bad-rollout"

    def _category(self, inp: RouteInput, info: dict) -> tuple[Optional[str], str]:
        if inp.category is not None and str(inp.category).strip():
            c = str(inp.category).strip().lower()
            info["category"], info["category_source"] = (c if c in CATEGORIES else None), "request"
            return (c if c in CATEGORIES else None), "request"
        r = self.classifier.classify(inp.query)
        ok = r.confident(self.min_margin)
        info.update(category=r.category if ok else None, category_guess=r.category, category_source=r.source,
                    category_confidence=round(r.confidence, 4), category_margin=round(r.margin, 4))
        return (r.category if ok else None), r.source

    def _check_gate_matches(self, gate: dict) -> bool:
        """True if the gate may be applied to the model pair being served.

        A gate measured on one (backend, strong, cheap) pair is evidence about that pair only, so a mismatch fails
        safe: every request goes to the strong tier until `python -m eval.gate_router` is re-run for the new pair.
        The mock backend has no quality to protect, so there a mismatch only warns (keeps demos and CI routing)."""
        if not self.expect:
            return True
        g = (gate.get("backend"), (gate.get("models") or {}).get("strong"), (gate.get("models") or {}).get("cheap"))
        e = (self.expect.get("backend"), self.expect.get("models", {}).get("strong"), self.expect.get("models", {}).get("cheap"))
        if g[0] is None or g == e:
            return True
        blocking = e[0] != "mock"
        key = repr((g, e))
        if key not in self._warned:
            self._warned.add(key)
            log.warning("router gate %s was computed for %s but serving %s; %s. Re-run `python -m eval.gate_router`",
                        self.gate.path, g, e, "routing everything to strong" if blocking else "applying it anyway (mock)")
        return not blocking

    # ------------------------------------------------------------------ introspection
    def status(self) -> dict:
        g = self.gate.get() or {}
        return {"gate_file": str(self.gate.path), "gate_present": bool(g), "gate_error": self.gate.error,
                "dry_run": bool(g.get("dry_run")), "allowed": self.gate.allowed(), "judge": g.get("judge"),
                "created": g.get("created"), "classifier": self.classifier.backend}


def _bucket(query: str) -> float:
    """Stable [0, 1) bucket per normalised query: the same question always lands in the same canary arm."""
    h = hashlib.sha256(" ".join((query or "").lower().split()).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def _float(x: str) -> float:
    try:
        return min(1.0, max(0.0, float(x)))
    except ValueError:
        return 0.0


def gate_path_for(policy: Optional[Policy] = None) -> Path:
    """COSTGUARD_ROUTER_GATE env > policy.yaml `router.gate_file` > configs/router_gate.json."""
    if os.environ.get("COSTGUARD_ROUTER_GATE"):
        return Path(os.environ["COSTGUARD_ROUTER_GATE"])
    cfg = ((policy.raw if policy else {}) or {}).get("router") or {}
    if cfg.get("gate_file"):
        p = Path(cfg["gate_file"])
        return p if p.is_absolute() else ROOT / p
    return DEFAULT_GATE_PATH


def build_router(settings: Settings, policy: Policy) -> GatedRouter:
    """Factory entry point (costguard/factory.py). Optional policy.yaml section:

        router:
          gate_file: configs/router_gate.json
          classifier: embedding        # or "keywords" (no model load; gated then downshifts only explicit categories)
          inferred_min_margin: 0.03    # confidence margin an inferred category needs before the gate is consulted
          hardness: {long_query_tokens: 120, long_input_tokens: 4000, max_history_turns: 4, max_context_docs: 8}
    """
    cfg = (policy.raw or {}).get("router") or {}
    use_emb = cfg.get("classifier", "embedding") != "keywords" and os.environ.get("COSTGUARD_ROUTER_EMBEDDINGS", "1") != "0"
    clf = CategoryClassifier(use_embeddings=use_emb).warmup()   # pay the model load at startup, not on a request
    expect = {}
    try:
        expect = {"backend": settings.backend, "models": dict(policy.backends[settings.backend])}
    except Exception:
        pass
    return GatedRouter(gate_path_for(policy), clf, features.HardnessConfig.from_dict(cfg.get("hardness")), expect,
                       float(cfg.get("inferred_min_margin", GATED_MIN_MARGIN)))
