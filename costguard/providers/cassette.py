"""Record/replay wrapper. Every upstream call is stored under a hash of (model, messages, max_tokens, temperature),
so re-running an A/B arm, a threshold sweep or CI costs $0 and is exactly reproducible."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from ..schemas import ChatMessage, Completion


def call_key(messages: list[ChatMessage], model: str, max_tokens: int, temperature: float) -> str:
    payload = json.dumps({"m": model, "msgs": [[x.role, x.content] for x in messages], "max": max_tokens,
                          "t": round(float(temperature), 3)}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class CassetteMiss(KeyError):
    pass


class CassetteProvider:
    """mode: 'replay' (miss -> error), 'record' (always call + append), 'auto' (replay if present else call + append)."""

    def __init__(self, inner, path: Path, mode: str = "auto"):
        self.inner, self.path, self.mode = inner, Path(path), mode
        self.name = f"{inner.name}+cassette"
        self._lock = threading.Lock()
        self._store: dict[str, dict] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._store[rec["key"]] = rec["completion"]
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def count_tokens(self, messages, model):
        return self.inner.count_tokens(messages, model)

    def complete(self, messages, model, max_tokens, temperature) -> Completion:
        """The returned completion's raw["cassette"] says "replay" or "new" for THIS call; callers running in threads
        must use it rather than diffing the shared hits/misses counters."""
        key = call_key(messages, model, max_tokens, temperature)
        with self._lock:
            rec = self._store.get(key) if self.mode != "record" else None
            if rec is not None:
                self.hits += 1
            else:
                self.misses += 1
        if rec is not None:
            return _tagged(Completion(**rec), "replay")
        if self.mode == "replay":
            raise CassetteMiss(f"no cassette entry for model={model} (key {key[:12]}) in {self.path}")
        comp = self.inner.complete(messages, model, max_tokens, temperature)
        with self._lock:
            self._store[key] = comp.model_dump()
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "model": model, "completion": comp.model_dump()}, ensure_ascii=False) + "\n")
        return _tagged(comp, "new")


def _tagged(comp: Completion, status: str) -> Completion:
    """Copy with the per-call replay status in raw (never written to the cassette file)."""
    return comp.model_copy(update={"raw": {**(comp.raw or {}), "cassette": status}})
