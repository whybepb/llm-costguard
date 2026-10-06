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
        self._inflight: dict[str, threading.Event] = {}   # key -> set when the first caller has recorded it
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
        must use it rather than diffing the shared hits/misses counters.

        Single flight per key: when several threads miss the same key at once (two arms, or two judge passes, asking
        the identical question), one calls the model and the others wait and replay its answer. Without this, a
        non-deterministic model gives the same input two different answers (and is paid twice)."""
        key = call_key(messages, model, max_tokens, temperature)
        wait_for = None
        with self._lock:
            rec = self._store.get(key) if self.mode != "record" else None
            if rec is not None:
                self.hits += 1
            elif self.mode != "record" and key in self._inflight:
                wait_for = self._inflight[key]
            else:
                self.misses += 1
                if self.mode != "record":
                    self._inflight[key] = threading.Event()
        if rec is not None:
            return _tagged(Completion(**rec), "replay")
        if wait_for is not None:
            wait_for.wait()
            return self.complete(messages, model, max_tokens, temperature)   # replays, or retries if the first call failed
        if self.mode == "replay":
            self._release(key)
            raise CassetteMiss(f"no cassette entry for model={model} (key {key[:12]}) in {self.path}")
        try:
            comp = self.inner.complete(messages, model, max_tokens, temperature)
            with self._lock:
                self._store[key] = comp.model_dump()
                with self.path.open("a") as f:
                    f.write(json.dumps({"key": key, "model": model, "completion": comp.model_dump()},
                                       ensure_ascii=False) + "\n")
        finally:
            self._release(key)
        return _tagged(comp, "new")

    def _release(self, key: str) -> None:
        with self._lock:
            ev = self._inflight.pop(key, None)
        if ev is not None:
            ev.set()


def _tagged(comp: Completion, status: str) -> Completion:
    """Copy with the per-call replay status in raw (never written to the cassette file)."""
    return comp.model_copy(update={"raw": {**(comp.raw or {}), "cassette": status}})
