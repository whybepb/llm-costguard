"""Stage contracts. Each optimisation lives behind one of these Protocols, so it can be developed, tested
and A/B-switched independently. The no-op defaults let the pipeline run end to end before any stage exists."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .schemas import (CacheEntry, ChatMessage, Completion, CompressResult, ContextResult, RouteDecision,
                      RouteInput, SemanticHit)


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, messages: list[ChatMessage], model: str, max_tokens: int, temperature: float) -> Completion: ...

    def count_tokens(self, messages: list[ChatMessage], model: str) -> int: ...


@runtime_checkable
class ExactCache(Protocol):
    def get(self, key: str) -> Optional[CacheEntry]: ...

    def put(self, key: str, entry: CacheEntry) -> None: ...

    def clear(self) -> None: ...


@runtime_checkable
class SemanticCache(Protocol):
    def lookup(self, query: str, partition: str, threshold: float) -> SemanticHit: ...

    def insert(self, query: str, partition: str, entry: CacheEntry) -> None: ...

    def clear(self) -> None: ...


@runtime_checkable
class ContextOptimizer(Protocol):
    def optimize(self, query: str, docs: list[str], budget_tokens: int, min_score: Optional[float]) -> ContextResult: ...


@runtime_checkable
class Compressor(Protocol):
    def compress(self, text: str, rate: float, query: Optional[str] = None) -> CompressResult: ...


@runtime_checkable
class Router(Protocol):
    def route(self, inp: RouteInput, policy: str) -> RouteDecision: ...


# ---------------------------------------------------------------- no-op defaults

class NoExactCache:
    def get(self, key):
        return None

    def put(self, key, entry):
        pass

    def clear(self):
        pass


class NoSemanticCache:
    def lookup(self, query, partition, threshold):
        return SemanticHit()

    def insert(self, query, partition, entry):
        pass

    def clear(self):
        pass


class PassthroughContext:
    def optimize(self, query, docs, budget_tokens, min_score):
        return ContextResult(docs=list(docs), kept_indices=list(range(len(docs))), note="passthrough")


class NoCompressor:
    def compress(self, text, rate, query=None):
        n = len(text.split())
        return CompressResult(text=text, tokens_before=n, tokens_after=n, method="none")


class AlwaysRequestedRouter:
    def route(self, inp, policy):
        return RouteDecision(alias=inp.requested_alias, reason="router-off")
