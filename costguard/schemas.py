"""Shared data types. Every stage and tool speaks these; change them only with the whole team."""
from __future__ import annotations

import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]
Mode = Literal["off", "quality", "balanced", "economy"]
CacheStatus = Literal["miss", "exact", "semantic", "bypass", "disabled"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class CostGuardOptions(BaseModel):
    """CostGuard-specific request extension (an OpenAI client can send it via `extra_body`)."""

    mode: Optional[Mode] = None                       # None -> policy default_mode
    tenant: str = "default"                           # cache partition / quota key
    category: Optional[str] = None                    # task category hint, used by the router and its eval gate
    context: list[str] = Field(default_factory=list)  # retrieved documents (RAG). Optimiser may rerank/trim/compress
    no_cache: bool = False                            # force a fresh answer
    arm: Optional[str] = None                         # A/B arm label (logging only)
    trace_pos: Optional[int] = None                   # position in a replayed trace (logging only)
    item_id: Optional[str] = None                     # eval/trace item id (logging only)


class ChatRequest(BaseModel):
    """OpenAI chat.completions subset + `costguard` extension."""

    model: Optional[str] = None            # tier alias ("strong"/"cheap") or None -> "strong"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    user: Optional[str] = None
    costguard: CostGuardOptions = Field(default_factory=CostGuardOptions)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0     # prompt-cache reads (billed at the cached price)
    cache_write_tokens: int = 0      # prompt-cache writes (billed at the cache-write price; Anthropic)


class Completion(BaseModel):
    """What a provider returns for one upstream call."""

    text: str
    model: str                 # concrete model id actually called
    usage: Usage
    latency_ms: float
    finish_reason: str = "stop"
    raw: dict[str, Any] = Field(default_factory=dict)


class CacheEntry(BaseModel):
    entry_id: str
    query_text: str
    response_text: str
    model_alias: str           # tier that produced the answer
    input_tokens: int          # tokens of the prompt that produced it (for baseline-cost accounting)
    output_tokens: int
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticHit(BaseModel):
    entry: Optional[CacheEntry] = None      # None -> miss
    similarity: Optional[float] = None      # best cosine similarity found (even on a miss)
    neighbor_query: Optional[str] = None    # the cached query that matched / came closest
    guard_rejected: Optional[str] = None    # reason a near-hit was refused by a guard (e.g. "entity_mismatch")


class ContextResult(BaseModel):
    docs: list[str]                         # documents to send, in final order
    kept_indices: list[int] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    note: str = ""


class CompressResult(BaseModel):
    text: str
    tokens_before: int
    tokens_after: int
    method: str = "none"


class RouteInput(BaseModel):
    query: str
    category: Optional[str] = None
    input_tokens: int = 0
    has_context: bool = False
    context_docs: int = 0
    history_turns: int = 0
    requested_alias: str = "strong"


class RouteDecision(BaseModel):
    alias: str                              # "strong" | "cheap"
    reason: str
    category: Optional[str] = None          # category the router used (given or inferred)
    signals: list[str] = Field(default_factory=list)   # hardness signals that fired


class TraceRecord(BaseModel):
    """One row per request: the single source of truth for the dashboard, README numbers and A/B analysis."""

    request_id: str
    ts: float
    arm: Optional[str] = None
    trace_pos: Optional[int] = None
    item_id: Optional[str] = None
    tenant: str = "default"
    mode: str = "balanced"
    category: Optional[str] = None
    config_hash: str = ""
    provider: str = ""
    cache_status: CacheStatus = "miss"
    cache_similarity: Optional[float] = None
    cache_neighbor: Optional[str] = None
    cache_guard: Optional[str] = None
    model_requested: str = "strong"
    model_used: str = "strong"             # tier alias actually used ("" on cache hit)
    model_id: str = ""                     # concrete model id called
    route_reason: str = ""
    route_category: Optional[str] = None   # category used by the router (given or inferred)
    route_signals: list[str] = Field(default_factory=list)
    input_tokens_original: int = 0         # full prompt (system + history + all context + question), before optimisation
    input_tokens_sent: int = 0             # what was actually billed upstream (0 on cache hit)
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    context_docs_in: int = 0
    context_docs_kept: int = 0
    compression_ratio: Optional[float] = None
    cost_usd: float = 0.0                  # actual list-price cost of this request (incl. overhead)
    baseline_cost_usd: float = 0.0         # same request on the strong tier, full prompt, no cache
    saved_usd: float = 0.0
    overhead_cost_usd: float = 0.0         # e.g. paid embedding calls; 0 for local models
    latency_ms: float = 0.0                # end-to-end inside CostGuard
    upstream_latency_ms: float = 0.0
    overhead_ms: float = 0.0               # latency added by CostGuard stages
    stage_ms: dict[str, float] = Field(default_factory=dict)
    stage_errors: dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
    query: str = ""
    response_text: str = ""
