"""Cost accounting: one formula, one price table (configs/prices.yaml).

cost = input_price * uncached_input + cached_price * cache_reads + cache_write_price * cache_writes + output_price * output
(USD per 1M tokens)
Tier aliases ("strong"/"cheap") are billed at the list price of the model they stand for (policy.billing), so
free/local runs still report list-price-equivalent dollars.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cached_input: float
    cache_write: float


class PriceBook:
    def __init__(self, path: Path, billing: dict[str, str]):
        raw = yaml.safe_load(Path(path).read_text())
        self.checked_on = raw.get("checked_on")
        self.prices = {k: Price(float(v["input"]), float(v["output"]), float(v.get("cached_input", v["input"])),
                                float(v.get("cache_write", float(v["input"]) * 1.25)))
                       for k, v in raw["models"].items()}
        self.billing = billing

    def price_for(self, alias_or_model: str) -> Price:
        key = self.billing.get(alias_or_model, alias_or_model)
        if key not in self.prices:
            raise KeyError(f"no price for {alias_or_model!r} (billing key {key!r}); add it to configs/prices.yaml")
        return self.prices[key]

    def cost(self, alias_or_model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
             cache_write_tokens: int = 0) -> float:
        """input_tokens = ALL prompt tokens (incl. cache reads/writes); those subsets are re-priced."""
        p = self.price_for(alias_or_model)
        uncached = max(0, input_tokens - cached_input_tokens - cache_write_tokens)
        return (p.input * uncached + p.cached_input * cached_input_tokens + p.cache_write * cache_write_tokens
                + p.output * output_tokens) / 1_000_000
