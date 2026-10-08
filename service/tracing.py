"""Langfuse tracing for the live pipeline: one trace per request, one observation per stage.

    costguard.answer (chain)            request in, answer + cost out; scores: savings_pct, quality
      exact_cache (retriever)           key, hit or miss
      semantic_cache (retriever)        every KNN candidate: cosine, cached query, verdict
        cache_judge (generation)        Haiku 4.5 SAME/DIFFERENT, only when it ran
      window (span)                     sections kept (with text) and dropped, tokens before/after
      route (span)                      features, P(large), threshold, model
      answer (generation)               the exact prompt sent, the answer, tokens, cost
      cache_store (span)                what was written and its depends_on
      shadow_baseline (generation)      Sonnet 5.5 on the full context (shadow mode)
      quality_judge (evaluator)         optimized vs baseline verdict (shadow mode)

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set; mock-mode requests (load
tests) are never traced. Spans are batched and sent in the background, so tracing adds no
network wait to a request.
"""

import os
from contextlib import contextmanager, nullcontext

from config import price


class _Off:
    """Stands in for a span when tracing is off: every call is a no-op."""
    trace_id = None

    def update(self, **_):
        return self

    def end(self, **_):
        return self

    def score_trace(self, **_):
        pass

    def set_trace_io(self, **_):
        pass

    def start_observation(self, **_):
        return self


OFF = _Off()


def llm_fields(model, input_tokens, output_tokens):
    """model, usage and cost of one LLM call, as Langfuse generation fields."""
    pin, pout = price(model)
    cin, cout = input_tokens * pin / 1e6, output_tokens * pout / 1e6
    return {"model": model, "usage_details": {"input": input_tokens, "output": output_tokens},
            "cost_details": {"input": cin, "output": cout, "total": cin + cout}}


class Tracer:
    def __init__(self):
        self.lf = None
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            from langfuse import Langfuse
            self.lf = Langfuse(public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                               secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                               base_url=os.environ.get("LANGFUSE_HOST", "http://localhost:3001"))

    @contextmanager
    def request(self, llm_mode, tenant, query, metadata):
        if self.lf is None or llm_mode == "mock":
            yield OFF
            return
        from langfuse import propagate_attributes
        with propagate_attributes(trace_name="costguard.answer", user_id=tenant, tags=[llm_mode],
                                  metadata={k: str(v) for k, v in metadata.items()}):
            with self.lf.start_as_current_observation(name="costguard.answer", as_type="chain",
                                                      input={"query": query}) as root:
                yield root

    @staticmethod
    def stage(parent, name, as_type="span", **fields):
        """A child observation of parent, as a context manager (a no-op one when tracing is off)."""
        if parent is OFF:
            return nullcontext(OFF)
        return parent.start_as_current_observation(name=name, as_type=as_type, **fields)

    def url(self, root):
        if self.lf is None or root is OFF:
            return None
        return self.lf.get_trace_url(trace_id=root.trace_id)

    def flush(self):
        if self.lf is not None:
            self.lf.flush()
