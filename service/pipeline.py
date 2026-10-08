"""The live CostGuard pipeline for one request.

exact cache (Redis) -> semantic cache (Redis vector KNN -> context check -> numbers and
negation guards -> Haiku 4.5 judge) -> on a miss: hybrid window -> router -> Haiku 5.5
or Sonnet 5.5 -> store in both caches.

Savings are measured against the baseline: Sonnet 5.5 answering with the full context.
Its input tokens come from the token-counting endpoint (free); its output is assumed
to match the optimized answer's length. In shadow mode the baseline really runs, its
real cost replaces the estimate, and Sonnet 5.5 compares the two answers.

llm_mode per request:
  live    real API calls (the demo)
  replay  answers and baseline from the e2e memo (results/e2e__api_memo.json), cache-judge
          verdicts from their memo, quality from the offline grades: a fixed trace replays
          through the real service and its metrics for free; an unmemoized call is an error
  mock    canned answers, no API at all (load testing the service itself)
"""

import csv
import hashlib
import os
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import anthropic
import numpy as np

from cache.exact import cache_key
from config import CFG, config_version, prompt as load_prompt
from cache.guards import GUARDS
from cache.judge import Judge
from demo_data.build_dataset import render_attachment, render_case
from e2e.llm import FALLBACK_BETA, LLM, cost
from e2e.run import ANSWER_SYSTEM, optimized_context, prompt
from routing.router import LARGE, SMALL, text_features
from window.manager import MIN_CONTEXT_TOKENS, approx_tokens, chunk_docs, repeated_sentences, select_hybrid
from service.store import tenant_slug
from service.tracing import Tracer, llm_fields

BASELINE_MODEL = CFG["models"]["baseline"]
QUALITY_MODEL = CFG["models"]["quality_judge"]
CACHE_JUDGE_MODEL = CFG["models"]["cache_judge"]
SEMANTIC_THRESHOLD = CFG["cache"]["semantic_threshold"]
MAX_TOKENS = CFG["llm"]["max_tokens"]
QUALITY_SYSTEM = load_prompt("shadow_quality_system")
CONFIG_VERSION = config_version()
MOCK_LATENCY_S = float(os.environ.get("COSTGUARD_MOCK_LATENCY_S", "0"))
MOCK_OUTPUT_TOKENS = 120
GRADE_RANK = {"wrong": 0, "partial": 1, "correct": 2}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def parse_doc(title, text):
    """A pasted policy doc: '## Heading' lines split it into sections."""
    sections, current, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None or buf:
                sections[current or "main"] = "\n".join(buf).strip()
            current, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    if current is not None or buf:
        sections[current or "main"] = "\n".join(buf).strip()
    sections = {k: v for k, v in sections.items() if v}
    doc_id = "custom_" + _sha(title + text)[:10]
    return {"id": doc_id, "title": title or doc_id, "sections": sections or {"main": text.strip()}}


class Pipeline:
    def __init__(self, store, embed_model, corpus, router, cache_judge_path, memo_path=None, grade_paths=()):
        self.store = store
        self.model = embed_model
        self.corpus = corpus
        self.router = router
        self.client = anthropic.Anthropic(max_retries=4) if os.environ.get("ANTHROPIC_API_KEY") else None
        self.judge = Judge(cache_judge_path)
        self.memo = None
        if memo_path and os.path.exists(memo_path):
            self.memo = LLM(memo_path)
            self.memo.client = None  # replay never calls the API
        self.grades = {}  # (query, pipeline) -> offline verdict vs the reference answer
        for path in grade_paths:
            if os.path.exists(path):
                for row in csv.DictReader(open(path)):
                    self.grades[(row["query"], row["pipeline"])] = row["verdict"]
        self.judge_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(8)
        self.tracer = Tracer()
        self.corpus_boilerplate = repeated_sentences(corpus)
        self._vec_cache = {}
        self.embed([c["text"] for d in corpus for c in chunk_docs(corpus, [{"doc_id": d}])])

    # --- helpers ---------------------------------------------------------------
    def embed(self, texts):
        texts = list(texts)
        missing = [t for t in dict.fromkeys(texts) if _sha(t) not in self._vec_cache]
        if missing:
            for t, v in zip(missing, self.model.encode(missing, normalize_embeddings=True)):
                self._vec_cache[_sha(t)] = v
        return [self._vec_cache[_sha(t)] for t in texts]

    def _llm(self, model, system, user, effort=None, mode="live"):
        if mode == "mock":
            time.sleep(MOCK_LATENCY_S)
            it = int(approx_tokens(system + " " + user) * 1.3)
            return {"text": "[mock answer]", "stop_reason": "end_turn", "model": model, "input_tokens": it,
                    "output_tokens": MOCK_OUTPUT_TOKENS, "cost_usd": cost(model, it, MOCK_OUTPUT_TOKENS)}
        if mode == "replay":
            if self.memo is None:
                raise RuntimeError("replay mode needs results/e2e__api_memo.json")
            r = self.memo.call(model, system, user, effort=effort)
            return {**r, "model": model, "cost_usd": cost(model, r["input_tokens"], r["output_tokens"])}
        if self.client is None:
            raise RuntimeError("live mode needs ANTHROPIC_API_KEY")
        kwargs = {"model": model, "max_tokens": MAX_TOKENS, "system": system,
                  "messages": [{"role": "user", "content": user}]}
        if effort:
            kwargs["output_config"] = {"effort": effort}
        if model == "claude-sonnet-5-5":
            msg = self.client.beta.messages.create(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            msg = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        return {"text": text, "stop_reason": msg.stop_reason, "model": model,
                "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
                "cost_usd": cost(model, msg.usage.input_tokens, msg.usage.output_tokens)}

    def _count_tokens(self, model, user, mode="live"):
        estimate = int(approx_tokens(ANSWER_SYSTEM + " " + user) * 1.3)  # words*1.3 undercounts the new tokenizer
        if mode == "replay" and self.memo:
            r = self.memo.get(model, ANSWER_SYSTEM, user)
            return r["input_tokens"] if r else estimate
        if mode != "live" or self.client is None:
            return estimate
        try:
            return self.client.messages.count_tokens(
                model=model, system=ANSWER_SYSTEM, messages=[{"role": "user", "content": user}]).input_tokens
        except anthropic.APIError:
            return estimate

    def _judge_same(self, new_query, cached_query, mode="live"):
        """Haiku 4.5 cache judge; returns (same, cost of a new call or 0 if memoized)."""
        key = json.dumps([cached_query, new_query])
        with self.judge_lock:
            if key in self.judge.verdicts:
                return self.judge.verdicts[key]["verdict"] == "SAME", 0.0
        if mode == "mock":
            return False, 0.0  # no verdict without a call: treat as a miss
        if mode == "replay" or self.judge.client is None:
            raise RuntimeError(f"no memoized cache-judge verdict for {cached_query!r} vs {new_query!r}")
        v = self.judge._ask(cached_query, new_query)
        with self.judge_lock:
            self.judge.verdicts[key] = v
            self.judge.path.parent.mkdir(exist_ok=True)
            self.judge.path.write_text(json.dumps(self.judge.verdicts, indent=1))
        return v["verdict"] == "SAME", cost(CACHE_JUDGE_MODEL, v["input_tokens"], v["output_tokens"])

    def _shadow(self, root, baseline_prompt, mode):
        """Shadow baseline (Sonnet 5.5, full context), run on the pool; traced as a child of root."""
        obs = root.start_observation(name="shadow_baseline", as_type="generation",
                                     input={"system": ANSWER_SYSTEM, "user": baseline_prompt})
        try:
            res = self._llm(BASELINE_MODEL, ANSWER_SYSTEM, baseline_prompt, None, mode)
            obs.update(output=res["text"], **llm_fields(BASELINE_MODEL, res["input_tokens"], res["output_tokens"]))
            return res
        finally:
            obs.end()

    def _quality(self, query, full_context, baseline_text, optimized_text):
        user = (f"<context>\n{full_context}\n</context>\n\nCustomer message:\n{query}\n\n"
                f"Answer A (reference):\n{baseline_text or '(empty)'}\n\nAnswer B (optimized):\n{optimized_text or '(empty)'}")
        g = self._llm(QUALITY_MODEL, QUALITY_SYSTEM, user, effort="low")
        m = re.search(r"\{.*\}", g["text"], re.S)
        try:
            parsed = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            parsed = {}
        verdict = parsed.get("verdict") if parsed.get("verdict") in ("equivalent", "better", "worse") else "unparsed"
        return {"verdict": verdict, "reason": parsed.get("reason", g["text"][:300]), "cost_usd": g["cost_usd"],
                "input_tokens": g["input_tokens"], "output_tokens": g["output_tokens"]}

    def _offline_quality(self, query, window_budget):
        """Replay: compare the offline grades (vs the reference answer) of this pipeline and the baseline."""
        mine = self.grades.get((query, f"optimized_w{window_budget}"))
        base = self.grades.get((query, "baseline"))
        if mine is None or base is None or mine not in GRADE_RANK or base not in GRADE_RANK:
            return None
        diff = GRADE_RANK[mine] - GRADE_RANK[base]
        verdict = "equivalent" if diff == 0 else "better" if diff > 0 else "worse"
        return {"verdict": verdict, "reason": f"offline grades vs reference: optimized {mine}, baseline {base}",
                "cost_usd": 0.0}

    # --- the request -------------------------------------------------------------
    def answer(self, query, tenant="default", llm_mode="live", window_budget=800, shadow=False, use_cache=True,
               **request):
        """One request, traced in Langfuse (when configured) as one trace with a span per stage."""
        meta = {"config_version": CONFIG_VERSION, "window_budget": window_budget, "shadow": shadow,
                "use_cache": use_cache}
        with self.tracer.request(llm_mode, tenant_slug(tenant), query.strip(), meta) as root:
            r = self._answer(root, query, tenant=tenant, llm_mode=llm_mode, window_budget=window_budget,
                             shadow=shadow, use_cache=use_cache, **request)
            summary = {k: r[k] for k in ("answer", "layer", "model", "cost_usd", "baseline_cost_usd",
                                         "savings_pct", "latency_s")}
            root.update(output=summary, metadata={"layer": r["layer"], "model": r["model"] or "none"})
            root.set_trace_io(input={"query": r["query"], "tenant": r["tenant"], "docs": r["docs"],
                                     "case": r["case_text"]}, output=summary)
            root.score_trace(name="savings_pct", value=float(r["savings_pct"]))
            if r["quality"]:
                root.score_trace(name="quality", value=r["quality"]["verdict"], data_type="CATEGORICAL",
                                 comment=r["quality"]["reason"])
            r["trace_url"] = self.tracer.url(root)
        return r

    def _answer(self, root, query, tenant="default", doc_ids=(), custom_docs=(), case=None,
                window_budget=800, shadow=False, use_cache=True, attachments=None, llm_mode="live"):
        stage = self.tracer.stage
        t_start = time.perf_counter()
        timings, overhead, rejections = {}, {}, []
        tenant = tenant_slug(tenant)
        query = query.strip()

        # Assemble the request: docs, case, full baseline context.
        if llm_mode == "mock":
            shadow = False  # nothing to compare against
        attachments = [a for a in (attachments or [{"doc_id": d} for d in doc_ids]) if a["doc_id"] in self.corpus]
        docs = {a["doc_id"]: self.corpus[a["doc_id"]] for a in attachments}
        for cd in custom_docs:
            if cd.get("text", "").strip():
                d = parse_doc(cd.get("title", "").strip(), cd["text"])
                docs[d["id"]] = d
                attachments.append({"doc_id": d["id"]})
        if isinstance(case, dict):
            case_text = render_case(case) if case else ""
        else:
            case_text = (case or "").strip()
        doc_texts = {a["doc_id"]: render_attachment(docs, a) for a in attachments}
        full_context = "\n\n".join((["## Case\n" + case_text] if case_text else []) + list(doc_texts.values()))
        # "case:none" marks a no-case answer, so it is never reused for a request that has a case.
        case_hash = {"case:" + _sha(case_text)} if case_text else {"case:none"}
        context_hashes = {_sha(t) for t in doc_texts.values()} | case_hash
        # The config version is part of the key: a new answer prompt or model set starts a fresh cache.
        key = CONFIG_VERSION + ":" + cache_key(query, case_text or None, list(doc_texts.values()))
        baseline_prompt = prompt(query, full_context)

        shadow_future = self.pool.submit(self._shadow, root, baseline_prompt, llm_mode) if shadow else None
        count_future = self.pool.submit(self._count_tokens, BASELINE_MODEL, baseline_prompt, llm_mode)

        layer, model, served_from, entry, sem_score = None, None, None, None, None
        sent_input_tokens = output_tokens = 0
        answer_cost, answer_text, router_info, window_info = 0.0, "", None, None

        # 1. Exact cache.
        with stage(root, "exact_cache", "retriever", input={"key": key, "use_cache": use_cache}) as sp:
            t = time.perf_counter()
            if use_cache:
                entry = self.store.exact_get(tenant, key)
            timings["exact_cache"] = time.perf_counter() - t
            if entry:
                layer = "exact_cache"
            sp.update(output={"hit": bool(entry), "cached_query": entry["query"] if entry else None})

        # 2. Semantic cache.
        qvec = None
        if not layer:
            with stage(root, "semantic_cache", "retriever", input={"query": query}) as sp:
                t = time.perf_counter()
                qvec = self.embed([query])[0]
                candidates = []  # every KNN candidate and what happened to it, for the trace
                if use_cache:
                    for score, cand in self.store.semantic_candidates(tenant, qvec):
                        seen = {"cosine": round(score, 4), "cached_query": cand["query"]}
                        candidates.append(seen)
                        if score < SEMANTIC_THRESHOLD:
                            seen["verdict"] = f"below_threshold_{SEMANTIC_THRESHOLD}"
                            break
                        reason = None
                        if cand.get("config_version") != CONFIG_VERSION:
                            reason = "config_version_mismatch"
                        elif not set(cand["depends_on"]) <= context_hashes:
                            reason = "context_mismatch"
                        else:
                            for name, ok in GUARDS:
                                if not ok(query, cand["query"]):
                                    reason = f"{name}_mismatch"
                                    break
                        if reason is None:
                            with stage(sp, "cache_judge", "generation", model=CACHE_JUDGE_MODEL,
                                       input={"cached": cand["query"], "new": query}) as jsp:
                                same, jc = self._judge_same(query, cand["query"], llm_mode)
                                jsp.update(output="SAME" if same else "DIFFERENT", cost_details={"total": jc},
                                           metadata={"memoized": jc == 0})
                            overhead["cache_judge"] = overhead.get("cache_judge", 0) + jc
                            if not same:
                                reason = "judge_rejected"
                        seen["verdict"] = reason or "hit"
                        if reason:
                            rejections.append(reason)
                            continue
                        entry, sem_score, layer = cand, score, "semantic_cache"
                        break
                timings["semantic_cache"] = time.perf_counter() - t
                sp.update(output={"hit": bool(entry), "candidates": candidates})

        if entry:
            answer_text, served_from = entry["answer"], entry["query"]
            output_tokens = entry["output_tokens"]
        else:
            # 3. Context window.
            with stage(root, "window", input={"budget": window_budget, "min_context_tokens": MIN_CONTEXT_TOKENS}) as sp:
                t = time.perf_counter()
                chunks = chunk_docs(docs, attachments)
                all_sections = [c["id"] for c in chunks]
                full_tokens = approx_tokens(case_text) + sum(c["tokens"] for c in chunks)
                windowed = full_tokens >= MIN_CONTEXT_TOKENS and len(chunks) > 1
                if windowed:
                    boilerplate = self.corpus_boilerplate | repeated_sentences(docs)
                    chunks, _ = select_hybrid(chunks, qvec, self.embed(c["text"] for c in chunks), boilerplate,
                                              budget=window_budget)
                ctx_tokens = approx_tokens(case_text) + sum(c["tokens"] for c in chunks)
                kept_docs = sorted({c["doc_id"] for c in chunks})
                window_info = {"applied": windowed, "budget": window_budget, "tokens_before": full_tokens,
                               "tokens_after": ctx_tokens, "docs_kept": kept_docs,
                               "sections_kept": [c["id"] for c in chunks]}
                timings["window"] = time.perf_counter() - t
                kept = {c["id"] for c in chunks}
                sp.update(output={**window_info, "sections_dropped": [s for s in all_sections if s not in kept],
                                  "kept_text": {c["id"]: c["text"] for c in chunks}})

            # 4. Route.
            with stage(root, "route") as sp:
                t = time.perf_counter()
                feats = [text_features(query, len(kept_docs), ctx_tokens)]
                p_large = float(self.router.p_large(np.array([qvec]), np.array(feats, float))[0])
                model = LARGE if p_large >= self.router.threshold else SMALL
                router_info = {"p_large": round(p_large, 4), "threshold": self.router.threshold, "model": model}
                timings["route"] = time.perf_counter() - t
                sp.update(input={"features": dict(zip(
                              ["words", "sentences", "questions", "numbers", "connectives", "ends_right_q",
                               "docs_kept", "log_context_tokens"], (float(f) for f in feats[0])))},
                          output=router_info)

            # 5. Answer.
            user = prompt(query, optimized_context(case_text, chunks))
            with stage(root, "answer", "generation", model=model,
                       input={"system": ANSWER_SYSTEM, "user": user}) as sp:
                t = time.perf_counter()
                res = self._llm(model, ANSWER_SYSTEM, user, mode=llm_mode)
                timings["llm"] = time.perf_counter() - t
                sp.update(output=res["text"], **llm_fields(model, res["input_tokens"], res["output_tokens"]))
            layer = "llm"
            answer_text, answer_cost = res["text"], res["cost_usd"]
            sent_input_tokens, output_tokens = res["input_tokens"], res["output_tokens"]

            # 6. Store (only real answers). The answer depends on the docs windowing kept and the case.
            if use_cache and res["stop_reason"] != "refusal" and answer_text:
                with stage(root, "cache_store") as sp:
                    depends_on = sorted({_sha(doc_texts[d]) for d in kept_docs if d in doc_texts} | case_hash)
                    new_entry = {"answer": answer_text, "query": query, "model": model, "depends_on": depends_on,
                                 "output_tokens": output_tokens, "config_version": CONFIG_VERSION,
                                 "created": int(time.time())}
                    self.store.exact_put(tenant, key, new_entry)
                    self.store.semantic_put(tenant, key, qvec, query, new_entry)
                    sp.update(output={"key": key, "depends_on": depends_on})

        latency = time.perf_counter() - t_start

        # Baseline cost: real if the shadow ran, else estimated.
        baseline, quality = None, None
        if shadow_future:
            t = time.perf_counter()
            baseline = shadow_future.result()
            timings["shadow_wait"] = time.perf_counter() - t
            t = time.perf_counter()
            if llm_mode == "replay":  # nothing is spent: baseline and grades are memoized
                quality = self._offline_quality(query, window_budget)
            else:
                overhead["shadow_baseline"] = baseline["cost_usd"]
                with stage(root, "quality_judge", "evaluator", model=QUALITY_MODEL,
                           input={"baseline": baseline["text"], "optimized": answer_text}) as sp:
                    quality = self._quality(query, full_context, baseline["text"], answer_text)
                    sp.update(output={"verdict": quality["verdict"], "reason": quality["reason"]},
                              **llm_fields(QUALITY_MODEL, quality["input_tokens"], quality["output_tokens"]))
                overhead["quality_judge"] = quality["cost_usd"]
            timings["quality_judge"] = time.perf_counter() - t
            baseline_input, baseline_cost, baseline_source = baseline["input_tokens"], baseline["cost_usd"], "measured"
        else:
            baseline_input = count_future.result()
            baseline_cost = cost(BASELINE_MODEL, baseline_input, output_tokens)
            baseline_source = "estimated"
        total_cost = answer_cost + overhead.get("cache_judge", 0)

        return {
            "request_id": uuid.uuid4().hex[:12], "tenant": tenant, "query": query, "llm_mode": llm_mode,
            "config_version": CONFIG_VERSION,
            "answer": answer_text, "layer": layer, "model": model or "",
            "served_from_query": served_from, "semantic_score": round(sem_score, 4) if sem_score else None,
            "semantic_rejections": rejections, "window": window_info, "router": router_info,
            "input_tokens": sent_input_tokens, "output_tokens": output_tokens,
            "baseline_input_tokens": baseline_input, "baseline_source": baseline_source,
            "cost_usd": round(total_cost, 6), "baseline_cost_usd": round(baseline_cost, 6),
            "savings_usd": round(baseline_cost - total_cost, 6),
            "savings_pct": round(1 - total_cost / baseline_cost, 4) if baseline_cost else 0.0,
            "overhead_usd": {k: round(v, 6) for k, v in overhead.items()},
            "quality": quality, "baseline_answer": baseline["text"] if baseline else None,
            "timings": {k: round(v, 4) for k, v in timings.items()}, "latency_s": round(latency, 3),
            "docs": list(docs), "case_text": case_text, "created": int(time.time()),
        }
