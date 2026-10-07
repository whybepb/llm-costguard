"""Context window management: keep the policy sections that matter for a query.

Sections are scored by cosine similarity to the query (normalized embeddings),
then kept best-first until the token budget is reached. The case block is always
kept and does not count toward the budget. repeated_reminders sections are always
dropped. Kept sections are put back in document order so the prompt reads naturally.
Text is never rewritten here; that is compression's job.
"""

import re

from config import CFG

DROP_SECTIONS = {"repeated_reminders"}
MIN_CONTEXT_TOKENS = CFG["window"]["min_context_tokens"]  # below this, windowing isn't worth it


def approx_tokens(text):
    # Same estimate the dataset uses for approx_context_tokens.
    return int(round(len(re.findall(r"\S+", text)) * 1.3))


def chunk_docs(docs, attachments):
    """One chunk per section of each attached doc, in attachment order."""
    chunks = []
    for a in attachments:
        doc = docs[a["doc_id"]]
        wanted = a.get("sections")
        for name, text in doc["sections"].items():
            if wanted and name not in wanted:
                continue
            chunks.append({
                "id": f"{doc['id']}#{name}",
                "doc_id": doc["id"],
                "section": name,
                # Title and section name help the embedding place a short section.
                "text": f"{doc['title']} - {name}\n{text}",
                "tokens": approx_tokens(text),
            })
    return chunks


def select(chunks, query_vec, chunk_vecs, budget):
    """Return (kept chunks in document order, scores by chunk id)."""
    scores = {c["id"]: float(v @ query_vec) for c, v in zip(chunks, chunk_vecs)}
    candidates = [c for c in chunks if c["section"] not in DROP_SECTIONS]
    ranked = sorted(candidates, key=lambda c: -scores[c["id"]])
    kept, used = set(), 0
    for c in ranked:
        # Always keep the best section, even if it alone exceeds the budget.
        if kept and used + c["tokens"] > budget:
            continue
        kept.add(c["id"])
        used += c["tokens"]
    return [c for c in chunks if c["id"] in kept], scores


# --- Doc-first selection -----------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def sentences(text):
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def repeated_sentences(docs, min_docs=3):
    """Sentences that appear in at least min_docs different documents: boilerplate."""
    seen = {}
    for doc in docs.values():
        for s in {s for t in doc["sections"].values() for s in sentences(t)}:
            seen[s] = seen.get(s, 0) + 1
    return {s for s, n in seen.items() if n >= min_docs}


def strip_boilerplate(chunk, boilerplate, drop_ratio=0.3):
    """Remove repeated sentences; drop the whole section if they made up drop_ratio of it.

    A section that is mostly copied text is a boilerplate block, and what's left of
    it is usually filler written around the copied part.
    """
    body = chunk["text"].split("\n", 1)[1]
    ss = sentences(body)
    keep = [s for s in ss if s not in boilerplate]
    if not ss or len(keep) == len(ss):
        return chunk
    if (len(ss) - len(keep)) / len(ss) >= drop_ratio:
        return None
    text = " ".join(keep)
    return {**chunk, "text": chunk["text"].split("\n", 1)[0] + "\n" + text, "tokens": approx_tokens(text)}


def doc_scores(chunks, scores, method):
    """Score each doc from its sections: 'max' (best section) or 'mean'."""
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(scores[c["id"]])
    agg = max if method == "max" else (lambda v: sum(v) / len(v))
    return {d: agg(v) for d, v in by_doc.items()}


def select_docs(chunks, query_vec, chunk_vecs, boilerplate, method="max", gap=0.05, max_docs=3):
    """Keep the top doc, plus any doc scoring within `gap` of it (up to max_docs),
    with all of their sections minus boilerplate. Returns (kept chunks, doc scores)."""
    scores = {c["id"]: float(v @ query_vec) for c, v in zip(chunks, chunk_vecs)}
    ds = doc_scores(chunks, scores, method)
    ranked = sorted(ds, key=lambda d: -ds[d])
    chosen = {d for d in ranked[:max_docs] if ds[d] >= ds[ranked[0]] - gap}
    kept = []
    for c in chunks:
        if c["doc_id"] in chosen:
            c = strip_boilerplate(c, boilerplate)
            if c is not None:
                kept.append(c)
    return kept, ds


def select_hybrid(chunks, query_vec, chunk_vecs, boilerplate, method="max", budget=800):
    """Keep the top doc whole (minus boilerplate), then fill the rest of the budget
    with the best-scoring sections from the other docs. The top doc is kept even if
    it alone exceeds the budget. Returns (kept chunks in document order, doc scores)."""
    scores = {c["id"]: float(v @ query_vec) for c, v in zip(chunks, chunk_vecs)}
    ds = doc_scores(chunks, scores, method)
    top = max(ds, key=ds.get)
    cleaned = {c["id"]: strip_boilerplate(c, boilerplate) for c in chunks}
    kept = {i for i, c in cleaned.items() if c is not None and c["doc_id"] == top}
    used = sum(cleaned[i]["tokens"] for i in kept)
    others = [c for i, c in cleaned.items() if c is not None and c["doc_id"] != top]
    for c in sorted(others, key=lambda c: -scores[c["id"]]):
        if used + c["tokens"] > budget:
            continue
        kept.add(c["id"])
        used += c["tokens"]
    return [cleaned[c["id"]] for c in chunks if c["id"] in kept], ds
