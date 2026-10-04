"""Labelled query pairs for calibrating the semantic-cache threshold -> eval/data/cache_pairs/pairs_v1.jsonl

    python -m eval.cache_pairs            # downloads Bitext (+ QQP) from Hugging Face once, then builds the file

Each row: {"id", "query_a", "query_b", "label", "source", "kind", ...}. label 1 = serving query_a's cached answer for
query_b is correct; label 0 = it would be a false hit.

Sources
  bitext   Bitext customer-support dataset (CDLA-Sharing-1.0). Responses are written per intent, so two queries with
           the same intent can share one answer -- *if* they also carry the same specifics. Bitext responses echo the
           customer's specifics ({{Order Number}} 100%, {{Account Type}} 98%, literal tiers ~70%), so a pair with the
           same intent but a different order number / account tier / placeholder set is labelled 0
           (kind "same_intent_diff_specifics"). One relabel: the dataset's `newsletter_subscription` intent mixes
           subscribe and unsubscribe requests whose responses differ, so it is split into two labels.
           Negatives: near-miss intents in the same category (cancel_order vs track_order), embedder-mined nearest
           neighbours with a different intent, and random different-intent pairs.
  qqp      Quora Question Pairs (GLUE validation split), 1,000 duplicates + 1,000 non-duplicates: general-domain check.
  trap     hand-written lookalikes that need different answers (numbers, negation, entities, constraints).
           Rows of eval/data/evalset/*.jsonl with type == "trap_pair" are used when present; the in-file seed set
           (author "seed") is always included.
  seed     hand-written paraphrases that *should* hit, to measure what the guards wrongly block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from costguard.config import ROOT

DATA_DIR = ROOT / "eval" / "data" / "cache_pairs"
PAIRS_PATH = DATA_DIR / "pairs_v1.jsonl"
MANIFEST_PATH = DATA_DIR / "pairs_v1.manifest.json"
BITEXT_PARQUET = DATA_DIR / "bitext_queries.parquet"
EVALSET_DIR = ROOT / "eval" / "data" / "evalset"
EMB_CACHE_DIR = ROOT / "data" / "runtime" / "emb_cache"     # git-ignored; embeddings are cheap to recompute

BITEXT_REPO = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BITEXT_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
QQP_REPO, QQP_FILE = "nyu-mll/glue", "qqp/validation-00000-of-00001.parquet"
SEED = 13

LICENCES = {
    "bitext": {"dataset": f"hf:{BITEXT_REPO}", "licence": "CDLA-Sharing-1.0",
               "note": "Bitext sample customer-support dataset; we keep a compact copy of instruction/category/intent."},
    "qqp": {"dataset": f"hf:{QQP_REPO} (qqp, validation)", "licence": "Quora original release terms (GLUE card: 'other')",
            "note": "Used for non-commercial evaluation only; only a 2,000-pair sample is stored."},
    "trap": {"dataset": "hand-written (this repo)", "licence": "project licence"},
    "seed": {"dataset": "hand-written (this repo)", "licence": "project licence"},
}

# ============================================================================================ Bitext

# placeholders that stand for a customer-specific value; the response echoes them, so they are part of the "answer key"
VALUE_PLACEHOLDERS = {"Order Number", "Invoice Number", "Refund Amount", "Account Type", "Account Category",
                      "Delivery City", "Delivery Country", "Person Name"}
LITERAL_TIERS = {"free", "freemium", "standard", "premium", "pro", "gold", "platinum", "business", "personal"}
_PH = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
_UNSUB = re.compile(r"(?i)\bun\w*s\w*b|\bunsu|opt.?out|stop (?:receiv|gett)|cancel|remove|leave")


def refine_intent(intent: str, text: str) -> str:
    if intent == "newsletter_subscription":
        return "newsletter_unsubscribe" if _UNSUB.search(text) else "newsletter_subscribe"
    return intent


def specifics(text: str) -> frozenset:
    """What a templated answer would echo back: value placeholders, literal numbers, literal account tiers."""
    out = {f"<{m.group(1)}>" for m in _PH.finditer(text) if m.group(1) in VALUE_PLACEHOLDERS}
    t = _PH.sub(" ", text.lower())
    out |= set(re.findall(r"\d+(?:\.\d+)?", t))
    out |= {w for w in re.findall(r"[a-z]+", t) if w in LITERAL_TIERS}
    return frozenset(out)


def load_bitext() -> pd.DataFrame:
    """Compact Bitext table (instruction, category, intent, flags, label). Cached as parquet under DATA_DIR."""
    if not BITEXT_PARQUET.exists():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(BITEXT_REPO, BITEXT_FILE, repo_type="dataset")
        df = pd.read_csv(p)[["instruction", "category", "intent", "flags"]]
        df["instruction"] = df["instruction"].astype(str).str.strip()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(BITEXT_PARQUET, index=False)
    df = pd.read_parquet(BITEXT_PARQUET)
    df["label"] = [refine_intent(i, t) for i, t in zip(df["intent"], df["instruction"])]
    df["spec"] = [specifics(t) for t in df["instruction"]]
    return df


# ============================================================================================ embeddings (cached)


def embed_texts(texts: list[str], model_name: Optional[str] = None) -> np.ndarray:
    """Embed with the cache's embedder; results cached on disk (data/runtime/emb_cache) keyed by model + texts."""
    from costguard.cache.embedder import DEFAULT_MODEL, get_embedder
    name = model_name or DEFAULT_MODEL
    h = hashlib.sha256(("\x00".join(texts) + "\x01" + name).encode()).hexdigest()[:16]
    path = EMB_CACHE_DIR / f"{re.sub(r'[^a-z0-9]+', '_', name.lower())}_{h}.npy"
    if path.exists():
        return np.load(path)
    m = get_embedder(name).embed(texts, batch_size=256)
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, m)
    return m


# ============================================================================================ hand-written pairs

SEED_TRAPS: list[tuple[str, str, str]] = [  # (query_a, query_b, what differs) -- a hit between them is a false hit
    ("I want to cancel my order #4821", "I want to cancel my order #4822", "number"),
    ("Where is my order 55210?", "Where is my order 55201?", "number"),
    ("I was charged $49.99 twice for one order", "I was charged $59.99 twice for one order", "number"),
    ("Can I return an item after 30 days?", "Can I return an item after 60 days?", "number"),
    ("I ordered 2 jackets but only received 1", "I ordered 3 jackets but only received 1", "number"),
    ("Can my order be delivered by 12/10?", "Can my order be delivered by 21/10?", "number"),
    ("I want to cancel my order #4821", "I don't want to cancel my order #4821", "negation"),
    ("My package was delivered to the wrong address", "My package was not delivered to my address", "negation"),
    ("Can I return an item without the receipt?", "Can I return an item with the receipt?", "negation"),
    ("How do I subscribe to the newsletter?", "How do I unsubscribe from the newsletter?", "negation"),
    ("Can I return an opened item?", "Can I return an unopened item?", "negation"),
    ("I received my refund but it is the wrong amount", "I never received my refund", "negation"),
    ("What is the return window for laptops?", "What is the return window for phones?", "entity"),
    ("I want a refund for my jacket", "I want an exchange for my jacket", "entity"),
    ("Do you ship to Pune?", "Do you ship to Dubai?", "entity"),
    ("Can I pay with PayPal?", "Can I pay with UPI?", "entity"),
    ("How long does express shipping take?", "How long does standard shipping take?", "entity"),
    ("How do I track my order?", "How do I cancel my order?", "entity"),
    ("How do I change my shipping address?", "How do I change my password?", "entity"),
    ("Is there a warranty on the headphones?", "Is there a warranty on the sofa?", "entity"),
    ("How do I delete my account?", "How do I create an account?", "entity"),
    ("Can I return apparel bought on sale?", "Can I return electronics bought on sale?", "entity"),
    ("Can I switch to the premium plan?", "Can I switch to the free plan?", "entity"),
    ("Do you deliver on weekends?", "Do you deliver on weekdays?", "entity"),
    ("Do you ship to Nagpur?", "Do you ship to Indore?", "entity_unlisted"),
    ("Can I cancel my order before it ships?", "Can I cancel my order after it ships?", "constraint"),
    ("What happens if my package arrives late?", "What happens if my package arrives early?", "constraint"),
    ("Can I return a gift without the gift receipt?", "Can I return a gift I bought for someone?", "constraint"),
]

SEED_PARAPHRASES: list[tuple[str, str]] = [  # should hit: the same answer serves both
    ("How do I cancel my order #4821?", "how can i cancel order 4821"),
    ("Where is my package?", "Can you tell me where my parcel is?"),
    ("What is your return policy?", "How do returns work at your store?"),
    ("I forgot my password", "How do I reset my password?"),
    ("Do you ship to Pune?", "Can you deliver to Pune?"),
    ("Can I pay with PayPal?", "Is PayPal accepted as a payment method?"),
    ("I can't log in to my account", "Help me log in to my account"),
    ("How long does express shipping take?", "What is the delivery time for express shipping?"),
    ("How do I unsubscribe from the newsletter?", "Please remove me from your mailing list"),
    ("I want to return my laptop", "How do I send back my laptop?"),
    ("What is the return window for laptops?", "How many days do I have to return a laptop?"),
    ("Is there a warranty on headphones?", "Do headphones come with a warranty?"),
    ("How do I talk to a human agent?", "I want to speak with a real person"),
    ("What payment methods do you accept?", "Which ways can I pay?"),
]


def hand_written_pairs() -> list[dict]:
    rows = []
    for i, (a, b, what) in enumerate(SEED_TRAPS):
        rows.append({"id": f"trap-seed-{i:03d}", "query_a": a, "query_b": b, "label": 0, "source": "trap",
                     "kind": "trap", "trap_type": what, "author": "seed"})
    for i, (a, b) in enumerate(SEED_PARAPHRASES):
        rows.append({"id": f"para-seed-{i:03d}", "query_a": a, "query_b": b, "label": 1, "source": "seed",
                     "kind": "paraphrase", "author": "seed"})
    return rows


def evalset_trap_pairs() -> list[dict]:
    """trap_pair rows from the team's hand-written eval set, paired by pair_id."""
    rows = []
    for f in sorted(EVALSET_DIR.glob("*.jsonl")) if EVALSET_DIR.exists() else []:
        groups: dict[str, list[dict]] = defaultdict(list)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "trap_pair" and r.get("pair_id"):
                groups[str(r["pair_id"])].append(r)
        for pid, g in groups.items():
            if len(g) < 2:
                continue
            a, b = g[0], g[1]
            hit_ok = bool(a.get("should_cache_hit")) and bool(b.get("should_cache_hit"))
            rows.append({"id": f"evalset-{pid}", "query_a": a["query"], "query_b": b["query"], "label": int(hit_ok),
                         "source": "trap", "kind": "trap", "trap_type": a.get("category") or "evalset",
                         "author": a.get("author", f.stem)})
    return rows


# ============================================================================================ QQP


def qqp_pairs(n_pos: int, n_neg: int, seed: int) -> list[dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(QQP_REPO, QQP_FILE, repo_type="dataset")
    df = pd.read_parquet(p)
    df = df[(df.question1.str.len() > 5) & (df.question2.str.len() > 5)]
    pos = df[df.label == 1].sample(n_pos, random_state=seed)
    neg = df[df.label == 0].sample(n_neg, random_state=seed)
    out = []
    for r in pd.concat([pos, neg]).itertuples():
        out.append({"id": f"qqp-{int(r.idx)}", "query_a": r.question1.strip(), "query_b": r.question2.strip(),
                    "label": int(r.label), "source": "qqp", "kind": "duplicate" if r.label == 1 else "non_duplicate"})
    return out


# ============================================================================================ Bitext pairs


def bitext_pairs(df: pd.DataFrame, seed: int, n_pos=1200, n_spec=300, n_near=600, n_mined=600, n_rand=300) -> list[dict]:
    rng = np.random.default_rng(seed)
    df = df.drop_duplicates("instruction").reset_index(drop=True)
    texts = df["instruction"].tolist()
    labels = df["label"].to_numpy()
    cats = df["category"].to_numpy()
    spec = df["spec"].tolist()
    by_label: dict[str, np.ndarray] = {l: np.flatnonzero(labels == l) for l in sorted(set(labels))}
    label_names = sorted(by_label)
    seen: set[tuple[int, int]] = set()
    rows: list[dict] = []

    def add(i: int, j: int, kind: str) -> bool:
        key = (min(i, j), max(i, j))
        if i == j or key in seen:
            return False
        seen.add(key)
        same = labels[i] == labels[j]
        lab = int(same and spec[i] == spec[j])
        rows.append({"id": f"bitext-{len(rows):05d}", "query_a": texts[i], "query_b": texts[j], "label": lab,
                     "source": "bitext", "kind": kind, "intent_a": labels[i], "intent_b": labels[j],
                     "label_intent_only": int(same)})
        return True

    def n_kind(kind: str) -> int:
        return sum(1 for r in rows if r["kind"] == kind)

    # 1) same intent, same specifics  (positives)
    while n_kind("same_intent") < n_pos:
        pool = by_label[label_names[rng.integers(len(label_names))]]
        i = int(rng.choice(pool))
        cand = [j for j in rng.choice(pool, size=min(40, len(pool)), replace=False) if spec[j] == spec[i] and j != i]
        if cand:
            add(i, int(cand[0]), "same_intent")
    # 2) same intent, different specifics (order number vs none, premium vs gold ...) -> label 0
    tries = 0
    while n_kind("same_intent_diff_specifics") < n_spec and tries < 100_000:
        tries += 1
        pool = by_label[label_names[rng.integers(len(label_names))]]
        i = int(rng.choice(pool))
        cand = [j for j in rng.choice(pool, size=min(40, len(pool)), replace=False) if spec[j] != spec[i]]
        if cand:
            add(i, int(cand[0]), "same_intent_diff_specifics")
    # 3) near-miss intents within one category (cancel_order vs track_order, get_refund vs track_refund)
    cat_labels = defaultdict(set)
    for l, c in zip(labels, cats):
        cat_labels[c].add(l)
    multi = sorted(c for c, ls in cat_labels.items() if len(ls) > 1)
    while n_kind("near_miss_intent") < n_near:
        c = multi[rng.integers(len(multi))]
        la, lb = rng.choice(sorted(cat_labels[c]), size=2, replace=False)
        add(int(rng.choice(by_label[la])), int(rng.choice(by_label[lb])), "near_miss_intent")
    # 4) embedder-mined: nearest neighbour with a different intent (the pairs a cache actually confuses)
    emb = embed_texts(texts)
    anchors = rng.choice(len(texts), size=n_mined * 3, replace=False)
    for i in anchors:
        if n_kind("mined_hard_negative") >= n_mined:
            break
        sims = emb @ emb[i]
        sims[labels == labels[i]] = -1
        add(int(i), int(np.argmax(sims)), "mined_hard_negative")
    # 5) random different-intent pairs
    while n_kind("random_negative") < n_rand:
        i, j = rng.integers(len(texts), size=2)
        if labels[i] != labels[j]:
            add(int(i), int(j), "random_negative")
    return rows


# ============================================================================================ main


def build(seed: int = SEED, with_qqp: bool = True) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_bitext()
    rows = bitext_pairs(df, seed)
    notes = []
    if with_qqp:
        try:
            rows += qqp_pairs(1000, 1000, seed)
        except Exception as e:  # offline / HF outage: keep going without the general-domain check
            notes.append(f"QQP skipped: {type(e).__name__}: {e}")
            print("!! " + notes[-1])
    ev = evalset_trap_pairs()
    if not ev:
        notes.append("eval/data/evalset/*.jsonl had no trap_pair rows; only the in-file seed traps are used")
    rows += ev + hand_written_pairs()
    with PAIRS_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "file": str(PAIRS_PATH.relative_to(ROOT)), "seed": seed, "rows": len(rows),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_kind": dict(Counter(f'{r["source"]}:{r["kind"]}' for r in rows)),
        "by_label": {s: dict(Counter(r["label"] for r in rows if r["source"] == s)) for s in sorted({r["source"] for r in rows})},
        "bitext_rows": int(len(df)), "bitext_labels": int(df["label"].nunique()),
        "sha256": hashlib.sha256(PAIRS_PATH.read_bytes()).hexdigest(),
        "licences": LICENCES, "notes": notes,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_pairs(path: Path = PAIRS_PATH) -> list[dict]:
    if not path.exists():
        build()
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-qqp", action="store_true")
    a = ap.parse_args(argv)
    m = build(a.seed, with_qqp=not a.no_qqp)
    print(json.dumps({k: m[k] for k in ("file", "rows", "by_source", "by_kind", "by_label", "notes")}, indent=2))
    print(f"size: {PAIRS_PATH.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
