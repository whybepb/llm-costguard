"""Build the frozen replay workload (eval/data/trace_v1.jsonl + .sha256) and the CI subset.

    python -m eval.build_trace --size 600 --dup-rate 0.3 --seed 7      # -> trace_v1.jsonl (the headline trace)
    python -m eval.build_trace --dup-rate 0.15                         # -> trace_v1_dup15.jsonl (sensitivity)
    python -m eval.build_trace --all-variants                          # 0 / 0.15 / 0.3 / 0.5
    python -m eval.build_trace --ci-subset                             # -> eval/data/ci_subset.jsonl
    python -m eval.build_trace --stats eval/data/trace_v1.jsonl        # describe an existing trace

Mix (of N requests):
  ~10%  trap pairs: the eval set's trap_pair rows (all authors), topped up with Bitext near-miss pairs (cancel vs track
        on the same order, the same template with a different order number, ...). Pair members sit 1-3 positions apart.
  rest  split 55:25 between Bitext customer-support queries (HF bitext/Bitext-customer-support-llm-chatbot-training-dataset,
        CDLA-Sharing 1.0; the dataset response is the reference) and KB questions with retrieved context (eval.kb).
        If eval.kb is unavailable the KB slice is skipped (logged) and Bitext fills it.
Duplicates: exactly round(dup_rate * N) requests have an earlier semantically equivalent request (`dup_of`):
  1/3 of them are exact repeats (half verbatim, half case/punctuation variants that normalise to the same exact-cache
  key) and 2/3 are paraphrases taken from the dataset itself (Bitext has hundreds of phrasings per intent).
  Which cluster a duplicate repeats follows a Zipf(s=1.1) popularity law: FAQ-like clusters (generic Bitext intents,
  KB questions) form the head, customer-specific clusters (a particular order number) the tail.
Clusters ("semantically equivalent" = a cache may legitimately serve one for the other):
  bitext  intent + slot values. "cancel order SN-48213" and "cancel order SN-48231" are different clusters, so serving
          one for the other counts as a false hit. Phrasings without slots form one generic cluster per intent.
  kb      one cluster per KB question id (its paraphrases included).
  trap    every trap row is its own cluster.
Unique items are drawn from a seeded stream that does not depend on the dup rate, so the dup-rate variants are nested
(the 0.3 trace's unique items are a subset of the 0.15 trace's) and share cassette entries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

from costguard.config import ROOT
from costguard.pipeline import normalize_query

log = logging.getLogger("costguard.eval.trace")

DATA = ROOT / "eval" / "data"
RAW = DATA / "raw"
EVALSET_DIR = DATA / "evalset"
TRACE = DATA / "trace_v1.jsonl"
CI_SUBSET = DATA / "ci_subset.jsonl"

CATEGORIES = ("order", "shipping", "returns", "refund", "payment", "account", "product", "other")
EVAL_TYPES = ("answerable", "trap_pair", "hard")
BITEXT_REPO = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BITEXT_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
BITEXT_LICENSE = "CDLA-Sharing-1.0"
ZIPF_S = 1.1
SHARES = {"bitext": 0.55, "kb": 0.25}            # of the whole trace; traps 0.10, repeats ~0.10 spread over both
TRAP_SHARE = 0.10
KB_K = 8

INTENT_CATEGORY = {
    "cancel_order": "order", "change_order": "order", "place_order": "order", "track_order": "order",
    "check_cancellation_fee": "order",
    "delivery_options": "shipping", "delivery_period": "shipping", "change_shipping_address": "shipping",
    "set_up_shipping_address": "shipping",
    "check_refund_policy": "refund", "get_refund": "refund", "track_refund": "refund",
    "check_payment_methods": "payment", "payment_issue": "payment", "check_invoice": "payment", "get_invoice": "payment",
    "create_account": "account", "delete_account": "account", "edit_account": "account", "recover_password": "account",
    "registration_problems": "account", "switch_account": "account", "newsletter_subscription": "account",
    "contact_customer_service": "other", "contact_human_agent": "other", "complaint": "other", "review": "other",
}

# Bitext intents that need the same answer at ShopNest are one equivalence class for clustering (a cache hit between
# them is correct): "see my bill" / "get my bill" (both: My Orders > Download invoice), "contact customer service" /
# "talk to a human agent" (both: 24x7 chat).
EQUIVALENT_INTENTS = {"get_invoice": "check_invoice", "contact_human_agent": "contact_customer_service"}

# Bitext near-miss pairs: two intents that read alike but need different answers. Only intents that share a slot
# (order number, amount, name, account type) are used, so every trap row gets fresh slot values and therefore its own
# cluster; generic phrasings would make trap rows from different pairs equivalent to each other.
# Deliberately excluded: check_invoice/get_invoice (near-synonyms: "see my bill" vs "get my bill") and
# get_refund/track_refund (many Bitext phrasings, e.g. "rebate ₹499", are ambiguous between the two intents).
CROSS_INTENT_PAIRS = [
    ("cancel_order", "track_order"), ("cancel_order", "change_order"), ("track_order", "change_order"),
    ("create_account", "delete_account"), ("edit_account", "switch_account"), ("create_account", "switch_account"),
    ("delete_account", "edit_account"),
]

# ------------------------------------------------------------------------------------------- slot filling
_SLOT_RE = re.compile(r"\{\{([^}]+)\}\}")
PERSON_NAMES = ["Aarav Mehta", "Priya Nair", "Rohan Gupta", "Ananya Iyer", "Kabir Singh", "Meera Pillai", "Arjun Rao",
                "Sneha Kulkarni", "Vikram Joshi", "Isha Banerjee", "Dev Malhotra", "Nisha Reddy", "Farhan Qureshi",
                "Tara D'Souza", "Neel Chatterjee", "Ritu Agarwal"]
ACCOUNT_TYPES = ["Plus", "Business", "Standard", "Premium", "Gold", "Platinum", "Family", "Student"]
ACCOUNT_CATEGORIES = ["Basic", "Pro", "Elite", "Seller", "Freemium", "Diamond"]
CITIES = ["Bengaluru", "Mumbai", "Pune", "Hyderabad", "Chennai", "Jaipur", "Kochi", "Lucknow", "Indore", "Guwahati",
          "Port Blair", "Leh"]
COUNTRIES = ["the UAE", "Singapore", "the UK", "the USA", "Canada", "Australia", "Nepal", "Germany", "France", "Japan"]
AMOUNTS = ["349", "499", "799", "899", "1,299", "1,599", "1,999", "2,499", "3,999", "5,499", "7,250", "12,999"]
CONSTANTS = {
    "Online Order Interaction": "My Orders", "Customer Support Phone Number": "1800-202-6378",
    "Website URL": "www.shopnest.example", "Customer Support Hours": "24x7", "Online Company Portal Info": "ShopNest account",
    "Date Range": "the last 30 days", "Salutation": "Mr./Ms.", "Settings": "Settings", "Profile": "Profile",
    "Account Change": "Change account type", "Upgrade Account": "Upgrade", "Login Page URL": "www.shopnest.example/login",
    "Store Location": "our store locator", "Forgot Password": "Forgot password", "Order Status": "Order status",
    "Customer Support Email": "support@shopnest.example", "Profile Type": "Profile type", "Order Tracking": "Track order",
    "Forgot PIN": "Forgot PIN", "Currency Symbol": "₹",
}
UNSET = {"Order Number": "your order number", "Invoice Number": "your invoice number", "Person Name": "the account holder",
         "Account Type": "the selected", "Account Category": "the selected", "Refund Amount": "the refund amount",
         "Delivery City": "your city", "Delivery Country": "your country", "Client Last Name": "Customer"}


_POOL_WORDS = re.compile(r"\b(" + "|".join(sorted({w.lower() for w in ACCOUNT_TYPES + ACCOUNT_CATEGORIES + CITIES}
                                                     | {c.replace("the ", "").lower() for c in COUNTRIES}
                                                     | {"freemium", "gold", "platinum", "premium", "standard"})) + r")\b",
                        re.IGNORECASE)


def slot_signature(text: str) -> str:
    return "|".join(sorted({s for s in _SLOT_RE.findall(text) if s != "Currency Symbol"}))


def _new_slot_values(sig: str, rng: random.Random, used_orders: set) -> dict:
    vals: dict[str, str] = {}
    for slot in sig.split("|") if sig else []:
        if slot == "Order Number":
            while True:
                v = f"SN-{rng.randint(10000, 99999)}"
                if v not in used_orders:
                    used_orders.add(v)
                    break
            vals[slot] = v
        elif slot == "Invoice Number":
            vals[slot] = f"INV-{rng.randint(100000, 999999)}"
        elif slot == "Person Name":
            vals[slot] = rng.choice(PERSON_NAMES)
        elif slot == "Account Type":
            vals[slot] = rng.choice(ACCOUNT_TYPES)
        elif slot == "Account Category":
            vals[slot] = rng.choice(ACCOUNT_CATEGORIES)
        elif slot == "Refund Amount":
            vals[slot] = rng.choice(AMOUNTS)
        elif slot == "Delivery City":
            vals[slot] = rng.choice(CITIES)
        elif slot == "Delivery Country":
            vals[slot] = rng.choice(COUNTRIES)
        else:
            vals[slot] = slot
    return vals


def fill_slots(text: str, values: dict, cluster_tag: str = "") -> str:
    """Replace Bitext {{placeholders}} with this cluster's slot values (or ShopNest constants)."""
    v = dict(values)
    if "Person Name" in v:
        v.setdefault("Client Last Name", v["Person Name"].split()[-1])
    amount = v.get("Refund Amount")
    text = re.sub(r"(\{\{Currency Symbol\}\}|\$)?\{\{Refund Amount\}\}(\s*d+ollars)?",
                  (lambda m: f"₹{amount}") if amount else (lambda m: UNSET["Refund Amount"]), text)
    if "{{Tracking Number}}" in text:
        v["Tracking Number"] = "SNT" + hashlib.sha256(cluster_tag.encode()).hexdigest()[:9].upper()

    def rep(m: re.Match) -> str:
        k = m.group(1)
        return v.get(k) or CONSTANTS.get(k) or UNSET.get(k) or k
    return re.sub(r"\s{2,}", " ", _SLOT_RE.sub(rep, text)).strip()


# ------------------------------------------------------------------------------------------- data sources
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def validate_evalset(rows: list[dict]) -> list[str]:
    errs, ids = [], Counter(r.get("id") for r in rows)
    for r in rows:
        rid = r.get("id")
        for k in ("id", "category", "query", "reference", "type", "author"):
            if not r.get(k):
                errs.append(f"{rid}: missing {k}")
        if r.get("category") not in CATEGORIES:
            errs.append(f"{rid}: bad category {r.get('category')!r}")
        if r.get("type") not in EVAL_TYPES:
            errs.append(f"{rid}: bad type {r.get('type')!r}")
        if r.get("type") == "trap_pair" and not r.get("pair_id"):
            errs.append(f"{rid}: trap_pair without pair_id")
        if ids[rid] > 1:
            errs.append(f"{rid}: duplicate id")
        if r.get("same_as") is not None and not isinstance(r["same_as"], str):
            errs.append(f"{rid}: same_as must be an id string")
    pairs = Counter(r["pair_id"] for r in rows if r.get("type") == "trap_pair" and r.get("pair_id"))
    errs += [f"pair {p}: {n} rows (need exactly 2)" for p, n in pairs.items() if n != 2]
    return sorted(set(errs))


def load_evalset(directory: Path = EVALSET_DIR, validate: bool = True) -> list[dict]:
    """All hand-written eval rows (eval/data/evalset/*.jsonl, skipping _* and TEMPLATE*), in file then line order."""
    rows: list[dict] = []
    for f in sorted(Path(directory).glob("*.jsonl")):
        if f.name.startswith(("_", "TEMPLATE")):
            continue
        for r in _read_jsonl(f):
            r.setdefault("needs_context", False)
            r.setdefault("pair_id", None)
            r.setdefault("should_cache_hit", None)
            r["_file"] = f.name
            rows.append(r)
    if validate:
        errs = validate_evalset(rows)
        if errs:
            raise ValueError("eval set has errors:\n  " + "\n  ".join(errs[:30]))
    return rows


def load_bitext(raw_dir: Path = RAW, allow_download: bool = True) -> list[dict]:
    """Bitext rows {instruction, intent, category, response}. Cached at eval/data/raw/bitext.parquet."""
    import pandas as pd
    cache = Path(raw_dir) / "bitext.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(BITEXT_REPO, BITEXT_FILE, repo_type="dataset", local_files_only=True)
        except Exception:
            if not allow_download:
                raise
            log.info("downloading %s (~19 MB)", BITEXT_REPO)
            path = hf_hub_download(BITEXT_REPO, BITEXT_FILE, repo_type="dataset")
        df = pd.read_csv(path)[["instruction", "intent", "category", "response"]]
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df.to_dict("records")


_CAT_RULES = [
    ("refund", r"refund|money back|reimburs"),
    ("returns", r"\breturn|exchange|send (it )?back|reverse.pickup"),
    ("payment", r"\bpay|upi|\bcard\b|emi\b|cash on delivery|\bcod\b|wallet|gift card|invoice|gst|nestcoin"),
    ("shipping", r"ship|deliver|courier|express|pincode|international|customs|same.day|next.day"),
    ("account", r"account|password|log ?in|sign ?in|\botp\b|mobile number|privacy|personal data|plus member"),
    ("product", r"warrant|install|assembl|mattress|laptop|phone|tv\b|television|price match|price drop|shopnest care"),
    ("order", r"\border|cancel|track"),
]


def infer_category(text: str) -> str:
    t = (text or "").lower()
    return next((c for c, pat in _CAT_RULES if re.search(pat, t)), "other")


def load_kb(require: bool = False) -> tuple[list[dict], Optional[Callable], str]:
    """KB questions + retriever from eval.kb (context workstream). Returns ([], None, reason) if unavailable."""
    try:
        from eval import kb as kbmod
        fn, retrieve = getattr(kbmod, "kb_questions"), getattr(kbmod, "retrieve")
        raw = fn()
    except Exception as e:  # module missing or broken: skip the KB slice
        if require:
            raise
        return [], None, f"eval.kb unavailable ({type(e).__name__}: {str(e)[:120]})"
    out = []
    for i, q in enumerate(raw):
        if isinstance(q, str):
            q = {"question": q}
        elif isinstance(q, (tuple, list)):
            q = {"question": q[0], "answer": q[1] if len(q) > 1 else None}
        text = q.get("query") or q.get("question") or q.get("q") or q.get("text")
        if not text:
            continue
        paras = q.get("paraphrases") or q.get("variants") or q.get("alternates") or []
        if isinstance(paras, str):
            paras = [paras]
        out.append({"id": str(q.get("id") or q.get("qid") or f"kbq-{i:03d}"), "query": text.strip(),
                    "reference": q.get("reference") or q.get("answer") or q.get("a"),
                    "category": q.get("category") if q.get("category") in CATEGORIES else infer_category(text),
                    "paraphrases": [p.strip() for p in paras if p and p.strip() != text.strip()],
                    "kb_type": q.get("type"), "key_facts": q.get("key_facts") or []})

    def _retrieve(query: str, k: int = KB_K) -> list[str]:
        docs = retrieve(query, k=k)
        return [d if isinstance(d, str) else (d.get("text") or str(d)) for d in docs]
    return out, _retrieve, f"eval.kb: {len(out)} questions"


# ------------------------------------------------------------------------------------------- the builder
class _Cluster:
    __slots__ = ("cid", "source", "intent", "category", "items", "used", "faq", "first", "extra")

    def __init__(self, cid, source, intent, category, items, faq, extra=None):
        self.cid, self.source, self.intent, self.category, self.faq = cid, source, intent, category, faq
        self.items = items        # list of (item_id, query, reference); items[0] is the canonical phrasing
        self.used: list[int] = []
        self.first = None
        self.extra = extra or {}


def _zipf_weights(n: int, s: float = ZIPF_S) -> list[float]:
    return [1.0 / (r ** s) for r in range(1, n + 1)]


def _surface_variant(q: str, rng: random.Random) -> str:
    """A different string with the same exact-cache key (normalize_query): case, trailing punctuation, spacing."""
    opts = [q.lower(), q.upper() if len(q) < 40 else q.lower(), q.rstrip("?.! ") + "?", q.rstrip("?.! ") + "!!",
            q[:1].upper() + q[1:], "  ".join(q.split(" ", 1)) if " " in q else q + "?"]
    rng.shuffle(opts)
    for v in opts:
        if v != q and normalize_query(v) == normalize_query(q):
            return v
    return q + " ?" if normalize_query(q + " ?") == normalize_query(q) else q


class _BitextIndex:
    def __init__(self, rows: list[dict]):
        self.by: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.rows = rows
        for i, r in enumerate(rows):
            sig = slot_signature(r["instruction"])
            # a literal number or slot value outside a slot ("bill #00108", "delete my standard account") is an
            # unlabelled entity: keep such phrasings out of the generic (no-slot) clusters
            if r["intent"] in INTENT_CATEGORY and not (not sig and (re.search(r"\d", r["instruction"])
                                                                    or _POOL_WORDS.search(r["instruction"]))):
                self.by[(EQUIVALENT_INTENTS.get(r["intent"], r["intent"]), sig)].append(i)
        self.intents = sorted({k[0] for k in self.by})
        self.slot_sigs = {it: sorted([s for (i2, s) in self.by if i2 == it and s and len(self.by[(it, s)]) >= 20])
                          for it in self.intents}

    @staticmethod
    def cid(intent: str, values: dict) -> str:
        return f"bx:{intent}" + (":" + "|".join(f"{k}={v}" for k, v in sorted(values.items())) if values else "")

    def item(self, idx: int, values: dict, tag: str) -> tuple[str, str, str]:
        r = self.rows[idx]
        suffix = "-" + hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:6] if values else ""
        return (f"bx-{idx}{suffix}", fill_slots(r["instruction"], values, tag), fill_slots(r["response"], values, tag))


def _unique_stream(bx: _BitextIndex, rng: random.Random, n: int, used_norm: set,
                   used_cids: Optional[set] = None) -> list[_Cluster]:
    """Seeded stream of distinct Bitext clusters, round-robin over intents (generic cluster first, then slot clusters)."""
    used_orders: set = set()
    out: list[_Cluster] = []
    generic_done: set = set()
    cids: set = used_cids if used_cids is not None else set()
    order = list(bx.intents)
    rng.shuffle(order)
    stale = 0
    while len(out) < n:
        progressed = False
        for it in order:
            if len(out) >= n:
                break
            if it not in generic_done and bx.by.get((it, "")):
                generic_done.add(it)
                sig, values = "", {}
            elif bx.slot_sigs[it]:
                sig = rng.choice(bx.slot_sigs[it])
                values = _new_slot_values(sig, rng, used_orders)
            else:
                continue
            cid = bx.cid(it, values)
            if cid in cids:                          # small slot pools (account types) can repeat a value set
                continue
            cids.add(cid)
            idxs = list(bx.by[(it, sig)])
            rng.shuffle(idxs)
            items = []
            for i in idxs[:60]:                      # 60 phrasings per cluster is plenty for paraphrase duplicates
                item = bx.item(i, values, cid)
                key = normalize_query(item[1])
                if key in used_norm:
                    continue
                used_norm.add(key)
                items.append(item)
            if items:
                out.append(_Cluster(cid, "bitext", it, INTENT_CATEGORY[it], items, faq=not values))
                progressed = True
        stale = 0 if progressed else stale + 1
        if stale > 50:
            raise ValueError("ran out of distinct Bitext clusters")
    return out


_WRAPPERS = ["Hi, {q}", "{q} Thanks!", "Quick question: {q}", "Hello, {lq}", "{q} Please help.",
             "Could you tell me: {q}", "Hey team, {lq}", "{q} Thank you in advance."]


def _wrapped(q: str, rng: random.Random, n_prev: int = 0) -> str:
    """A conversational wrapper around the same question: same meaning, different exact-cache key."""
    lq = q[:1].lower() + q[1:] if q[:2] != q[:2].upper() else q
    start = int(hashlib.sha256(q.encode()).hexdigest(), 16) % len(_WRAPPERS)
    return _WRAPPERS[(start + n_prev) % len(_WRAPPERS)].format(q=q.strip(), lq=lq.strip())


def _jaccard(a: str, b: str) -> float:
    x, y = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(x & y) / max(1, len(x | y))


def _bitext_trap_pairs(bx: _BitextIndex, rng: random.Random, n_pairs: int, used_norm: set,
                       used_cids: Optional[set] = None) -> list[list[dict]]:
    """Near-miss pairs from Bitext: cross-intent with the same slot values, or the same template with a different
    order number / amount / account type / country."""
    pairs: list[list[dict]] = []
    used_orders: set = set()
    used_cids = used_cids if used_cids is not None else set()
    kinds = ["cross_intent", "entity"]
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        kind = kinds[len(pairs) % 2]
        if kind == "cross_intent":
            ia, ib = rng.choice(CROSS_INTENT_PAIRS)
            common = sorted(set(bx.slot_sigs.get(ia, [])) & set(bx.slot_sigs.get(ib, [])))
            if not common:
                continue
            sig = rng.choice(common)
            if not bx.by.get((ia, sig)) or not bx.by.get((ib, sig)):
                continue
            va = vb = _new_slot_values(sig, rng, used_orders)
            a_idx = rng.choice(bx.by[(ia, sig)])
            cands = rng.sample(bx.by[(ib, sig)], min(300, len(bx.by[(ib, sig)])))
            b_idx = max(cands, key=lambda j: _jaccard(bx.rows[a_idx]["instruction"], bx.rows[j]["instruction"]))
        else:
            ia = ib = rng.choice([it for it in bx.intents if bx.slot_sigs[it]])
            sig = rng.choice(bx.slot_sigs[ia])
            va = _new_slot_values(sig, rng, used_orders)
            vb = dict(va)
            for k in va:                                   # change exactly the entity, keep everything else
                if k == "Order Number":
                    d = list(va[k][3:])
                    j = rng.randrange(len(d) - 1)
                    d[j], d[j + 1] = d[j + 1], d[j]
                    vb[k] = "SN-" + "".join(d) if d != list(va[k][3:]) else f"SN-{int(va[k][3:]) + 1}"
                else:
                    vb[k] = _new_slot_values(k, rng, used_orders)[k]
            if vb == va:
                continue
            a_idx = b_idx = rng.choice(bx.by[(ia, sig)])
        tag = f"bxtp-{len(pairs):03d}"
        cids = (bx.cid(ia, va), bx.cid(ib, vb))
        if cids[0] == cids[1] or any(c in used_cids for c in cids):
            continue
        rows = []
        for side, (it, idx, vals), cid in zip("ab", ((ia, a_idx, va), (ib, b_idx, vb)), cids):
            iid, q, ref = bx.item(idx, vals, tag + side)
            rows.append({"item_id": f"{tag}{side}", "cluster_id": cid, "query": q, "reference": ref,
                         "category": INTENT_CATEGORY[it], "intent": it, "source": "trap", "pair_id": tag,
                         "trap_kind": kind, "trap_origin": "bitext"})
        if rows[0]["query"] == rows[1]["query"] or any(normalize_query(r["query"]) in used_norm for r in rows):
            continue
        for r in rows:
            used_norm.add(normalize_query(r["query"]))
        used_cids.update(cids)
        pairs.append(rows)
    return pairs


def eval_cluster(r: dict) -> str:
    """Cluster of a hand-written row: its own, unless `same_as` names a KB question or another eval row that asks the
    same thing (so a cache hit between them is correct, not a false hit)."""
    same = r.get("same_as")
    if same:
        return f"kb:{same}" if str(same).startswith("kbq-") else f"eval:{same}"
    return f"eval:{r['id']}"


def _evalset_trap_pairs(evalset: list[dict]) -> list[list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in evalset:
        if r.get("type") == "trap_pair" and r.get("pair_id"):
            by[r["pair_id"]].append(r)
    pairs = []
    for pid in sorted(by):
        rows = []
        for r in by[pid]:
            rows.append({"item_id": f"ev-{r['id']}", "cluster_id": eval_cluster(r), "query": r["query"],
                         "reference": r["reference"], "category": r["category"], "intent": f"eval:{r['id']}",
                         "source": "trap", "pair_id": pid, "trap_kind": "evalset", "trap_origin": "evalset",
                         "author": r.get("author"), "needs_context": bool(r.get("needs_context"))})
        if len(rows) == 2:
            pairs.append(rows)
    return pairs


def build_trace(size: int = 600, dup_rate: float = 0.3, seed: int = 7, *, bitext: Optional[list[dict]] = None,
                evalset: Optional[list[dict]] = None, kb: Optional[tuple] = None) -> tuple[list[dict], dict]:
    """Return (rows, stats). Sources can be injected (tests); otherwise they are loaded from disk / eval.kb."""
    if not 0 <= dup_rate < 0.9:
        raise ValueError("dup_rate must be in [0, 0.9)")
    bitext = bitext if bitext is not None else load_bitext()
    evalset = evalset if evalset is not None else load_evalset()
    kb_questions, retrieve, kb_status = kb if kb is not None else load_kb()
    if not kb_questions:
        log.warning("KB slice skipped: %s", kb_status)

    used_norm: set = set()
    bx = _BitextIndex(bitext)
    trap_rng, kb_rng = random.Random(f"{seed}:trap"), random.Random(f"{seed}:kb")

    # ---- traps (fixed across dup-rate variants)
    n_trap_target = int(round(TRAP_SHARE * size)) // 2 * 2
    ev_pairs = _evalset_trap_pairs(evalset)
    trap_rng.shuffle(ev_pairs)
    ev_pairs = ev_pairs[: n_trap_target // 2]
    for p in ev_pairs:
        for r in p:
            used_norm.add(normalize_query(r["query"]))
    used_cids: set = set()
    bx_pairs = _bitext_trap_pairs(bx, trap_rng, n_trap_target // 2 - len(ev_pairs), used_norm, used_cids)
    trap_pairs = ev_pairs + bx_pairs
    n_trap = 2 * len(trap_pairs)

    # ---- unique items and duplicate budget
    n_dup = int(round(dup_rate * size))
    n_unique = size - n_trap - n_dup
    if n_unique <= 0:
        raise ValueError("size too small for this dup rate")
    kb_frac = SHARES["kb"] / (SHARES["kb"] + SHARES["bitext"]) if kb_questions else 0.0
    kb_pool = list(kb_questions)
    kb_rng.shuffle(kb_pool)
    u_kb = min(len(kb_pool), int(round(n_unique * kb_frac)))
    # KB duplicates make up the KB slice's shortfall (small KB pool), capped at 60% of all duplicates
    kb_dups = min(n_dup, max(0, int(round((size - n_trap) * kb_frac)) - u_kb), int(round(0.6 * n_dup))) if u_kb else 0

    clusters: list[_Cluster] = []
    for q in kb_pool[:u_kb]:
        items = [(f"kb-{q['id']}", q["query"], q.get("reference"))]
        items += [(f"kb-{q['id']}-p{j + 1}", p, q.get("reference")) for j, p in enumerate(q["paraphrases"])]
        items = [it for it in items if normalize_query(it[1]) not in used_norm]
        for it in items:
            used_norm.add(normalize_query(it[1]))
        if items:
            clusters.append(_Cluster(f"kb:{q['id']}", "kb", f"kb:{q['id']}", q["category"], items, faq=True,
                                     extra={k: q[k] for k in ("key_facts", "kb_type") if q.get(k)}))
    u_bx = n_unique - len(clusters)
    clusters += _unique_stream(bx, random.Random(f"{seed}:uniques"), u_bx, used_norm, used_cids)

    # ---- assign duplicates: kinds, then clusters (source-proportional, Zipf within source, FAQ head first)
    rng = random.Random(f"{seed}:dups:{dup_rate}")
    n_exactish = int(round(n_dup / 3))
    kinds = (["exact"] * (n_exactish - n_exactish // 2) + ["surface"] * (n_exactish // 2)
             + ["paraphrase"] * (n_dup - n_exactish))
    rng.shuffle(kinds)
    by_src: dict[str, list[_Cluster]] = defaultdict(list)
    for c in clusters:
        by_src[c.source].append(c)
    ranked, weights = {}, {}
    for src, cs in by_src.items():
        head = [c for c in cs if c.faq]
        tail = [c for c in cs if not c.faq]
        rng.shuffle(head)
        rng.shuffle(tail)
        ranked[src] = head + tail
        weights[src] = _zipf_weights(len(ranked[src]))
    srcs = ["kb"] * kb_dups + ["bitext"] * (n_dup - kb_dups)
    rng.shuffle(srcs)
    plan: dict[str, list[str]] = defaultdict(list)
    for kind, src in zip(kinds, srcs):
        src = src if ranked.get(src) else next(s for s in ranked if ranked[s])
        c = rng.choices(ranked[src], weights=weights[src])[0]
        plan[c.cid].append(kind)

    # ---- order: shuffle all occurrences; a cluster's first occurrence is its unique item
    events = [c for c in clusters for _ in range(1 + len(plan[c.cid]))]
    rng.shuffle(events)
    rows: list[dict] = []
    ctx_cache: dict[str, list[str]] = {}

    def context_for(query: str) -> list[str]:
        if retrieve is None:
            return []
        if query not in ctx_cache:
            ctx_cache[query] = list(retrieve(query, k=KB_K))
        return ctx_cache[query]

    for c in events:
        base = {"cluster_id": c.cid, "category": c.category, "source": c.source, "intent": c.intent, **c.extra}
        if c.first is None:
            iid, q, ref = c.items[0]
            c.used.append(0)
            row = {**base, "item_id": iid, "query": q, "reference": ref, "dup_of": None, "dup_kind": None,
                   "context": context_for(q) if c.source == "kb" else []}
            c.first = row
        else:
            kind = plan[c.cid].pop()
            unused = [j for j in range(len(c.items)) if j not in c.used]
            if kind == "paraphrase" and not unused:
                kind = "wrapped" if c.source == "kb" else "surface"
            if kind == "wrapped":
                src_row = c.first
                n_w = sum(1 for r in rows if r["cluster_id"] == c.cid and r["dup_kind"] == "wrapped")
                q = _wrapped(c.items[0][1], rng, n_w)
                iid, ref = f"{c.items[0][0]}-w{n_w + 1}", c.items[0][2]
                ctx = context_for(q)
            elif kind == "paraphrase":
                j = rng.choice(unused)
                c.used.append(j)
                iid, q, ref = c.items[j]
                src_row = c.first
                ctx = context_for(q) if c.source == "kb" else []
            else:
                src_row = rng.choice([r for r in rows if r["cluster_id"] == c.cid and r["dup_kind"] != "surface"]
                                     or [c.first])
                q = src_row["query"] if kind == "exact" else _surface_variant(src_row["query"], rng)
                iid, ref, ctx = src_row["item_id"].split("~")[0], src_row["reference"], list(src_row["context"])
            n_rep = sum(1 for r in rows if r["item_id"].split("~")[0] == iid)
            row = {**base, "item_id": f"{iid}~{n_rep}" if n_rep else iid, "query": q, "reference": ref,
                   "dup_of": src_row, "dup_kind": kind, "context": ctx}
        rows.append(row)

    # ---- insert trap pairs close together
    for pair in trap_pairs:
        a, b = (dict(r) for r in pair)
        for r in (a, b):
            needs_ctx = r.pop("needs_context", False)
            r.update(dup_of=None, dup_kind=None, context=context_for(r["query"]) if needs_ctx else [])
        i = trap_rng.randint(0, len(rows))
        rows.insert(i, a)
        rows.insert(min(len(rows), i + 1 + trap_rng.randint(0, 2)), b)

    # ---- finalise: positions, dup_of as the position of the earlier equivalent request
    first_of: dict[str, dict] = {}
    for pos, r in enumerate(rows):
        r["pos"] = pos
        if r["dup_of"] is None and r["cluster_id"] in first_of:     # e.g. an eval trap row `same_as` a KB question
            r["dup_of"], r["dup_kind"] = first_of[r["cluster_id"]], "cross_source"
        first_of.setdefault(r["cluster_id"], r)
    out = []
    for r in rows:
        dup = r["dup_of"]
        rec = {"pos": r["pos"], "item_id": r["item_id"], "cluster_id": r["cluster_id"], "query": r["query"],
               "category": r["category"], "context": r["context"], "reference": r["reference"], "source": r["source"],
               "dup_of": dup["pos"] if dup is not None else None, "dup_kind": r["dup_kind"], "intent": r["intent"]}
        for k in ("pair_id", "trap_kind", "trap_origin", "author", "kb_type", "key_facts"):
            if r.get(k) is not None:
                rec[k] = r[k]
        out.append(rec)
    stats = trace_stats(out)
    stats.update(seed=seed, size_requested=size, dup_rate_requested=dup_rate, kb_status=kb_status,
                 bitext_license=BITEXT_LICENSE, evalset_trap_pairs=len(ev_pairs), bitext_trap_pairs=len(bx_pairs))
    return out, stats


def trace_stats(rows: list[dict]) -> dict:
    n = len(rows)
    src = Counter(r["source"] for r in rows)
    dups = [r for r in rows if r.get("dup_of") is not None]
    trap_rows = [r for r in rows if r["source"] == "trap"]
    by_pair: dict[str, list[int]] = defaultdict(list)
    for r in trap_rows:
        by_pair[r.get("pair_id")].append(r["pos"])
    gaps = [abs(p[1] - p[0]) for p in by_pair.values() if len(p) == 2]
    return {"n": n, "sources": dict(src), "source_share": {k: round(v / n, 3) for k, v in src.items()} if n else {},
            "categories": dict(Counter(r["category"] for r in rows)),
            "dup_rate": round(len(dups) / n, 4) if n else 0.0, "dup_kinds": dict(Counter(r["dup_kind"] for r in dups)),
            "clusters": len({r["cluster_id"] for r in rows}), "trap_pairs": len(by_pair),
            "trap_gap_max": max(gaps) if gaps else None, "with_context": sum(1 for r in rows if r["context"])}


def write_jsonl(rows: list[dict], path: Path, with_sha: bool = True) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    if with_sha:
        path.with_suffix(".sha256").write_text(f"{sha}  {path.name}\n")
    return sha


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant_path(dup_rate: float, headline: float = 0.3) -> Path:
    return TRACE if abs(dup_rate - headline) < 1e-9 else DATA / f"trace_v1_dup{int(round(dup_rate * 100)):02d}.jsonl"


# ------------------------------------------------------------------------------------------- CI subset
def build_ci_subset(seed: int = 7, n_eval: int = 30, n_trap_pairs: int = 20, *, bitext: Optional[list[dict]] = None,
                    evalset: Optional[list[dict]] = None, kb: Optional[tuple] = None) -> list[dict]:
    """~30 eval-set rows + ~20 trap pairs (eval set first, Bitext near-misses as top-up) + a few legitimate repeats.

    Same row format as the trace, plus `type`/`eval_id`/`author` for eval rows. Pair members are adjacent."""
    evalset = evalset if evalset is not None else load_evalset()
    _, retrieve, _ = kb if kb is not None else load_kb()
    rng = random.Random(f"{seed}:ci")
    used_norm = {normalize_query(r["query"]) for r in evalset}

    def ctx(q: str, needs: bool) -> list[str]:
        return list(retrieve(q, k=KB_K)) if (needs and retrieve is not None) else []

    pool = [r for r in evalset if r.get("type") != "trap_pair"]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["id"]):
        by_cat[r["category"]].append(r)
    for v in by_cat.values():
        rng.shuffle(v)
    picked: list[dict] = []
    while len(picked) < min(n_eval, len(pool)):
        for cat in CATEGORIES:
            if by_cat.get(cat) and len(picked) < n_eval:
                picked.append(by_cat[cat].pop())
    blocks: list[list[dict]] = []
    for r in picked:
        blocks.append([{"item_id": f"ev-{r['id']}", "cluster_id": eval_cluster(r), "query": r["query"],
                        "category": r["category"], "context": ctx(r["query"], r.get("needs_context")),
                        "reference": r["reference"], "source": "eval", "dup_of": None, "dup_kind": None,
                        "intent": f"eval:{r['id']}", "type": r["type"], "eval_id": r["id"], "author": r.get("author")}])
    ev_pairs = _evalset_trap_pairs(evalset)[:n_trap_pairs]
    need = n_trap_pairs - len(ev_pairs)
    bx_pairs = []
    if need > 0:
        bitext = bitext if bitext is not None else load_bitext()
        bx_pairs = _bitext_trap_pairs(_BitextIndex(bitext), rng, need, used_norm)
    for p in ev_pairs + bx_pairs:
        blk = []
        for r in p:
            r = dict(r)
            needs = r.pop("needs_context", False)
            r.update(dup_of=None, dup_kind=None, context=ctx(r["query"], needs), type="trap_pair")
            blk.append(r)
        blocks.append(blk)
    rng.shuffle(blocks)
    rows = [r for b in blocks for r in b]
    # legitimate repeats (exact + surface) of a few answerable rows, placed later in the subset
    singles = [b[0] for b in blocks if len(b) == 1]
    for j, src in enumerate(rng.sample(singles, min(8, len(singles)))):
        kind = "exact" if j % 2 == 0 else "surface"
        q = src["query"] if kind == "exact" else _surface_variant(src["query"], rng)
        at = rows.index(src)
        slots = [i for i in range(at + 1, len(rows) + 1)       # never split a trap pair
                 if i == len(rows) or not rows[i].get("pair_id") or rows[i].get("pair_id") != rows[i - 1].get("pair_id")]
        rows.insert(rng.choice(slots), {**src, "item_id": src["item_id"] + "~1", "query": q, "dup_of": src,
                                        "dup_kind": kind})
    for pos, r in enumerate(rows):
        r["pos"] = pos
    for r in rows:
        if isinstance(r.get("dup_of"), dict):
            r["dup_of"] = r["dup_of"]["pos"]
    return rows


# ------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--dup-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", help="output path (default: trace_v1.jsonl, or trace_v1_dupXX.jsonl for other rates)")
    ap.add_argument("--all-variants", action="store_true", help="write dup-rate variants 0, 0.15, 0.3 and 0.5")
    ap.add_argument("--ci-subset", action="store_true", help="write eval/data/ci_subset.jsonl instead")
    ap.add_argument("--stats", metavar="TRACE", help="print stats for an existing trace file and exit")
    args = ap.parse_args(argv)
    if args.stats:
        rows = _read_jsonl(Path(args.stats))
        print(json.dumps({**trace_stats(rows), "sha256": file_sha256(Path(args.stats))}, indent=2))
        return 0
    if args.ci_subset:
        rows = build_ci_subset(seed=args.seed)
        out = Path(args.out) if args.out else CI_SUBSET
        sha = write_jsonl(rows, out, with_sha=False)
        st = trace_stats(rows)
        print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: {st['n']} rows, "
              f"{st['trap_pairs']} trap pairs, sources {st['sources']}, sha256 {sha[:16]}")
        return 0
    rates = [0.0, 0.15, 0.3, 0.5] if args.all_variants else [args.dup_rate]
    for rate in rates:
        rows, st = build_trace(args.size, rate, args.seed)
        out = Path(args.out) if (args.out and not args.all_variants) else variant_path(rate)
        sha = write_jsonl(rows, out)
        st["sha256"] = sha
        print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} (sha256 {sha})")
        print(json.dumps(st, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
