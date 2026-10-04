"""ShopNest store-policy knowledge base: chunking, deliberately generous retrieval, and seed workload questions.

- `load_kb()` chunks `eval/data/kb/*.md` into ~150-250-token passages: {"id", "title", "text"}.
- `retrieve(query, k=8)` is naive "stuff 8 chunks" RAG: a bge-small bi-encoder top-k with no reranking and no score
  floor. It returns plenty of loosely related text on purpose, so the context optimiser has something to cut.
- `kb_questions()` gives ~60 realistic customer questions with reference answers, gold doc ids and `key_facts` (the
  numbers and short phrases an answer needs). They are workload for the trace and for the LLM-free evidence-retention
  proxy in `eval/compression_eval.py`, not the graded eval set (`"author": "seed"`).

The KB is fictional; see eval/data/kb/README.md.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from costguard.config import ROOT
from costguard.tokens import count_text

KB_DIR = ROOT / "eval" / "data" / "kb"
INDEX_PATH = ROOT / "data" / "runtime" / "kb_index.npz"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS = 150, 250


# ----------------------------------------------------------------------------------------------- chunking

def _doc_units(md: str) -> tuple[str, list[tuple[str, str]]]:
    """Split one markdown doc into (section, block) units. A block is a paragraph, list or table."""
    lines = md.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("# ") else ""
    units, section, buf = [], "", []

    def flush():
        if buf and "\n".join(buf).strip():
            units.append((section, "\n".join(buf).strip()))
        buf.clear()

    for line in lines[1:] if title else lines:
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
        elif not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    return title, units


def _split_long(block: str, max_tokens: int) -> list[str]:
    """Split an oversized block at line, then sentence boundaries (never mid-sentence). Small blocks pass as-is."""
    if count_text(block) <= max_tokens:
        return [block]
    out, cur = [], ""
    for line in block.split("\n"):
        segs = [line] if count_text(line) <= max_tokens else re.split(r"(?<=[.!?])\s+", line)
        for j, seg in enumerate(segs):
            cand = (cur + ("\n" if j == 0 else " ") + seg) if cur else seg
            if cur and count_text(cand) > max_tokens:
                out.append(cur)
                cur = seg
            else:
                cur = cand
    if cur:
        out.append(cur)
    return out


def chunk_doc(doc_id: str, md: str, min_tokens: int = CHUNK_MIN_TOKENS, max_tokens: int = CHUNK_MAX_TOKENS) -> list[dict]:
    """Greedy packing of whole blocks into ~min..max-token chunks, each prefixed with the doc title (a breadcrumb,
    as most RAG chunkers do). Section headings travel with their first block."""
    title, units = _doc_units(md)
    parts: list[tuple[str, str]] = []
    prev_section = None
    for section, block in units:
        head = f"## {section}\n" if section and section != prev_section else ""
        prev_section = section
        for i, piece in enumerate(_split_long(block, max_tokens)):
            parts.append((section, (head if i == 0 else "") + piece))

    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0
    for section, text in parts:
        t = count_text(text)
        # flush when full; an under-filled chunk may overflow a little, but never past max + 50
        if cur and cur_tok + t > max_tokens and (cur_tok >= min_tokens or cur_tok + t > max_tokens + 50):
            chunks.append(cur)
            cur, cur_tok = [], 0
        if not cur and not text.startswith("## ") and section:
            text = f"## {section} (continued)\n{text}"
            t = count_text(text)
        cur.append(text)
        cur_tok += t
    if cur:
        # a short tail is merged into the previous chunk instead of becoming a tiny passage
        if chunks and cur_tok < min_tokens // 2:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return [{"id": f"{doc_id}#{i}", "doc_id": doc_id, "title": title, "text": f"# {title}\n" + "\n\n".join(c)}
            for i, c in enumerate(chunks)]


@lru_cache(maxsize=1)
def _load_kb_cached(kb_dir: str) -> tuple[dict, ...]:
    out = []
    for p in sorted(Path(kb_dir).glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        out.extend(chunk_doc(p.stem, p.read_text()))
    return tuple(out)


def load_kb(kb_dir: Optional[Path] = None) -> list[dict]:
    """All KB chunks: {"id": "<doc>#<n>", "doc_id", "title", "text"}."""
    return [dict(c) for c in _load_kb_cached(str(kb_dir or KB_DIR))]


def kb_doc_ids() -> list[str]:
    return sorted({c["doc_id"] for c in load_kb()})


# ----------------------------------------------------------------------------------------------- retrieval

def _kb_hash(chunks: list[dict]) -> str:
    h = hashlib.sha256(EMBED_MODEL.encode())
    for c in chunks:
        h.update(c["id"].encode() + b"\x00" + c["text"].encode() + b"\x01")
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def _index() -> tuple[list[dict], np.ndarray]:
    from costguard.context.optimizer import embed_passages
    chunks = load_kb()
    h = _kb_hash(chunks)
    if INDEX_PATH.exists():
        try:
            z = np.load(INDEX_PATH, allow_pickle=False)
            if str(z["kb_hash"]) == h and z["emb"].shape[0] == len(chunks):
                return chunks, z["emb"]
        except Exception:
            pass
    emb = embed_passages([c["text"] for c in chunks])
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INDEX_PATH, emb=emb, ids=np.array([c["id"] for c in chunks]), kb_hash=np.array(h))
    return chunks, emb


def retrieve_chunks(query: str, k: int = 8) -> list[dict]:
    """Top-k chunks by bi-encoder cosine, with "score". No reranking, no floor: deliberately generous."""
    from costguard.context.optimizer import embed_queries
    chunks, emb = _index()
    q = embed_queries([query])[0]
    sims = emb @ q
    top = np.argsort(-sims)[:k]
    return [{**chunks[i], "score": float(sims[i])} for i in top]


def retrieve(query: str, k: int = 8) -> list[str]:
    """Contract API: texts of the top-k chunks, best first."""
    return [c["text"] for c in retrieve_chunks(query, k)]


# ----------------------------------------------------------------------------------------------- key-fact matching

_DASHES = re.compile(r"[‐-―−]")
_RUPEE = re.compile(r"\b(?:rs\.?|inr)\s*(?=\d)", re.I)


def fact_tokens(text: str) -> list[str]:
    """Normalised tokens for robust substring checks: lowercase, unified dashes, '₹' as its own token,
    Indian digit grouping removed (1,00,000 -> 100000), Rs./INR -> ₹."""
    t = _DASHES.sub("-", text or "")
    t = _RUPEE.sub("₹", t)
    t = re.sub(r"(?<=\d), ?(?=\d{2,3}\b)", "", t)     # "1,499" and LLMLingua's "10, 000"
    return re.findall(r"₹|[a-z0-9%]+", t.lower())


# function words a token-dropping compressor removes without changing the fact ("refuse the delivery" -> "refuse delivery")
_FILLER = frozenset("a an the of to for from by with in on at is are be and or your you it its this that".split())


def fact_present(fact: str, text: str, lenient: bool = False) -> bool:
    """True if the fact's token sequence appears contiguously in the text's token sequence.
    `lenient=True` ignores filler words on both sides first (fairer to token-level compressors)."""
    f, t = fact_tokens(fact), fact_tokens(text)
    if lenient:
        f, t = [w for w in f if w not in _FILLER] or f, [w for w in t if w not in _FILLER]
    if not f:
        return True
    return (" " + " ".join(f) + " ") in (" " + " ".join(t) + " ")


def facts_present(facts: list[str], text: str, lenient: bool = False) -> bool:
    return all(fact_present(f, text, lenient) for f in facts)


# ----------------------------------------------------------------------------------------------- seed questions

_UNANSWERABLE_REF = ("Not covered by the ShopNest policy knowledge base. The assistant should say it is not sure and "
                     "offer to connect the customer with a human agent.")

# (query, reference, doc_ids, category, type, key_facts)
_Q = [
    # ---- single-fact
    ("How many days do I have to return a pair of jeans?",
     "Apparel can be returned within 30 days of delivery.", ["returns-policy"], "returns", "single", ["30 days"]),
    ("What's the return window for a laptop?",
     "Electronics, including laptops, can be returned within 10 days of delivery.", ["returns-policy"], "returns",
     "single", ["10 days from delivery"]),
    ("Is there a fee if I return something just because I changed my mind?",
     "Yes. A ₹79 reverse-pickup fee is deducted from the refund for change-of-mind returns; ShopNest Plus members "
     "don't pay it.", ["returns-policy"], "returns", "single", ["₹79"]),
    ("Courier pickup isn't available at my pincode. How do I send my return back?",
     "Ship it yourself to the returns centre within 7 days of raising the return; ShopNest reimburses shipping up to "
     "₹150 as ShopNest Wallet balance.", ["returns-policy"], "returns", "single", ["₹150"]),
    ("Can I return earbuds after I've opened them?",
     "No. In-ear earphones and earbuds can't be returned once the hygiene seal is opened, unless they are damaged, "
     "defective or wrong.", ["returns-policy"], "returns", "single", ["hygiene seal"]),
    ("How long does a refund to UPI take?",
     "UPI refunds take 1–3 business days after the refund is initiated.", ["refunds"], "refund", "single",
     ["1-3 business days"]),
    ("I paid with my credit card. When will the refund show up?",
     "Card refunds take 5–7 business days after the refund is initiated.", ["refunds"], "refund", "single",
     ["5-7 business days"]),
    ("I paid cash on delivery. How do I get my money back?",
     "COD refunds go to the ShopNest Wallet within 2 hours, or to your bank account by NEFT in 5–7 business days "
     "after you add your account number and IFSC.", ["refunds", "cod-and-emi"], "refund", "single",
     ["within 2 hours", "NEFT"]),
    ("Do I get the COD fee back if I return the item?",
     "No. The ₹40 COD handling fee isn't refunded, unless the item was damaged, defective or wrong.",
     ["refunds", "cod-and-emi"], "refund", "single", ["₹40"]),
    ("What is the ARN on my refund message?",
     "ARN is the Acquirer Reference Number, the refund reference your bank can use to trace a card or net banking "
     "refund.", ["refunds"], "refund", "single", ["Acquirer Reference Number"]),
    ("Can I exchange my shirt for a bigger size?",
     "Yes. Apparel can be exchanged for another size or colour of the same product within 30 days of delivery, for "
     "free, once per item.", ["exchanges"], "returns", "single", ["30 days", "only once"]),
    ("Can I swap my new phone for a different colour?",
     "No. Electronics can't be exchanged for a different model, brand or colour; a defective item can be replaced "
     "with the same model within 10 days.", ["exchanges"], "returns", "single", ["different model, brand or colour"]),
    ("I exchanged a kurta for a larger size. Can I still return the new one?",
     "Yes. The exchanged item gets a fresh 30-day return window from its own delivery date, but it can't be "
     "exchanged again.", ["exchanges"], "returns", "single", ["fresh return window"]),
    ("How long does standard delivery take to Bengaluru?",
     "Bengaluru is a metro city, where standard delivery takes 2–4 business days.", ["shipping-domestic"],
     "shipping", "single", ["2-4 business days"]),
    ("What's the minimum order value for free shipping?",
     "Standard shipping is free on orders of ₹499 and above; below that the fee is ₹49.", ["shipping-domestic"],
     "shipping", "single", ["₹499", "₹49"]),
    ("Why is the delivery agent asking me for an OTP?",
     "Orders above ₹10,000 need a delivery OTP sent to your registered mobile number.", ["shipping-domestic"],
     "shipping", "single", ["delivery OTP", "₹10,000"]),
    ("What's the cut-off time for next-day delivery?",
     "Order by 12 noon in one of the 25 next-day cities; it costs ₹99, free for Plus members.",
     ["shipping-express"], "shipping", "single", ["12 noon"]),
    ("How much does same-day delivery cost?",
     "Same-day delivery costs ₹149 per order, or ₹49 for ShopNest Plus members.", ["shipping-express"], "shipping",
     "single", ["₹149"]),
    ("My express order arrived late. Do I get the express fee back?",
     "Yes. The express fee is refunded automatically to your ShopNest Wallet within 48 hours, unless the delay was "
     "due to a wrong address or nobody being available.", ["shipping-express"], "shipping", "single",
     ["refunded automatically"]),
    ("Do you ship to Germany?",
     "No. Outside India ShopNest ships only to 7 countries: UAE, Singapore, UK, USA, Canada, Australia and Nepal.",
     ["shipping-international"], "shipping", "single", ["7 countries"]),
    ("How much is shipping to the USA?",
     "₹2,499 per order for parcels up to 5 kg, with delivery in 7–14 business days.", ["shipping-international"],
     "shipping", "single", ["₹2,499"]),
    ("Who pays customs duty on an order shipped to Australia?",
     "The recipient pays customs duties and import taxes at delivery; orders ship duties unpaid.",
     ["shipping-international"], "shipping", "single", ["paid by the recipient"]),
    ("Can I cancel my order after it has shipped?",
     "No. Once shipped it can't be cancelled; you can refuse the delivery or return it after delivery.",
     ["order-changes-cancellation"], "order", "single", ["refuse the delivery"]),
    ("Can I change the delivery address after placing my order?",
     "Yes, before the order ships, but only to an address with the same pincode; otherwise cancel and reorder.",
     ["order-changes-cancellation"], "order", "single", ["same pincode"]),
    ("How long do I have to cancel an engraved bracelet order?",
     "Personalised or engraved products can be cancelled only within 6 hours of placing the order.",
     ["order-changes-cancellation"], "order", "single", ["6 hours"]),
    ("Which card networks do you accept?",
     "Visa, Mastercard, RuPay, American Express and Diners Club.", ["payments"], "payment", "single",
     ["RuPay", "Diners Club"]),
    ("Money was debited but my order failed. What happens now?",
     "The amount is reversed automatically: UPI within 48 hours, cards and net banking within 5–7 business days.",
     ["payments"], "payment", "single", ["reversed automatically"]),
    ("What's the maximum order value for cash on delivery?",
     "Cash on Delivery is available on orders up to ₹50,000.", ["cod-and-emi"], "payment", "single",
     ["up to ₹50,000"]),
    ("Is no-cost EMI available, and for which tenures?",
     "Yes, on orders of ₹10,000 and above with select bank credit cards, for 3 and 6 months.", ["cod-and-emi"],
     "payment", "single", ["3 and 6 months"]),
    ("How long is the warranty on a Nestra speaker?",
     "Nestra electronics carry a 1-year ShopNest warranty against manufacturing defects.",
     ["warranty-electronics"], "product", "single", ["1-year ShopNest warranty"]),
    ("How much does ShopNest Care cost for 2 extra years?",
     "2 extra years of ShopNest Care cost 12% of the product price.", ["warranty-electronics"], "product", "single",
     ["12%"]),
    ("Is a cracked phone screen covered under warranty?",
     "No, physical damage isn't covered unless you bought the Accidental and Liquid Damage plan.",
     ["warranty-electronics"], "product", "single", ["Accidental and Liquid Damage plan"]),
    ("The sole of my shoes came off after two months. Is that covered?",
     "Yes. Footwear has a 90-day quality promise against sole separation and broken stitching; ShopNest replaces or "
     "refunds the pair.", ["warranty-home-apparel"], "product", "single", ["90-day"]),
    ("How soon do I have to report a damaged item?",
     "Within 48 hours of delivery, with photos of the item, packaging and shipping label.", ["damaged-wrong-items"],
     "returns", "single", ["within 48 hours of delivery"]),
    ("Do I need an unboxing video to report a damaged TV?",
     "Yes, for electronics priced ₹10,000 and above a continuous unboxing video is required.",
     ["damaged-wrong-items"], "returns", "single", ["unboxing video", "₹10,000"]),
    ("My account got locked. How long until I can try again?",
     "After 5 failed sign-in attempts the account is locked for 30 minutes; you can also reset your password.",
     ["account-security"], "account", "single", ["30 minutes"]),
    ("How many NestCoins do I earn when I shop?",
     "1 NestCoin for every ₹100 spent after discounts; ShopNest Plus members earn 2.", ["nestcoins-and-plus"],
     "account", "single", ["1 NestCoin for every ₹100"]),
    ("How long is a ShopNest gift card valid?",
     "12 months from the date it is issued; validity can't be extended.", ["gift-cards"], "payment", "single",
     ["valid for 12 months"]),
    ("How much does split AC installation cost?",
     "Split AC installation costs ₹1,499 and includes up to 3 metres of copper pipe.", ["installation-services"],
     "product", "single", ["Split air conditioner installation", "₹1,499"]),
    ("Can I add my GSTIN to an invoice after the order is placed?",
     "No. The GSTIN must be added before you place the order; it can't be added after the invoice is generated.",
     ["bulk-business-orders"], "order", "single", ["before you place the order"]),
    ("How long do you keep recordings of my support calls?",
     "Support call recordings are kept for 90 days.", ["privacy-policy"], "account", "single", ["90 days"]),
    ("What are your customer care phone hours?",
     "Toll-free 1800-202-6378, 8 AM to 10 PM IST every day; chat is 24x7.", ["contact-escalation"], "other",
     "single", ["8 AM to 10 PM"]),
    ("When is the Diwali sale this year?",
     "The Big Nest Sale Diwali edition runs from 16 to 22 October 2026; Plus members get access 24 hours early.",
     ["big-nest-sale-terms"], "other", "single", ["16 to 22 October"]),
    ("I wear US size 10 in men's shoes. Which size should I order?",
     "UK/India 9, because US men's sizes are 1 more than the UK size.", ["size-guide"], "product", "single",
     ["1 more than the UK size"]),
    # ---- multi-hop (needs two docs)
    ("I paid for my laptop with UPI and want to return it. How long do I have, and how long will the refund take?",
     "Laptops can be returned within 10 days of delivery; UPI refunds then take 1–3 business days after the refund "
     "is initiated.", ["returns-policy", "refunds"], "returns", "multi_hop", ["10 days", "1-3 business days"]),
    ("I'm a Plus member. Do I pay for same-day delivery, and is there a fee if I return something I don't need?",
     "Same-day delivery costs ₹49 for Plus members, and Plus members never pay the ₹79 reverse-pickup fee.",
     ["shipping-express", "returns-policy", "nestcoins-and-plus"], "shipping", "multi_hop",
     ["₹49", "reverse-pickup fee"]),
    ("I bought a TV in the Big Nest Sale using the bank offer and want to return it. What's the window and how much "
     "do I get back?",
     "TVs are electronics, so the window is 10 days from delivery; you're refunded the amount you actually paid, not "
     "the pre-discount price.", ["big-nest-sale-terms", "returns-policy", "refunds"], "refund", "multi_hop",
     ["10 days", "amount you actually paid"]),
    ("Can I pay cash on delivery for a ₹60,000 phone? If not, which cards can I use?",
     "No, COD is only for orders up to ₹50,000. You can pay by Visa, Mastercard, RuPay, American Express or Diners "
     "Club cards (or UPI, net banking, EMI).", ["cod-and-emi", "payments"], "payment", "multi_hop",
     ["up to ₹50,000", "RuPay"]),
    ("My fridge was delivered yesterday but the installer comes later. Should I unbox it to check for damage?",
     "No. Don't unbox large appliances; the technician unboxes it, and damage found then is treated as reported "
     "within 48 hours of delivery.", ["installation-services", "damaged-wrong-items"], "product", "multi_hop",
     ["do not unbox", "treated as reported within 48 hours"]),
    ("I'm in Nepal. How much is shipping, and can I pay cash on delivery?",
     "Shipping to Nepal is ₹599 per order (5–8 business days). Cash on Delivery isn't available for international "
     "orders.", ["shipping-international", "cod-and-emi"], "shipping", "multi_hop",
     ["₹599", "not available for international orders"]),
    ("I returned a shirt I paid for with a gift card that has since expired. Where does the refund go, and does it "
     "expire?",
     "It goes to your ShopNest Wallet instead of the expired gift card, and Wallet balance never expires.",
     ["gift-cards", "refunds"], "refund", "multi_hop", ["ShopNest Wallet", "never expires"]),
    ("Two chairs in our bulk office order arrived broken and we only noticed 5 days after delivery. Is it too late?",
     "No. Bulk orders can report damaged, defective, missing or wrong items within 7 days of delivery, instead of the "
     "usual 48 hours.", ["bulk-business-orders", "damaged-wrong-items"], "order", "multi_hop",
     ["within 7 days of delivery"]),
    ("As a Plus member, how many NestCoins do I earn on a ₹20,000 phone in the Diwali sale, and when are they "
     "credited?",
     "Plus members earn 3 NestCoins per ₹100 during the sale, so 600 NestCoins, credited 10 days after delivery once "
     "the electronics return window closes.", ["big-nest-sale-terms", "nestcoins-and-plus"], "account", "multi_hop",
     ["3 NestCoins for every ₹100", "10 days after delivery"]),
    ("My UK parcel was refused because of customs charges. What will be refunded, and how long until it reaches my "
     "international card?",
     "Only the item price is refunded (not the international shipping fee); refunds to international cards take "
     "7–14 business days.", ["shipping-international", "refunds"], "refund", "multi_hop",
     ["item price only", "7-14 business days"]),
    ("Can I get a price match on a TV I'm buying in the Big Nest Sale?",
     "No. Neither Price Drop Protection nor Competitor Price Match applies to orders placed during a Big Nest Sale.",
     ["price-match", "big-nest-sale-terms"], "payment", "multi_hop",
     ["Neither Price Drop Protection nor Competitor Price Match"]),
    ("I want to return a ₹2,000 lamp I no longer need. I'm not a Plus member and paid by UPI. What do I get back and "
     "when?",
     "₹1,921: the ₹79 reverse-pickup fee is deducted for change-of-mind returns; the UPI refund takes 1–3 business "
     "days after the quality check.", ["returns-policy", "refunds"], "refund", "multi_hop",
     ["₹79", "1-3 business days"]),
    # ---- unanswerable (not in the KB)
    ("Do you sell groceries?", _UNANSWERABLE_REF, [], "product", "unanswerable", []),
    ("Can I pick up my order from a ShopNest store?", _UNANSWERABLE_REF, [], "order", "unanswerable", []),
    ("Do you have a student discount?", _UNANSWERABLE_REF, [], "other", "unanswerable", []),
    ("Do I get anything for referring a friend?", _UNANSWERABLE_REF, [], "other", "unanswerable", []),
    ("Can you gift-wrap my order?", _UNANSWERABLE_REF, [], "order", "unanswerable", []),
    ("Can I trade in my old phone when I buy a new one?", _UNANSWERABLE_REF, [], "payment", "unanswerable", []),
    ("What interest rate do your cardless EMI partners charge?", _UNANSWERABLE_REF, [], "payment", "unanswerable", []),
    ("Do you sell refurbished laptops?", _UNANSWERABLE_REF, [], "product", "unanswerable", []),
]


def kb_questions() -> list[dict]:
    """~60 seed workload questions over the KB (answerable single-fact, multi-hop across two docs, unanswerable)."""
    return [{"id": f"kbq-{i + 1:03d}", "query": q, "reference": ref, "doc_ids": list(docs), "category": cat,
             "type": typ, "key_facts": list(facts), "needs_context": True, "author": "seed"}
            for i, (q, ref, docs, cat, typ, facts) in enumerate(_Q)]
