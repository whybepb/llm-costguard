"""Hardness signals for the downshift router.

A request is "hard" when any cheap, explainable signal says a small model is likely to fail it: reasoning or
arithmetic, code, several questions at once, an angry or legal escalation, a non-English message, or simply a lot of
input (long message, deep history, many context documents). Hard requests always go to the strong tier, in every
router policy. The signals are deliberately simple (regex + token counts, ~0.1 ms): independent router benchmarks
find simple routers competitive, and the eval gate - not the router - is what carries the quality guarantee.

`extract()` returns `Hardness(hard, reasons, signals)`; `reasons` are ordered by priority, so `reasons[0]` is the
label the router logs (e.g. `hard:reasoning-keywords`).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

from ..tokens import count_text

# Priority order of reasons (first match becomes the logged route reason).
REASONS = ("escalation", "non-english", "code", "arithmetic", "reasoning-keywords", "multi-question",
           "long-query", "long-input", "deep-history", "many-context-docs")


@dataclass
class HardnessConfig:
    long_query_tokens: int = 120        # the customer's own message (not context) longer than this -> hard
    long_input_tokens: int = 4000       # whole prompt (system + history + all retrieved context) before trimming
    max_history_turns: int = 4          # prior messages; more than 2 full exchanges -> hard
    max_context_docs: int = 8           # retrieved documents to reconcile
    min_questions: int = 2              # distinct questions in one message
    shouting_min_letters: int = 12      # ALL-CAPS check only applies to messages with at least this many letters

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "HardnessConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class Hardness:
    hard: bool
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> Optional[str]:
        return self.reasons[0] if self.reasons else None

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ patterns
_I = re.IGNORECASE

REASONING = re.compile(
    r"\bwhy\b|\bhow come\b|\bcompar(e|ed|ing|ison)\b|\bdifference between\b|\bexplain the difference\b"
    r"|\bwhat'?s the difference\b|\bversus\b|\bvs\.?(?=\s)|\bwhich (one )?(is|would be) (better|cheaper|best)\b"
    r"|\bpros and cons\b|\btroubleshoot\w*\b|\bcalculat\w*\b|\bhow much would\b|\bstep[- ]by[- ]step\b"
    r"|\bwhat (would|will) happen if\b|\bwhat happens (to|if|when)\b|\bwhat if\b|\bwork out\b"
    r"|\b(is it|would it be|is that) better\b|\bbetter (to|option)\b[^?]*\bor\b|\bshould i\b[^?]*\bor\b"
    r"|\bif i\b[^.?]*,\s*(what|how|will|would|do|does|can|is)\b"
    r"|\b(isn'?t|is not|doesn'?t|does not|won'?t|will not|stopped|not) (work\w*|turn\w* on|charg\w*|connect\w*|pair\w*|boot\w*)\b",
    _I)

ESCALATION = re.compile(
    r"\bangry\b|\bfurious\b|\boutraged?\b|\blivid\b|\bpissed\b|\bfed up\b|\bunacceptable\b|\bridiculous\b"
    r"|\bdisgust\w*\b|\bworst\b|\bterrible (service|experience)\b|\bhorrible\b|\bpathetic\b|\bscam\w*\b|\bfraud\w*\b"
    r"|\bcharge ?backs?\b|\bdispute\b|\blawyer\b|\battorney\b|\blegal\b|\bsue\b|\bsuing\b|\bcourt\b|\bpolice\b"
    r"|\bconsumer (forum|court|protection)\b|\bbetter business bureau\b|\bbbb\b|\breport (you|this)\b"
    r"|\bstolen\b|\bunauthori[sz]ed\b|\bidentity theft\b|\bhacked\b|\bcomplain\w*\b|\bescalat\w*\b|\bmanager\b"
    r"|\bdamn\w*\b|\bgoddamn\w*\b|\bbloody\b|\bf+u+c+k\w*\b|\bshit\w*\b|\bcrap\w*\b|\bwtf\b|\bf\*+\w*",
    _I)

CODE = re.compile(
    r"```|Traceback \(most recent call last\)|^\s*(def|class|import|from\s+\w+\s+import|function|const|let|var|public"
    r"|private|SELECT|INSERT|UPDATE|curl)\b.*|=>|\{\s*\"\w+\"\s*:|</?[a-zA-Z][a-zA-Z0-9]*(\s[^<>]*)?>"
    r"|\b\w+\([^()]*\)\s*[;{]|;\s*$|\b(HTTP|API) (error|status) \d{3}\b|\berror code\s*[:#]?\s*\w*\d"
    r"|\b[A-Z][a-zA-Z]*(Error|Exception)\b|\bundefined is not\b|\bnull pointer\b|\bstack ?trace\b",
    re.MULTILINE)

ARITH_EXPR = re.compile(r"\d(\.\d+)?\s*[+*/×÷]\s*\d|\d\s+[-x]\s+\d|\d\s*\^\s*\d")
MONEY = re.compile(r"[$₹€£]\s?\d[\d,]*(\.\d+)?|\b\d[\d,]*(\.\d+)?\s?(dollars|usd|rupees|rs\.?|inr|eur|euros?|pounds)\b", _I)
PERCENT = re.compile(r"\d+(\.\d+)?\s?%|\b\d+(\.\d+)?\s?percent\b", _I)
NUMBER = re.compile(r"(?<![#\w])\d+(\.\d+)?(?![\w])")
# a price/amount question about a stated quantity ("7 kg to Canada, what will it cost?") needs a calculation
COST_QUESTION = re.compile(r"\bhow much (will|would|do|does|did|is|are|should|can)\b|\bwhat will [^?]*\bcost\b"
                           r"|\bwhat (is|'s) the (total|final|exact) (cost|price|amount)\b|\bhow much\b[^?]*\bget back\b", re.I)
QUANTITY = re.compile(r"[$₹€£]\s?\d|\b\d+(\.\d+)?\s?(kg|g|km|items?|units?|pieces|pcs|months?|years?|days?|%)\b"
                      r"|\b\d[\d,]{2,}\b", re.I)
ARITH_WORDS = re.compile(r"\b(total|sum|add up|altogether|in total|split|per (item|unit|month)|each costs?|average)\b", _I)

# two asks fused into one question: "How much will I get back, and when?"
FUSED_ASK = re.compile(r"(,|\band\b)\s*(and\s+)?(when|how (much|long|many|soon)|what|where|which|why)\b[^,]*\?\s*$", re.I)
MULTI_Q_MARKERS = re.compile(
    r"\b(also|additionally|another question|second question|secondly|one more (thing|question)|a few questions"
    r"|two questions|couple of questions|and then)\b", _I)
ENUMERATION = re.compile(r"^\s*(\d+[.)]|[-*•])\s+\S", re.MULTILINE)

# Function words that are common in one language and rare in English customer-support text. Ambiguous short words
# that are also English ("a", "no", "die", "me", "can", "do", "to", "is", "per", "come") are deliberately left out.
FOREIGN_WORDS = set("""
el los las que mi mis por para como cómo dónde donde cuándo cuando pedido quiero tengo necesito ayuda una con del
está estoy puedo hola gracias reembolso devolución envío cuenta
le les des du je mon ma mes est pas une pour avec comment où quand commande remboursement vous nous qui ne ai suis
bonjour merci livraison retour
der das ich nicht mein meine ist und wie wo wann bestellung ein eine mit für haben bitte kann rückerstattung
lieferung danke hallo
meu minha não onde você obrigado pedido entrega
il mio non dove sono ordine grazie ciao spedizione rimborso
mera meri mere kab kya kyun kyon kaise hai hain nahi nahin aaya aayega aayegi paisa paise karo mujhe bhai abhi
gaya gayi wapas kaha kahan kitna kitne chahiye milega mila hoga raha rahi
""".split())
ENGLISH_WORDS = set("""
the my i to and of it for in you how what where can do not with have this was order me when is a an are be on
your please help want need get did does will would could why which there been am at from by or if
""".split())
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _questions(text: str) -> int:
    # count question marks that close a clause with at least two words before them ("??" counts once)
    parts = re.split(r"\?+", text)
    return sum(1 for p in parts[:-1] if len(_WORD.findall(p)) >= 2)


def _non_english(text: str) -> tuple[bool, dict]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False, {}
    non_latin = sum(1 for c in letters if not unicodedata.name(c, "").startswith("LATIN"))
    if non_latin / len(letters) > 0.2:
        return True, {"non_latin_share": round(non_latin / len(letters), 2)}
    words = [w.lower() for w in _WORD.findall(text)]
    foreign = sum(1 for w in words if w in FOREIGN_WORDS)
    english = sum(1 for w in words if w in ENGLISH_WORDS)
    accented = sum(1 for c in letters if unicodedata.decomposition(c).startswith("00") or c in "ßæøå")
    if (foreign >= 2 and foreign > english) or (foreign >= 1 and accented >= 1 and foreign >= english):
        return True, {"foreign_words": foreign, "english_words": english}
    return False, {}


def _shouting(text: str, min_letters: int) -> bool:
    letters = [c for c in text if c.isalpha() and c.isascii()]
    if len(letters) < min_letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) > 0.6


def extract(query: str, *, input_tokens: int = 0, has_context: bool = False, context_docs: Optional[int] = None,
            history_turns: int = 0, cfg: Optional[HardnessConfig] = None) -> Hardness:
    """Compute hardness signals for one request. Pure, deterministic, no I/O."""
    cfg = cfg or HardnessConfig()
    q = query or ""
    found: dict[str, Any] = {}

    esc = [m.group(0).lower() for m in ESCALATION.finditer(q)]
    if esc or "!!" in q or _shouting(q, cfg.shouting_min_letters):
        found["escalation"] = esc[:5] or (["!!"] if "!!" in q else ["ALL-CAPS"])

    ne, info = _non_english(q)
    if ne:
        found["non-english"] = info

    code = [m.group(0).strip()[:30] for m in CODE.finditer(q)]
    if code:
        found["code"] = code[:3]

    money, pct, nums = MONEY.findall(q), PERCENT.findall(q), NUMBER.findall(q)
    if (ARITH_EXPR.search(q) or len(money) >= 2 or (pct and len(nums) >= 2) or (ARITH_WORDS.search(q) and len(nums) >= 2)
            or (COST_QUESTION.search(q) and QUANTITY.search(q))):
        found["arithmetic"] = {"money": len(money), "percent": len(pct), "numbers": len(nums)}

    rk = sorted({m.group(0).lower() for m in REASONING.finditer(q)})
    if rk:
        found["reasoning-keywords"] = rk[:5]

    nq = _questions(q)
    enum = len(ENUMERATION.findall(q))
    sentences = len([x for x in re.split(r"[.?!]+\s+|\n+", q.strip()) if len(_WORD.findall(x)) >= 2])
    fused = bool(FUSED_ASK.search(q)) and re.search(r"\b(what|when|how|where|which|why|is|are|can|do|does|will)\b",
                                                   re.split(r",|\band\b", q, maxsplit=1)[0], re.I) is not None
    if nq >= cfg.min_questions or enum >= 2 or fused or (MULTI_Q_MARKERS.search(q) and sentences >= 2):
        found["multi-question"] = {"questions": nq, "enumerated": enum, "fused": fused}

    qt = count_text(q)
    if qt > cfg.long_query_tokens:
        found["long-query"] = qt
    if input_tokens > cfg.long_input_tokens:
        found["long-input"] = input_tokens
    if history_turns > cfg.max_history_turns:
        found["deep-history"] = history_turns
    if context_docs is not None and context_docs > cfg.max_context_docs:
        found["many-context-docs"] = context_docs

    reasons = [r for r in REASONS if r in found]
    signals = {"query_tokens": qt, "input_tokens": input_tokens, "has_context": has_context,
               "context_docs": context_docs, "history_turns": history_turns, **found}
    return Hardness(hard=bool(reasons), reasons=reasons, signals=signals)


def from_route_input(inp, cfg: Optional[HardnessConfig] = None) -> Hardness:
    """Hardness for a `RouteInput`. `context_docs` is read if the schema ever carries it (see docs)."""
    return extract(inp.query, input_tokens=inp.input_tokens, has_context=inp.has_context,
                   context_docs=getattr(inp, "context_docs", None), history_turns=inp.history_turns, cfg=cfg)
