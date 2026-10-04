"""Cheap, deterministic guards that veto a semantic-cache hit between two lookalike queries.

A bi-encoder scores "cancel my order #4821" and "I don't want to cancel my order #4822" at ~0.9 cosine,
yet the two need different answers. Each guard below compares the incoming query with the cached query
it matched and returns a rejection reason (a short string) or None. They run only on candidates that
already cleared the similarity threshold, so their cost (~tens of microseconds) is paid on near-hits only.

  numbers   digits, order/invoice IDs, amounts, dates and number words must match exactly
  negation  a word negated in one query ("don't cancel") appears un-negated in the other ("cancel"),
            or an "un-" antonym pair (subscribe / unsubscribe)
  entities  a small ShopNest lexicon (products, actions, payment methods, account tiers, shipping speeds,
            time units, timing, places, objects): if both queries name something from one group and the two
            sets are disjoint ("laptop" vs "phone", "refund" vs "exchange"), reject
  content   generic fallback: the queries are otherwise near-identical but each has a different
            uncommon content word (e.g. two city names the lexicon does not know)

These guards are an engineering heuristic of this project, not a published method; the research
notes found no benchmark for entity/number guards. Their effect is measured in eval/sweep_threshold.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from typing import Any, Mapping, Optional

# --------------------------------------------------------------------------------------------- config


@dataclass(frozen=True)
class GuardConfig:
    numbers: bool = True
    negation: bool = True
    entities: bool = True
    content: bool = True

    @classmethod
    def from_mapping(cls, m: Optional[Mapping[str, Any]]) -> "GuardConfig":
        if not m:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: bool(v) for k, v in m.items() if k in names})

    @classmethod
    def off(cls) -> "GuardConfig":
        return cls(numbers=False, negation=False, entities=False, content=False)

    @property
    def any(self) -> bool:
        return self.numbers or self.negation or self.entities or self.content


ALL_ON = GuardConfig()
ALL_OFF = GuardConfig.off()

# --------------------------------------------------------------------------------------------- text utils

_PLACEHOLDER = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
_CONTRACTIONS = [
    (r"\bcan'?t\b", "can not"), (r"\bcannot\b", "can not"), (r"\bwon'?t\b", "will not"),
    (r"\bshan'?t\b", "shall not"), (r"\bain'?t\b", "is not"), (r"n't\b", " not"), (r"n’t\b", " not"),
    (r"\b(do|does|did|is|are|was|were|has|have|had|could|should|would|must|need)nt\b", r"\1 not"),
    (r"\bdont\b", "do not"), (r"\bdoesnt\b", "does not"), (r"\bdidnt\b", "did not"), (r"\bisnt\b", "is not"),
]
_CONTRACTIONS = [(re.compile(p), r) for p, r in _CONTRACTIONS]

# placeholder name (as in Bitext's {{Order Number}}) -> how the guards treat it
_ID_PLACEHOLDERS = {"order number": "<order_number>", "invoice number": "<invoice_number>",
                    "refund amount": "<amount>", "tracking number": "<tracking_number>"}
_TIER_PLACEHOLDERS = {"account type", "account category"}
_PLACE_PLACEHOLDERS = {"delivery city", "delivery country"}

_NUMBER_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "ninety": "90", "hundred": "100", "thousand": "1000",
    "twice": "2x", "half": "0.5", "dozen": "12",
}
_NUM_RE = re.compile(r"(?<![a-z])[#$€£₹]?\d[\d,]*(?:[./:-]\d+)*(?:\.\d+)?")
_CODE_RE = re.compile(r"\b(?:[a-z]+-?\d[a-z0-9]*|\d+[a-z]+[a-z0-9]*)\b")
_WORD_RE = re.compile(r"<[a-z_]+>|[a-z][a-z'-]*[a-z]|[a-z]")


def _stem(w: str) -> str:
    """Tiny suffix stripper (Porter step-1 flavour): enough to match cancel/cancelled/cancellation."""
    if len(w) <= 3 or w.startswith("<"):
        return w
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ss"):
        pass
    else:
        for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "edly", "ers", "er", "ed", "es", "s", "ly"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]                       # shipp -> ship, cancell -> cancel (keep 'll'/'ss' words)
    if len(w) > 3 and w.endswith("ll") and not w.endswith("all"):
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]                       # receive -> receiv, matches received/receiving
    return w


def _similar(a: str, b: str) -> bool:
    """Typo-tolerant word match (Bitext is full of typos: 'oorder', 'trackng')."""
    if a == b:
        return True
    if a.startswith("<") or b.startswith("<") or min(len(a), len(b)) < 4:
        return False
    if abs(len(a) - len(b)) > 2:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.8


# --------------------------------------------------------------------------------------------- lexicon
# group -> canonical -> variants (single words or short phrases, lower case). A word may live in several groups.
LEXICON: dict[str, dict[str, list[str]]] = {
    "product": {
        "laptop": ["laptop", "notebook", "macbook", "chromebook", "ultrabook"],
        "phone": ["phone", "smartphone", "mobile phone", "cellphone", "iphone", "android phone"],
        "tablet": ["tablet", "ipad"],
        "headphones": ["headphone", "headphones", "earbuds", "earphones", "airpods", "headset"],
        "tv": ["tv", "television", "smart tv"],
        "camera": ["camera", "webcam", "dslr"],
        "smartwatch": ["smartwatch", "watch", "fitness band", "fitness tracker"],
        "charger": ["charger", "power adapter", "power bank"],
        "cable": ["cable", "usb cable", "hdmi cable"],
        "speaker": ["speaker", "soundbar", "bluetooth speaker"],
        "monitor": ["monitor"],
        "keyboard": ["keyboard"], "mouse": ["mouse"], "printer": ["printer"],
        "console": ["console", "playstation", "xbox", "nintendo"],
        "router": ["wifi router", "modem"],
        "sofa": ["sofa", "couch"], "chair": ["chair"], "table": ["table", "desk"], "lamp": ["lamp"],
        "mattress": ["mattress"], "bed": ["bed", "bed frame"], "pillow": ["pillow"], "blanket": ["blanket", "duvet"],
        "bedsheet": ["bedsheet", "bed sheet", "bedsheets"], "blender": ["blender", "mixer grinder"],
        "microwave": ["microwave", "oven"], "fridge": ["fridge", "refrigerator"], "vacuum": ["vacuum", "hoover"],
        "kettle": ["kettle"], "toaster": ["toaster"], "cookware": ["cookware", "pan", "frying pan", "pot"],
        "rug": ["rug", "carpet"], "curtain": ["curtain", "curtains"],
        "shirt": ["shirt", "t-shirt", "tshirt", "tee"], "jeans": ["jeans", "denim"],
        "trousers": ["trousers", "pants", "chinos"], "jacket": ["jacket", "blazer"], "coat": ["coat"],
        "hoodie": ["hoodie", "sweatshirt"], "sweater": ["sweater", "jumper", "cardigan"],
        "dress": ["dress", "gown"], "skirt": ["skirt"], "shoes": ["shoe", "shoes", "sneakers", "trainers", "boots"],
        "socks": ["socks"], "hat": ["hat", "cap"], "bag": ["bag", "backpack", "handbag"],
        "electronics": ["electronics", "gadget", "gadgets"], "furniture": ["furniture"],
        "apparel": ["apparel", "clothing", "clothes", "garment"], "home_goods": ["home goods", "homeware", "kitchenware"],
        "gift_card": ["gift card", "gift voucher"],
    },
    "action": {
        "cancel": ["cancel", "cancellation", "call off", "revoke"],
        "track": ["track", "tracking", "where is", "status of"],
        "return": ["return", "send back", "ship back"],
        "refund": ["refund", "money back", "reimburse", "reimbursement", "rebate", "restitution", "compensation"],
        "exchange": ["exchange", "swap", "replace", "replacement"],
        "change": ["change", "modify", "edit", "update", "switch", "alter", "correct", "amend"],
        "buy": ["buy", "place an order", "place order", "make an order", "shop"],
        "delete": ["delete", "deletion", "remove", "removal", "close", "closure", "deactivate", "terminate"],
        "register": ["register", "sign up", "signup", "create", "open an account", "enrol", "enroll"],
        "recover": ["recover", "reset", "forgot", "forgotten", "retrieve"],
        "complain": ["complain", "complaint", "claim", "grievance"],
        "review": ["review", "feedback", "rate", "rating", "opinion"],
        "subscribe": ["subscribe", "subscription"], "unsubscribe": ["unsubscribe", "opt out"],
        "pay": ["pay", "paying", "payment"], "download": ["download"], "add": ["add", "include"],
        "login": ["log in", "login", "sign in", "signin"], "logout": ["log out", "logout", "sign out"],
    },
    "payment": {
        "card": ["card", "credit card", "debit card", "visa", "mastercard", "amex"],
        "paypal": ["paypal"], "upi": ["upi", "gpay", "phonepe", "paytm"], "cod": ["cash on delivery", "cod", "cash"],
        "gift_card": ["gift card", "voucher"], "bank_transfer": ["bank transfer", "wire transfer", "net banking", "netbanking"],
        "apple_pay": ["apple pay"], "google_pay": ["google pay"], "emi": ["emi", "installment", "instalment", "klarna"],
        "wallet": ["wallet", "store credit"], "crypto": ["crypto", "bitcoin"],
    },
    "tier": {t: [t] for t in ("free", "freemium", "basic", "standard", "premium", "pro", "gold", "platinum",
                              "silver", "business", "personal", "family", "student", "enterprise", "plus", "vip")},
    "shipping": {
        "standard": ["standard shipping", "standard delivery", "regular shipping"],
        "express": ["express", "expedited", "priority"], "overnight": ["overnight", "next day", "next-day"],
        "same_day": ["same day", "same-day"], "international": ["international", "overseas", "abroad"],
        "domestic": ["domestic", "local delivery"], "pickup": ["pickup", "pick up", "click and collect"],
    },
    "time": {
        "today": ["today", "tonight"], "tomorrow": ["tomorrow"], "yesterday": ["yesterday"],
        "weekend": ["weekend", "saturday", "sunday"], "weekday": ["weekday", "weekdays", "monday", "tuesday", "wednesday", "thursday", "friday"],
        "hour": ["hour", "hours"], "day": ["day", "days"], "week": ["week", "weeks"], "month": ["month", "months"],
        "year": ["year", "years"], "minute": ["minute", "minutes"],
        **{m: [m] for m in ("january", "february", "march", "april", "june", "july", "august", "september",
                            "october", "november", "december")},
    },
    "timing": {
        "before": ["before", "prior to", "ahead of"], "after": ["after", "once it has"],
        "early": ["early", "earlier"], "late": ["late", "delayed", "overdue"],
    },
    "place": {
        **{c: [c] for c in ("india", "usa", "america", "canada", "mexico", "brazil", "uk", "england", "ireland", "france",
                            "germany", "spain", "italy", "netherlands", "australia", "japan", "china", "singapore",
                            "dubai", "uae", "pune", "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "hyderabad",
                            "kolkata", "london", "paris", "berlin", "toronto", "sydney", "tokyo")},
        "us": ["united states", "the us", "the states"],
    },
    "object": {
        "order": ["order", "purchase", "package", "parcel", "shipment", "item"],
        "address": ["address", "shipping address", "delivery address", "billing address"],
        "account": ["account", "profile"],
        "password": ["password", "passcode", "pin", "access key", "credentials"],
        "invoice": ["invoice", "bill", "receipt"],
        "newsletter": ["newsletter", "mailing list"],
        "warranty": ["warranty", "guarantee"],
        "email": ["email", "e-mail", "email address"],
        "phone_number": ["phone number", "mobile number"],
    },
}


def _build_lexicon_index():
    single: dict[str, list[tuple[str, str]]] = {}
    phrases: list[tuple[re.Pattern, str, str]] = []
    for group, canon_map in LEXICON.items():
        for canon, variants in canon_map.items():
            for v in variants:
                if " " in v or "-" in v:
                    phrases.append((re.compile(r"\b" + re.escape(v) + r"\b"), group, canon))
                else:
                    single.setdefault(_stem(v), []).append((group, canon))
    return single, phrases


_LEX_SINGLE, _LEX_PHRASES = _build_lexicon_index()

# phrases that look like negation but are hedges/problem statements, not polarity
_HEDGES = [re.compile(p) for p in (
    r"\b(?:i|we)?\s*(?:do|did|does) not (?:know|understand|remember|see|get it|have a clue)\b",
    r"\bnot sure\b", r"\bno idea\b", r"\bno clue\b", r"\bnot certain\b",
    r"\b(?:can|could) (?:not|no longer)\b",   # "I can't log in" asks the same thing as "help me log in"
    r"\bunable to\b", r"\bnot able to\b",
    r"\bno one\b", r"\bnot only\b", r"\bnot yet sure\b",
)]
_VOLITION = {"want", "wanna", "wish", "need", "needed", "like", "plan", "intend", "mean", "try", "trying", "going",
             "have", "has", "had"}
_NEG_CUES = {"not", "no", "never", "without", "nothing", "none", "neither", "nor", "nobody", "unable"}
# words skipped when looking for the head of a negation's scope ("don't want to cancel" -> cancel)
_SCOPE_SKIP = {
    "to", "be", "been", "being", "able", "i", "me",
    "my", "the", "a", "an", "any", "it", "its", "this", "that", "these", "those", "really", "even", "yet", "still",
    "longer", "more", "ever", "do", "does", "did", "will", "would", "could", "should", "can", "shall", "may", "might",
    "is", "are", "was", "were", "am", "you", "your", "we", "our", "us", "they", "them", "please", "just", "so",
    "like", "going", "gonna", "try", "trying", "anymore", "at", "all", "for", "of", "with", "in", "on", "again",
}

# stopwords + support-domain generic words: never "rare" for the content guard
_COMMON_WORDS = set("""
a about above after again against all almost also am an and any anyone anything are around as ask asked asking at
available away back be because been before being below between both but by can could did do does doing done down
during each either else even ever every few for from further get gets getting give go goes going gone got gotten had
has have having he her here hers him his how however i if in into is it its itself just keep know known last least
let like likely make makes many may maybe me might mine more most much must my myself need needs neither never new
next no nor not now of off often on once one only or other our ours out over own please possible quite rather
really right said same say see seem should since so some someone something sometimes still such sure take tell than
thank thanks that the their them then there these they thing things this those though through till to too try
trying under until up upon us use used using very via want wanted wanna was way we well were what whatever when
where whether which while who whom whose why will with within without would yes yet you your yours yourself hi hello
hey ok okay pls plz u ur ya im ive id dont cant wont didnt doesnt isnt hey gotta gonna lemme kindly
help helping assistance assist assistant support customer customers service services agent agents person someone
human representative operator staff team company store shop shopnest website site app online portal page link
order orders ordered ordering purchase purchased item items product products article articles goods stuff thing
account accounts profile user username password email mail message inform informing information info details detail
question questions issue issues problem problems trouble error wrong mistake fix solve report reporting reported
status update updates news current check checking see view look looking find finding locate show tell know learn
cancel cancelled canceling cancelling cancellation track tracking change changing modify edit switch update correct
return returns returning refund refunds refunded money back rebate rebates reimbursement reimburse restitution
compensation exchange replace replacement delivery deliveries deliver delivered shipping ship shipped shipment
address addresses package parcel arrive arrival arrived when time long soon fast quick days day week weeks hour
hours payment payments pay paid paying method methods option options card cards invoice invoices bill bills
receipt receipts fee fees charge charges charged cost costs price prices penalty penalties policy policies terms
create creating open opening register registration signup sign up subscribe subscription newsletter delete remove
removing close closing recover recovery reset forgot forgotten lost access key log login logged
complaint complain claim review reviews feedback opinion comment rate experience file filing lodge make submit
send sending sent receive received receiving get request requesting demand demanding obtain call talk talking
speak speaking contact contacting reach live chat phone number hours hold wait waiting expect expecting
early exit withdrawal termination standard set setting setup new another different second other secondary main
primary shipping delivery period estimated estimate eta date dates available accept accepted list allowed
corporate company's my mine own private personal business free paid amount dollars dollar euro euros rupees
bloody damn damned fucking fuckin fucked goddamn goddamned freaking frigging effing shitty crap crappy hell
able across actually add added ago agree allow already always among answer anybody anyhow anymore anyway apply
area aside away bad bank basic become began begin behind believe best better big bit body book bought bring
brought build busy buy call called came care case cases cause certain chance child city clear clearly close
come comes coming common complete consider continue country couple course create cut deal decide decided
describe did die different difference difficult direct doing door double during easy easily else end enough
entire especially even evening event every everybody everyone everything exactly example except explain face
fact fail fair family far feel feeling felt fill final finally fine first follow following form found friend
full fully further gave general given giving glad good great group guess hand happen happened happens happy
hard head hear heard hello high hold home hope house idea important include including instead interest
interested job just keep kind knew large late later lead learn leave left less life light line little live
long longer lot lots love low made main major manage matter mean means meant meet mind minute minutes miss
moment money month months morning move name near nearly necessary never next nice night none normal note
nothing notice number numbers offer office old open order other others otherwise part particular past pay
people per perhaps person phone pick place plan play point possible power present pretty probably process
proper provide public put question quick quickly reach read ready real realize reason reasons recent
recently remember rest result results room run saw school second seen sell send sense set several share short
show side simple simply since single situation situations small sorry sort sound speak special spend
start started state stay step stop story strange student study sure system take taken talk tell thing
think thought three today together told took top total toward true turn type types understand unless usual
usually value wait walk wanna watch way ways week went whole wish word words work worked working world worry
worse worst write wrong year years young removal deletion closure choose chose chosen enter entered entering
select selected selecting pick picked type typed provide provided give gave acquire acquired earn get got
receive mention mentioned put placed added missing wrong incorrect valid invalid proper
""".split())


@lru_cache(maxsize=65536)
def _is_common(w: str) -> bool:
    if w in _COMMON_WORDS or w.startswith("<") or len(w) <= 4:   # short tokens: mostly typos ("foir", "hw")
        return True
    st = _stem(w)
    if st in _COMMON_STEMS or st in _LEX_SINGLE:
        return True
    if len(w) >= 5:                      # typo of a common word ("asistance", "invoce", "uhelp")
        if get_close_matches(w, _COMMON_BY_LEN.get(len(w), ()), n=1, cutoff=0.8):
            return True
        for k in range(2, len(w) - 2):  # missing space: "whichsituations", "freeaccount"
            l, r = w[:k], w[k:]
            if (l in _COMMON_WORDS or l in _STOP) and (r in _COMMON_WORDS or _stem(r) in _COMMON_STEMS):
                return True
    return False


_STOP = set("""a about am an and any are as at be been but by can could did do does for from had has have how i if in
into is it its me my no not of on or our please should so than that the their them then there these they this those
to too u ur us was we were what when where which who why will with would you your ya im""".split())
_COMMON_STEMS = {_stem(w) for w in _COMMON_WORDS}
_COMMON_BY_LEN: dict[int, list[str]] = {}
for _w in _COMMON_WORDS:
    if len(_w) >= 4:
        for _n in range(len(_w) - 2, len(_w) + 3):
            _COMMON_BY_LEN.setdefault(_n, []).append(_w)


@lru_cache(maxsize=65536)
def _lex_lookup(w: str) -> tuple:
    """Lexicon entries for a word; typo-tolerant for longer words ('subscriptikon', 'cancellaton')."""
    st = _stem(w)
    hit = _LEX_SINGLE.get(st)
    if hit:
        return tuple(hit)
    if len(st) >= 6 and w not in _COMMON_WORDS:
        close = get_close_matches(st, _LEX_LONG_STEMS.get(len(st), ()), n=1, cutoff=0.85)
        if close:
            return tuple(_LEX_SINGLE[close[0]])
        for k in range(len(st) - 1, 4, -1):
            hit = _LEX_SINGLE.get(st[:k])
            if hit:
                return tuple(hit)
    return ()


_LEX_LONG_STEMS: dict[int, list[str]] = {}
for _s in _LEX_SINGLE:
    if len(_s) >= 5:
        for _n in range(len(_s) - 2, len(_s) + 3):
            _LEX_LONG_STEMS.setdefault(_n, []).append(_s)


def _context_rules(entities: dict[str, set]) -> None:
    """Domain equivalences that depend on the object: 'cancel my account' == 'close my account', etc."""
    acts, objs = entities.get("action"), entities.get("object", set())
    if not acts:
        return
    if "account" in objs and acts & {"cancel", "delete"}:
        acts |= {"cancel", "delete"}                     # cancel / close / delete an account
    if "order" in objs and acts & {"delete", "exchange", "add"}:
        acts.add("change")                               # remove / swap / add an item of an order = change it
    if "newsletter" in objs or "subscribe" in acts:
        if "register" in acts:
            acts.add("subscribe")                        # sign up to the newsletter
        if acts & {"cancel", "delete"}:
            acts.add("unsubscribe")                      # cancel my subscription / remove me from the list
    if "unsubscribe" in acts:
        acts.discard("subscribe")                        # 'unsubscribe' is not also 'subscribe'


# --------------------------------------------------------------------------------------------- analysis


@dataclass(frozen=True)
class _Analysis:
    numbers: frozenset            # normalised numeric values + ID placeholders
    entities: dict                # group -> frozenset(canonicals)
    negated: frozenset            # stems in a negation's scope
    plain: frozenset              # stems seen outside any negation scope
    un_words: frozenset           # stems of "un-" words (unsubscribe)
    content: frozenset            # stems of uncommon words (candidates for the content guard)
    bag: frozenset                # stems of all non-stopwords (for the lookalike test)


def _normalise(text: str) -> str:
    t = text.lower().replace("’", "'")
    for pat, rep in _CONTRACTIONS:
        t = pat.sub(rep, t)
    return t


@lru_cache(maxsize=65536)
def analyse(text: str) -> _Analysis:
    raw = _normalise(text)
    numbers: set[str] = set()
    entities: dict[str, set[str]] = {}

    # placeholders (Bitext style {{Order Number}}) -> typed slots
    def _ph(m: re.Match) -> str:
        name = m.group(1).strip().lower()
        if name in _ID_PLACEHOLDERS:
            numbers.add(_ID_PLACEHOLDERS[name])
            return " "
        if name in _TIER_PLACEHOLDERS:
            entities.setdefault("tier", set()).add("<tier>")
            return " "
        if name in _PLACE_PLACEHOLDERS:
            entities.setdefault("place", set()).add("<place>")
            return " "
        return " <" + re.sub(r"\W+", "_", name) + "> "
    t = _PLACEHOLDER.sub(_ph, raw)

    for m in _CODE_RE.finditer(t):        # model numbers / SKUs / order codes: ps5, xps13, sn-48213, 5pm
        code = m.group(0).replace("-", "")
        digits = re.sub(r"\D", "", code)
        numbers.add(digits if len(digits) >= 4 else code)   # long IDs compare by digits, short codes as a whole
    t = _CODE_RE.sub(" ", t)
    for m in _NUM_RE.finditer(t):
        v = m.group(0).lstrip("#$€£₹").replace(",", "")
        if v:
            numbers.add(v.rstrip("."))
    t_nonum = _NUM_RE.sub(" ", t)

    # lexicon phrases first (so "credit card" is not also read as bare "card"), then single words
    t_lex = t_nonum
    for pat, group, canon in _LEX_PHRASES:
        if pat.search(t_lex):
            entities.setdefault(group, set()).add(canon)
            t_lex = pat.sub(" ", t_lex)
    words = _WORD_RE.findall(t_nonum)
    for w in _WORD_RE.findall(t_lex):
        if w in _NUMBER_WORDS:
            numbers.add(_NUMBER_WORDS[w])
            continue
        for group, canon in _lex_lookup(w):
            entities.setdefault(group, set()).add(canon)
    _context_rules(entities)

    # negation scope (after removing hedges like "I don't know how to")
    t_neg = t_nonum
    for h in _HEDGES:
        t_neg = h.sub(" ", t_neg)
    toks = _WORD_RE.findall(t_neg)
    negated, plain = set(), set()
    i = 0
    while i < len(toks):
        w = toks[i]
        if w in _NEG_CUES:
            j = i + 1
            while j < len(toks):
                if toks[j] in _SCOPE_SKIP:
                    j += 1
                elif toks[j] in _VOLITION and j + 1 < len(toks) and toks[j + 1] == "to":
                    j += 2                       # "don't want to cancel" -> the negated thing is "cancel"
                else:
                    break                        # "don't want the gold account" -> the negated thing is "want"
            if j < len(toks):
                negated.add(_stem(toks[j]))
                i = j + 1
                continue
        elif w not in _SCOPE_SKIP and not (w in _VOLITION and i + 1 < len(toks) and toks[i + 1] == "to"):
            plain.add(_stem(w))
        i += 1
    un_words = frozenset(_stem(w[2:]) for w in words if w.startswith("un") and len(w) >= 7)
    content = frozenset(_stem(w) for w in words if w not in _COMMON_WORDS and len(w) > 2 and not w.startswith("<")
                        and w not in _NEG_CUES and w not in _NUMBER_WORDS)
    bag = frozenset(_stem(w) for w in words if w not in _STOP and not w.startswith("<"))
    return _Analysis(frozenset(numbers), {g: frozenset(c) for g, c in entities.items()}, frozenset(negated),
                     frozenset(plain), un_words, content, bag)


# --------------------------------------------------------------------------------------------- the guards


def numbers_guard(a: _Analysis, b: _Analysis) -> Optional[str]:
    if a.numbers != b.numbers:
        return "number_mismatch"
    return None


def _contains(stems: frozenset, w: str) -> bool:
    return w in stems or any(_similar(w, s) for s in stems)


def negation_guard(a: _Analysis, b: _Analysis) -> Optional[str]:
    # a word negated on one side and plainly present on the other: "don't cancel" vs "cancel"
    for x, y in ((a, b), (b, a)):
        for w in x.negated:
            if not _contains(y.negated | y.un_words, w) and _contains(y.plain, w):
                return "negation_mismatch"
    # morphological antonyms: unsubscribe vs subscribe, unopened vs opened
    for x, y in ((a, b), (b, a)):
        if y.entities.get("action", frozenset()) & _UNDOING:
            continue                                     # "cancel my subscription" already negates it
        for w in x.un_words:
            if _contains(y.plain, w) and w not in x.plain and not _contains(y.un_words | y.negated, w):
                return "negation_mismatch"
    return None


_UNDOING = frozenset({"cancel", "delete", "unsubscribe"})


def entity_guard(a: _Analysis, b: _Analysis) -> Optional[str]:
    for group in LEXICON:
        ea, eb = a.entities.get(group), b.entities.get(group)
        if ea and eb and ea.isdisjoint(eb):
            return f"entity_mismatch:{group}"
    return None


def content_guard(a: _Analysis, b: _Analysis) -> Optional[str]:
    """Lookalike with a swapped uncommon word: most content words match, but each side has one the other lacks."""
    da = {w for w in a.content if not _contains(b.bag, w)}
    db = {w for w in b.content if not _contains(a.bag, w)}
    if not da or not db or len(da) > 2 or len(db) > 2:
        return None
    if not any(not _is_common(w) for w in da) or not any(not _is_common(w) for w in db):
        return None
    # lookalike test: apart from the swapped words, the two queries say the same thing
    shared = sum(1 for w in a.bag if _contains(b.bag, w))
    if shared and shared / max(len(a.bag), len(b.bag)) >= 0.5:
        return "content_mismatch"
    return None


_ORDER = (("numbers", numbers_guard), ("negation", negation_guard), ("entities", entity_guard),
          ("content", content_guard))


def check(query: str, cached_query: str, cfg: GuardConfig = ALL_ON) -> Optional[str]:
    """Return the first rejection reason, or None if the cached answer may be served."""
    if not cfg.any:
        return None
    a, b = analyse(query), analyse(cached_query)
    for name, fn in _ORDER:
        if getattr(cfg, name):
            r = fn(a, b)
            if r:
                return r
    return None


def check_all(query: str, cached_query: str) -> dict[str, Optional[str]]:
    """Every guard's verdict (for diagnostics and the per-guard breakdown in the sweep)."""
    a, b = analyse(query), analyse(cached_query)
    return {name: fn(a, b) for name, fn in _ORDER}
