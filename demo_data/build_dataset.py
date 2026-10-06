#!/usr/bin/env python3
"""Build the Northstar customer-support demo set.

Writes demo_data/cases.json, queries.jsonl, and manifest.json.
Replay queries.jsonl in order. An exact_hit row is a byte-identical
repeat of the earlier row named by exact_of.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"

REMINDER = (
    "These figures belong to other Northstar articles. They are repeated so a long prompt "
    "is not trimmed into the wrong clock. The return window is 30 calendar days from delivery. "
    "Card refunds generally post 5 to 10 business days after issuance. An issued refund that "
    "does not change for 10 business days should go to support. A first carrier scan can take "
    "24 hours. A missed delivery estimate can be given 48 hours before a late-parcel "
    "investigation. Seller silence uses 4 calendar days when the order states no other window. "
    "Price match uses 7 days. Review edits use 14 days. The narrow monthly membership fee "
    "exception uses 48 hours and only with unused benefits. A quoted figure from this reminder "
    "is valid only for the article that owns it. Repeating a neighboring number in the answer "
    "is a mistake when the question is about a different article. This reminder is boilerplate. "
    "It does not create a new exception, a new balance, a new tracking event, or a new approval."
)


def parse_markdown(path: Path) -> dict:
    raw = path.read_text()
    if not raw.startswith("---"):
        raise SystemExit(f"missing frontmatter: {path.name}")
    _, fm, body = raw.split("---", 2)
    meta = {}
    for line in fm.strip().splitlines():
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    sections = {}
    current = None
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        elif current is not None and not line.startswith("# "):
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    sections["repeated_reminders"] = REMINDER
    meta["sections"] = sections
    meta["path"] = str(path.relative_to(ROOT))
    return meta


def load_corpus() -> dict[str, dict]:
    docs = {}
    for path in sorted(CORPUS.glob("pol_*.md")):
        doc = parse_markdown(path)
        docs[doc["id"]] = doc
    return docs


def words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def approx_tokens(text: str) -> int:
    return int(round(words(text) * 1.3))


def tokens_of(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    left, right = tokens_of(a), tokens_of(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def full_docs(doc_ids: list[str]) -> list[dict]:
    return [{"doc_id": doc_id} for doc_id in doc_ids]


def routing(difficulty: str, context_size: str) -> dict:
    small_ok = difficulty == "easy"
    if difficulty == "easy" and context_size == "large":
        reason = (
            "One explicit rule. A smaller model can answer after irrelevant chunks are removed. "
            "The raw bundle is large enough that window management and compression should run first."
        )
    elif difficulty == "easy" and context_size == "small":
        reason = (
            "One explicit rule in a short context. Skip window management and compression. "
            "A smaller model can answer."
        )
    elif difficulty == "complex" and context_size == "large":
        reason = (
            "The answer depends on an exception, a case fact, or a refuse-to-claim rule. "
            "Use a larger model. The bundle is large, so window management and compression should run first."
        )
    else:
        reason = (
            "The answer depends on an exception or a refuse-to-claim rule. Context is already short, "
            "so skip window management and compression, and use a larger model."
        )
    return {
        "suggested_model": "small" if small_ok else "large",
        "small_model_ok": small_ok,
        "depends_on_pruning": context_size == "large",
        "reason": reason,
    }


def expand_cluster(cluster: dict) -> list[dict]:
    built: list[dict] = []
    large = full_docs(cluster["large_docs"])
    small = [cluster["small"]]
    for spec in cluster["rows"]:
        if spec["variant"] == "exact":
            src = built[spec["copy"]]
            row = deepcopy(src)
            row["variant"] = "exact"
            row["expected_cache"] = "exact_hit"
            row["exact_of_local"] = spec["copy"]
            row["confusable_local"] = None
            row["trap"] = None
            row["wrong_if_reused"] = None
            built.append(row)
            continue

        context_size = spec["context"]
        row = {
            "cluster_id": cluster["id"],
            "variant": spec["variant"],
            "context_size": context_size,
            "difficulty": spec["difficulty"],
            "query": spec["query"],
            "case_id": spec.get("case_id"),
            "answer_id": spec["answer_id"],
            "attachments": spec.get("attachments") or (large if context_size == "large" else small),
            "gold_doc_ids": spec["gold_doc_ids"],
            "gold_facts": spec["gold_facts"],
            "must_not_claim": spec["must_not_claim"],
            "reference_answer": spec["reference_answer"],
            "trap": spec.get("trap"),
            "wrong_if_reused": spec.get("wrong_if_reused"),
            "confusable_local": spec.get("confusable"),
            "confusable_ref": spec.get("confusable_ref"),
            "exact_of_local": None,
            "preserve_case": spec.get("case_id") is not None,
        }
        if spec["variant"] == "semantic":
            src = next(item for item in built if item["answer_id"] == spec["answer_id"])
            row["reference_answer"] = src["reference_answer"]
            row["gold_doc_ids"] = list(src["gold_doc_ids"])
            row["gold_facts"] = list(src["gold_facts"])
            row["must_not_claim"] = list(src["must_not_claim"])
            row["case_id"] = src["case_id"]
            row["difficulty"] = src["difficulty"]
            row["attachments"] = deepcopy(src["attachments"])
            row["context_size"] = src["context_size"]
            row["preserve_case"] = src["preserve_case"]
            row["expected_cache"] = "semantic_hit"
            row["semantic_of_local"] = built.index(src)
        elif spec["variant"] == "cold":
            row["expected_cache"] = "cold"
            row["semantic_of_local"] = None
        else:
            row["expected_cache"] = "miss"
            row["semantic_of_local"] = None
        row.update(routing(row["difficulty"], row["context_size"]))
        built.append(row)
    for index, row in enumerate(built):
        row["local_index"] = index
        if row["variant"] == "exact":
            row.update(routing(row["difficulty"], row["context_size"]))
    return built


def render_case(case: dict) -> str:
    lines = [f"Order {case['order_id']} for {case.get('customer', 'the signed-in customer')}."]
    for key, value in case.items():
        if key in {"id", "order_id", "customer"}:
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def render_attachment(docs: dict, attachment: dict) -> str:
    doc = docs[attachment["doc_id"]]
    wanted = attachment.get("sections")
    parts = [f"## {doc['title']} ({doc['id']})"]
    for name, text in doc["sections"].items():
        if wanted and name not in wanted:
            continue
        parts.append(f"### {name}\n{text}")
    return "\n\n".join(parts)


def assemble(docs: dict, cases: dict, row: dict) -> str:
    blocks = []
    if row.get("case_id"):
        blocks.append("## Case\n" + render_case(cases[row["case_id"]]))
    for attachment in row["attachments"]:
        blocks.append(render_attachment(docs, attachment))
    return "\n\n".join(blocks)


def row_template(
    variant,
    query,
    answer_id,
    gold_doc_ids,
    gold_facts,
    must_not_claim,
    reference_answer,
    difficulty,
    context,
    case_id=None,
    trap=None,
    confusable=None,
    confusable_ref=None,
    wrong_if_reused=None,
    attachments=None,
):
    return {
        "variant": variant,
        "query": query,
        "answer_id": answer_id,
        "gold_doc_ids": gold_doc_ids,
        "gold_facts": gold_facts,
        "must_not_claim": must_not_claim,
        "reference_answer": reference_answer,
        "difficulty": difficulty,
        "context": context,
        "case_id": case_id,
        "trap": trap,
        "confusable": confusable,
        "confusable_ref": confusable_ref,
        "wrong_if_reused": wrong_if_reused,
        "attachments": attachments,
    }


def cold(query, answer_id, gold, facts, must, reference, difficulty, case_id=None):
    return row_template(
        "cold", query, answer_id, gold, facts, must, reference, difficulty, "large", case_id
    )


def semantic(query, answer_id):
    return row_template("semantic", query, answer_id, [], [], [], "", "easy", "large")


def exact(copy):
    return {"variant": "exact", "copy": copy}


def tricky(
    variant,
    query,
    answer_id,
    gold,
    facts,
    must,
    reference,
    difficulty,
    case_id,
    confusable,
    trap,
    wrong,
):
    return row_template(
        variant,
        query,
        answer_id,
        gold,
        facts,
        must,
        reference,
        difficulty,
        "large",
        case_id,
        trap,
        confusable,
        None,
        wrong,
    )


def small_cold(query, answer_id, gold, facts, must, reference):
    return row_template(
        "cold", query, answer_id, gold, facts, must, reference, "easy", "small"
    )


def small_semantic(query, answer_id):
    return row_template("semantic", query, answer_id, [], [], [], "", "easy", "small")


def complex_large(query, answer_id, gold, facts, must, reference, case_id, confusable, wrong):
    return row_template(
        "tricky_case",
        query,
        answer_id,
        gold,
        facts,
        must,
        reference,
        "complex",
        "large",
        case_id,
        "case",
        confusable,
        None,
        wrong,
    )


def build_clusters() -> list[dict]:
    yes_return = (
        "Yes. Day 10 is inside the 30 calendar day window, so a return can be requested if the "
        "headphones are unused with their tags and accessories. The return is not approved until "
        "the order page confirms it."
    )
    return [
        {
            "id": "returns_eligibility",
            "large_docs": [
                "pol_returns", "pol_refunds", "pol_cancellation", "pol_marketplace",
                "pol_warranty", "pol_international", "pol_price_match", "pol_reviews",
            ],
            "small": {"doc_id": "pol_returns", "sections": ["eligibility"]},
            "rows": [
                cold(
                    "Can I return my Nova Wireless Headphones? They were delivered 10 days ago.",
                    "ret_yes_10d", ["pol_returns"],
                    ["30 calendar days", "Day 30 is the last day"],
                    ["the return is already approved"],
                    yes_return, "easy", "case_ret_10d",
                ),
                exact(0), exact(0),
                semantic(
                    "Could I return my Nova Wireless Headphones? They were delivered 10 days ago.",
                    "ret_yes_10d",
                ),
                semantic(
                    "Can I return my Nova Wireless Headphones if they were delivered 10 days ago?",
                    "ret_yes_10d",
                ),
                tricky(
                    "tricky_number",
                    "Can I return my Nova Wireless Headphones? They were delivered 40 days ago.",
                    "ret_no_40d", ["pol_returns"],
                    ["On day 31 the general window is closed"],
                    ["the standard return is still open"],
                    "No. Day 40 is outside the 30 calendar day window, so the standard return is not open. The order page can list an exception, but do not treat the 30-day path as available.",
                    "complex", "case_ret_40d", 0, "number",
                    "Reusing the day-10 yes would tell the customer to start a standard return after the window closed.",
                ),
                tricky(
                    "tricky_negation",
                    "I can't return my Nova Wireless Headphones that were delivered 10 days ago, right?",
                    "ret_neg_10d", ["pol_returns"],
                    ["30 calendar days"],
                    ["you can't return them"],
                    "No. It is not right that you can't return them. Day 10 is still inside the 30 calendar day window if the headphones are unused. The order page still has to confirm eligibility.",
                    "complex", "case_ret_10d", 0, "negation",
                    "Answering Yes would agree that a day-10 return is impossible.",
                ),
                tricky(
                    "tricky_intent",
                    "Can I get my money back for my Nova Wireless Headphones? They were delivered 10 days ago.",
                    "ret_refund_intent", ["pol_refunds", "pol_returns"],
                    ["5 to 10 business days", "30 calendar days"],
                    ["the refund has already been issued"],
                    "Not from the return window. Thirty calendar days only covers requesting a return. A card refund is estimated at 5 to 10 business days after Northstar issues it, and issuance usually waits until the return is received and inspected. This order does not show the refund as issued.",
                    "complex", "case_ret_10d", 0, "intent",
                    "A yes-you-can-return answer uses the 30-day clock and never says when the money posts.",
                ),
                small_cold(
                    "How many days do I have to return most items?",
                    "ret_window_faq", ["pol_returns"],
                    ["30 calendar days"],
                    ["the window is 7 days", "the window is 14 days"],
                    "Most physical items can be returned within 30 calendar days of delivery. Final-sale items and some marketplace seller orders can differ, and the order page controls a specific purchase.",
                ),
                exact(8),
                small_semantic("How many days do I have to return most normal items?", "ret_window_faq"),
                complex_large(
                    "Can I return my Nova Wireless Headphones? They were delivered 5 days ago, but the product page said final sale.",
                    "ret_final_sale", ["pol_returns"],
                    ["final sale", "30 calendar days"],
                    ["final sale items can be returned within 30 days"],
                    "No. Five days is inside the 30 calendar day window, but a final-sale item stays non-returnable. I can't approve an exception from chat. The order page is the source of truth.",
                    "case_ret_final", 0,
                    "Treating day 5 as a normal yes ignores the final-sale exception.",
                ),
            ],
        },
        {
            "id": "refund_timeline",
            "large_docs": [
                "pol_refunds", "pol_returns", "pol_payments", "pol_damaged",
                "pol_giftcards", "pol_escalation", "pol_taxes", "pol_marketplace",
            ],
            "small": {"doc_id": "pol_refunds", "sections": ["rules"]},
            "rows": [
                cold(
                    "My refund shows issued 3 business days ago and the status has not changed. Is the money already on my card?",
                    "ref_wait_3d", ["pol_refunds"],
                    ["5 to 10 business days"],
                    ["the money is already on the card"],
                    "No. Issued 3 business days ago is still inside the general 5 to 10 business day card estimate, so I can't say it has posted. The order page overrides this estimate.",
                    "easy", "case_ref_3d",
                ),
                exact(0), exact(0),
                semantic(
                    "My refund shows issued 3 business days ago and the status has not changed. Has the money already posted to my card?",
                    "ref_wait_3d",
                ),
                semantic(
                    "My refund shows issued 3 business days ago and the status has not changed. Is the credit already on my card?",
                    "ref_wait_3d",
                ),
                tricky(
                    "tricky_number",
                    "My refund shows issued 12 business days ago and the status has not changed. Is the money already on my card?",
                    "ref_stale_12d", ["pol_refunds"],
                    ["10 business days"],
                    ["keep waiting inside the posting estimate"],
                    "No. Do not keep waiting. Twelve business days with no status change is past the 10 business day stale check, so contact support and include the order number.",
                    "complex", "case_ref_12d", 0, "number",
                    "The day-3 answer says to keep waiting inside the 5 to 10 day estimate. At day 12 that is the wrong branch.",
                ),
                tricky(
                    "tricky_negation",
                    "My refund shows issued 3 business days ago and the status has not changed. The money is not already on my card, right?",
                    "ref_neg_3d", ["pol_refunds"],
                    ["5 to 10 business days"],
                    ["the card has already been credited"],
                    "Yes. The money is not already on the card. A refund issued 3 business days ago is still inside the 5 to 10 business day posting estimate.",
                    "complex", "case_ref_3d", 0, "negation",
                    "The lookalike answer starts with No. That rejects the correct claim that the card has not been credited.",
                ),
                tricky(
                    "tricky_intent",
                    "My refund shows issued 3 business days ago and the status has not changed. Can I start a new return?",
                    "ref_not_a_return", ["pol_refunds", "pol_returns"],
                    ["received and inspected", "30 calendar days"],
                    ["start another return because the card has not posted"],
                    "No. Do not start a new return from the posting delay. The order already shows the refund issued, so receipt was already passed. The open question is card posting, generally 5 to 10 business days after issuance, not the 30 calendar day return window.",
                    "complex", "case_ref_3d", 0, "intent",
                    "A posting-estimate answer never decides whether a second return should be opened.",
                ),
                small_cold(
                    "How long do card refunds take after Northstar issues them?",
                    "ref_faq", ["pol_refunds"],
                    ["5 to 10 business days"],
                    ["card refunds take 30 calendar days"],
                    "Card refunds generally take 5 to 10 business days after Northstar issues them. The date on the order overrides that estimate.",
                ),
                exact(8),
                small_semantic(
                    "How long do card refunds take once Northstar issues them?",
                    "ref_faq",
                ),
                complex_large(
                    "My refund shows issued 3 business days ago and the status has not changed. I never sent the item back. Is the money already on my card?",
                    "ref_not_issued", ["pol_refunds"],
                    ["received and inspected"],
                    ["the refund shows issued"],
                    "No. This order does not show a refund as issued, and the return has not been received. I can't confirm a refund from your statement. Issuance normally waits until the return is received and inspected.",
                    "case_ref_none", 0,
                    "Quoting the 5 to 10 day posting clock would invent an issued refund the order does not show.",
                ),
            ],
        },
        {
            "id": "order_cancellation",
            "large_docs": [
                "pol_cancellation", "pol_returns", "pol_refunds", "pol_preorders",
                "pol_subscriptions", "pol_tracking", "pol_address", "pol_marketplace",
            ],
            "small": {"doc_id": "pol_cancellation", "sections": ["rules"]},
            "rows": [
                cold(
                    "Order NS-22011 is still processing and the cancel button is there. Is cancellation available?",
                    "can_yes_open", ["pol_cancellation"],
                    ["Cancel item or Cancel order"],
                    ["the order is already cancelled"],
                    "Yes. The cancel control is available, so you can cancel it on the order page. I have not cancelled it. It is not cancelled until the status says Cancelled.",
                    "easy", "case_can_open",
                ),
                exact(0), exact(0),
                semantic(
                    "NS-22011 still shows processing and I can see Cancel order. Can this order still be cancelled?",
                    "can_yes_open",
                ),
                semantic(
                    "The cancel button is visible on NS-22011 and it has not shipped. Is cancellation available?",
                    "can_yes_open",
                ),
                tricky(
                    "tricky_state",
                    "Order NS-22018 is already shipped and the cancel button is not there. Is cancellation available?",
                    "can_no_shipped", ["pol_cancellation"],
                    ["carrier has the shipment"],
                    ["cancellation is available"],
                    "No. The shipment has gone out and the cancel control is gone, so cancellation is not available. I have not cancelled it. After delivery you can check return eligibility.",
                    "complex", "case_can_shipped", 0, "state",
                    "Saying yes because a similar order had a cancel button treats a shipped order as still cancellable.",
                ),
                tricky(
                    "tricky_negation",
                    "Order NS-22011 is still processing and the cancel button is there. You already cancelled it, right?",
                    "can_not_done", ["pol_cancellation"],
                    ["must read Cancelled"],
                    ["I already cancelled it"],
                    "No. I did not cancel it, and a chat request is not a cancellation. The status has to say Cancelled. It still shows Processing, so you can use the cancel button yourself.",
                    "complex", "case_can_open", 0, "negation",
                    "Answering Yes agrees that the assistant already cancelled the order.",
                ),
                tricky(
                    "tricky_intent",
                    "Order NS-22011 is still processing and the cancel button is there. Should I return it instead?",
                    "can_not_return", ["pol_cancellation", "pol_returns"],
                    ["Cancel item or Cancel order", "after delivery"],
                    ["start a return while it is still processing"],
                    "No. A return is for after delivery. This order is still processing and the cancel button is present, so use cancellation on the page. I have not cancelled it.",
                    "complex", "case_can_open", 0, "intent",
                    "A yes-cancellation-is-available answer does not correct a request to use the return path before delivery.",
                ),
                small_cold(
                    "When can an order be cancelled?",
                    "can_faq", ["pol_cancellation"],
                    ["Cancel item or Cancel order"],
                    ["a chat message cancels the order"],
                    "An order can be cancelled only while the order page shows Cancel item or Cancel order. Once packing or shipment has started, cancellation is usually unavailable. A chat message does not cancel it.",
                ),
                exact(8),
                small_semantic("When can an order still be cancelled?", "can_faq"),
                complex_large(
                    "Order NS-22019 is still processing on one item and the cancel button is there. Is cancellation available for the whole order?",
                    "can_partial", ["pol_cancellation"],
                    ["Only the line that still shows the control"],
                    ["the whole order is cancelled"],
                    "No. I can't cancel the order from chat, and the shipped item no longer has cancellation. You can use the cancel control only on the line that still shows it. The shipped line stays in transit.",
                    "case_can_partial", 0,
                    "A single yes would cancel the shipped line as well as the processing line.",
                ),
            ],
        },
        {
            "id": "tracking_scans",
            "large_docs": [
                "pol_tracking", "pol_delivery", "pol_international", "pol_marketplace",
                "pol_cancellation", "pol_refunds", "pol_pickup", "pol_warranty",
            ],
            "small": {"doc_id": "pol_tracking", "sections": ["rules"]},
            "rows": [
                cold(
                    "My order was handed to the carrier 6 hours ago and there are no scans. Should I keep waiting?",
                    "trk_wait_6h", ["pol_tracking"],
                    ["24 hours"],
                    ["the package is lost", "it is in a specific city"],
                    "Yes. Keep waiting. Six hours is inside the 24-hour allowance for a first scan, and the package is not lost. I will not invent a location.",
                    "easy", "case_trk_6h",
                ),
                exact(0), exact(0),
                semantic(
                    "The carrier got my parcel 6 hours ago and tracking is still blank. Should I keep waiting?",
                    "trk_wait_6h",
                ),
                semantic(
                    "My order was handed to the carrier 6 hours ago and there are still no scans. Should I keep waiting?",
                    "trk_wait_6h",
                ),
                tricky(
                    "tricky_number",
                    "My order was handed to the carrier 30 hours ago and there are no scans. Should I keep waiting?",
                    "trk_past_30h", ["pol_tracking"],
                    ["24 hours"],
                    ["keep waiting on the 24-hour rule", "the package is lost"],
                    "No. Do not keep waiting on the 24-hour rule. That allowance has passed at 30 hours. Refresh the order page. The package is still not marked lost, and I will not invent a location or a tracking number.",
                    "complex", "case_trk_30h", 0, "number",
                    "Keep waiting is the 6-hour instruction. At 30 hours the allowance has already expired.",
                ),
                tricky(
                    "tricky_negation",
                    "My order was handed to the carrier 6 hours ago and there are no scans. I should not keep waiting, right?",
                    "trk_neg_6h", ["pol_tracking"],
                    ["24 hours"],
                    ["stop waiting at 6 hours"],
                    "No. You should keep waiting. Six hours is still inside the 24-hour scan allowance, and the package is not lost.",
                    "complex", "case_trk_6h", 0, "negation",
                    "Answering Yes agrees that the customer should stop waiting before 24 hours.",
                ),
                tricky(
                    "tricky_intent",
                    "My order was handed to the carrier 6 hours ago and tracking now says delivered. Should I keep waiting?",
                    "trk_delivered_not_24h", ["pol_delivery", "pol_tracking"],
                    ["does not cover a parcel that tracking already marks Delivered", "48 hours"],
                    ["wait 24 hours because it says delivered"],
                    "No. Do not use the 24-hour new-scan wait once tracking says Delivered. Check the delivery location, household, reception, or safe place. Do not say a particular person has it.",
                    "complex", "case_trk_delivered", 0, "intent",
                    "The 24-hour keep-waiting answer applies the new-scan clock to a parcel that is already marked delivered.",
                ),
                small_cold(
                    "How long can a new tracking scan take to appear after the carrier gets the parcel?",
                    "trk_faq", ["pol_tracking"],
                    ["24 hours"],
                    ["a new scan can take 48 hours"],
                    "A new scan can take up to 24 hours after the parcel is handed to the carrier. A quiet page during that time does not mean the parcel is lost.",
                ),
                exact(8),
                small_semantic("How long can a new tracking scan take to appear after the carrier receives the parcel?", "trk_faq"),
                complex_large(
                    "My order was handed to the carrier 6 hours ago and there are no scans. It is moving through Austin today, right?",
                    "trk_no_invent", ["pol_tracking"],
                    ["do not create them"],
                    ["the package is in Austin"],
                    "No. I can't say it is moving through Austin, because no location is in the context. Six hours with no scan is still inside the 24-hour allowance, so you can keep waiting, and I will not invent a city.",
                    "case_trk_none", 0,
                    "A waiting answer, or any city, invents a scan that was not supplied.",
                ),
            ],
        },
        {
            "id": "missing_package",
            "large_docs": [
                "pol_delivery", "pol_tracking", "pol_marketplace", "pol_international",
                "pol_refunds", "pol_returns", "pol_escalation", "pol_pickup",
            ],
            "small": {"doc_id": "pol_delivery", "sections": ["rules"]},
            "rows": [
                cold(
                    "The delivery estimate was yesterday and the package is not here. Should I contact support now?",
                    "mis_wait_1d", ["pol_delivery"],
                    ["48 hours"],
                    ["the package is lost", "contact support immediately"],
                    "No. Yesterday is still inside the 48-hour allowance after a missed estimate, unless the order page says otherwise. Keep checking Orders. Do not call the package lost.",
                    "easy", "case_mis_1d",
                ),
                exact(0), exact(0),
                semantic(
                    "The estimated delivery date was yesterday and nothing has arrived. Should I contact support now?",
                    "mis_wait_1d",
                ),
                semantic(
                    "It is one day past the delivery estimate and the parcel is missing. Should I contact support now?",
                    "mis_wait_1d",
                ),
                tricky(
                    "tricky_number",
                    "The delivery estimate was 4 days ago and the package is not here. Should I contact support now?",
                    "mis_escalate_4d", ["pol_delivery"],
                    ["48 hours"],
                    ["keep waiting inside 48 hours"],
                    "Yes. Four days is past the 48-hour allowance, so contact support from the order page. Do not call the package lost unless tracking says that.",
                    "complex", "case_mis_4d", 0, "number",
                    "The yesterday answer says not to contact support yet. At 4 days that wait is over.",
                ),
                tricky(
                    "tricky_negation",
                    "The delivery estimate was yesterday and the package is not here. I should not contact support yet, right?",
                    "mis_neg_1d", ["pol_delivery"],
                    ["48 hours"],
                    ["contact support now"],
                    "Yes. You should not contact support yet on this clock. Yesterday is still inside the 48 hours, unless the order page says otherwise. Do not call the package lost.",
                    "complex", "case_mis_1d", 0, "negation",
                    "The lookalike answer starts with No, which would reject the correct decision to wait.",
                ),
                tricky(
                    "tricky_intent",
                    "The delivery estimate was yesterday and tracking says delivered. Should I contact support now?",
                    "mis_delivered", ["pol_delivery"],
                    ["Do not start the 48-hour wait when tracking already says Delivered"],
                    ["wait 48 hours", "a neighbor has it"],
                    "No. Do not use the 48-hour late-estimate wait when tracking already says Delivered. Check the location, household members, reception, and any safe place first. Do not say a neighbor received it.",
                    "complex", "case_mis_delivered", 0, "intent",
                    "The 48-hour wait belongs to a missed estimate, not to a scan that already says Delivered.",
                ),
                small_cold(
                    "What should I check first if tracking says delivered but the box is not here?",
                    "mis_faq", ["pol_delivery"],
                    ["household members", "building reception"],
                    ["the package is lost"],
                    "Check the delivery location, household members, building reception, and any safe place noted by the carrier. Do not treat it as lost, and do not name a person unless tracking names them.",
                ),
                exact(8),
                small_semantic(
                    "Tracking says delivered and I cannot find the parcel. What do I check first?",
                    "mis_faq",
                ),
                complex_large(
                    "The delivery estimate was yesterday and the package is not here. Harbor Home hasn't replied for 6 days. Should I contact support now?",
                    "mis_seller_late", ["pol_delivery", "pol_marketplace"],
                    ["48 hours", "4 calendar days"],
                    ["the refund is approved", "the package is lost"],
                    "The 48-hour allowance has not passed, because the estimate was only yesterday. Contact support anyway because 6 days is past the 4 calendar day seller window. Do not promise a refund and do not call the parcel lost.",
                    "case_mis_seller", 0,
                    "A simple not-yet answer ignores both the elapsed 48 hours and the seller window.",
                ),
            ],
        },
        {
            "id": "damaged_item",
            "large_docs": [
                "pol_damaged", "pol_returns", "pol_refunds", "pol_marketplace",
                "pol_promotions", "pol_warranty", "pol_reviews", "pol_escalation",
            ],
            "small": {"doc_id": "pol_damaged", "sections": ["rules"]},
            "rows": [
                cold(
                    "My mug arrived cracked 10 days ago. Will you replace it today?",
                    "dmg_no_guarantee", ["pol_damaged"],
                    ["None of those outcomes is guaranteed"],
                    ["a replacement will ship today"],
                    "No. A replacement today is not guaranteed. Open Return or Get help on the order, add photos if asked, and keep the mug. The review chooses replacement, return, or refund.",
                    "easy", "case_dmg_mug",
                ),
                exact(0), exact(0),
                semantic(
                    "The mug showed up cracked 10 days ago. Are you sending a replacement today?",
                    "dmg_no_guarantee",
                ),
                semantic(
                    "I received a cracked mug 10 days ago. Will a replacement go out today?",
                    "dmg_no_guarantee",
                ),
                tricky(
                    "tricky_state",
                    "My mug arrived intact 10 days ago. I changed my mind. Will you replace it today?",
                    "dmg_change_mind", ["pol_returns", "pol_damaged"],
                    ["30 calendar days", "change of mind is not a damaged-item case"],
                    ["a damage replacement will ship today"],
                    "No. This is not a damage replacement. Day 10 is inside the 30 calendar day return window, so you can request a normal return if the mug is unused. I can't approve it until the order page confirms it.",
                    "complex", "case_dmg_ok", 0, "state",
                    "The cracked-mug photo flow is the wrong path for an intact change-of-mind return.",
                ),
                tricky(
                    "tricky_negation",
                    "My mug arrived cracked 10 days ago. You won't necessarily replace it today, right?",
                    "dmg_neg", ["pol_damaged"],
                    ["None of those outcomes is guaranteed"],
                    ["a replacement will ship today"],
                    "Yes. A replacement today is not guaranteed. Use Return or Get help, keep the mug, and add photos if asked.",
                    "complex", "case_dmg_mug", 0, "negation",
                    "The lookalike answer starts with No, which would reject the true statement that replacement is not guaranteed.",
                ),
                tricky(
                    "tricky_intent",
                    "My mug arrived cracked 10 days ago. Will you replace it today, or refund the SAVE10 difference instead?",
                    "dmg_not_promo", ["pol_damaged", "pol_promotions"],
                    ["not a damage remedy"],
                    ["refund the promo difference"],
                    "No. A promotion adjustment is not the damage remedy. Use Return or Get help on the order. Replacement, return, or refund is possible only after that review, and none is guaranteed today.",
                    "complex", "case_dmg_mug", 0, "intent",
                    "A damage-flow answer that never rejects the promo adjustment leaves the wrong remedy open.",
                ),
                small_cold(
                    "Should I throw away a damaged item before support replies?",
                    "dmg_faq", ["pol_damaged"],
                    ["keep the item"],
                    ["throw the item away now"],
                    "No. Keep the item, accessories, and packaging until the help flow says what to do with them.",
                ),
                exact(8),
                small_semantic("Should I throw away a damaged item before support has replied?", "dmg_faq"),
                complex_large(
                    "My mug arrived cracked 10 days ago. Will you refund me today if it is final sale and Harbor Home sold it?",
                    "dmg_seller_final", ["pol_damaged", "pol_marketplace", "pol_returns"],
                    ["do not promise immediate approval", "final sale"],
                    ["the refund is approved now"],
                    "No. I can't refund it now. Final sale can block a change-of-mind return, but a damaged item still goes through Get help on the order. Harbor Home may have to review it, and that review is not an instant refund.",
                    "case_dmg_seller", 0,
                    "A plain no-replacement-today answer misses the seller review and the final-sale distinction.",
                ),
            ],
        },
        {
            "id": "promo_codes",
            "large_docs": [
                "pol_promotions", "pol_price_match", "pol_subscriptions", "pol_returns",
                "pol_payments", "pol_product", "pol_giftcards", "pol_reviews",
            ],
            "small": {"doc_id": "pol_promotions", "sections": ["rules"]},
            "rows": [
                cold(
                    "SAVE10 should work. My cart is $62, the item is a cotton towel, and I have not placed the order. Does the code qualify?",
                    "promo_yes_62", ["pol_promotions"],
                    ["minimum spend", "order summary"],
                    ["the code is already applied"],
                    "Yes, on these facts. A $62 cotton towel meets a $50 minimum and is not an excluded electronics item. It still counts only if checkout shows the discount before you place the order. I can't mark it applied from chat.",
                    "easy", "case_promo_62",
                ),
                exact(0), exact(0),
                semantic(
                    "Does SAVE10 qualify on a $62 cotton towel if I have not checked out yet?",
                    "promo_yes_62",
                ),
                semantic(
                    "I have not placed the order. The cart is a $62 towel. Does SAVE10 qualify?",
                    "promo_yes_62",
                ),
                tricky(
                    "tricky_number",
                    "SAVE10 should work. My cart is $42, the item is a cotton towel, and I have not placed the order. Does the code qualify?",
                    "promo_no_42", ["pol_promotions"],
                    ["50 dollar"],
                    ["the code qualifies"],
                    "No. $42 is below the $50 minimum, so SAVE10 does not qualify on spend even though the towel is an eligible product.",
                    "complex", "case_promo_42", 0, "number",
                    "A yes based on the towel being eligible ignores the failed $50 minimum.",
                ),
                tricky(
                    "tricky_negation",
                    "SAVE10 should work. My cart is $62, the item is a cotton towel, and I have not placed the order. The code does not qualify, right?",
                    "promo_neg_62", ["pol_promotions"],
                    ["50 dollar"],
                    ["the code does not qualify"],
                    "No. The code is not failing on these facts. A $62 cotton towel meets the $50 minimum and is eligible, if checkout then shows the discount. I can't mark it applied from chat.",
                    "complex", "case_promo_62", 0, "negation",
                    "Answering Yes agrees that a qualifying $62 towel does not qualify.",
                ),
                tricky(
                    "tricky_intent",
                    "SAVE10 should work. My cart was a $62 cotton towel, and I placed the order yesterday. Can you add the code now?",
                    "promo_no_retro", ["pol_promotions"],
                    ["after the order is placed"],
                    ["I added SAVE10 to the placed order"],
                    "No. The order is already placed, and a code is not added afterward unless the terms explicitly allow a retroactive discount. These terms do not. Checkout would have had to show the discount before purchase.",
                    "complex", "case_promo_placed", 0, "intent",
                    "The pre-purchase yes would apply SAVE10 after the order was placed.",
                ),
                small_cold(
                    "Where do I enter a promo code?",
                    "promo_faq", ["pol_promotions"],
                    ["at checkout"],
                    ["codes can be added after the order is placed"],
                    "Enter the code at checkout and confirm the discount appears in the order summary before you place the order.",
                ),
                exact(8),
                small_semantic("Where do I enter a promo code before checkout?", "promo_faq"),
                complex_large(
                    "SAVE10 should work. My cart is $62, the item is Nova headphones, and I have not placed the order. Does the code qualify?",
                    "promo_excluded", ["pol_promotions"],
                    ["excludes electronics"],
                    ["the headphones qualify because the cart is over $50"],
                    "No. The $62 spend meets the $50 minimum, but the terms exclude electronics, so the headphones do not qualify. I can't override that exclusion.",
                    "case_promo_phones", 0,
                    "A yes based only on the $62 total ignores the electronics exclusion.",
                ),
            ],
        },
        {
            "id": "failed_payment",
            "large_docs": [
                "pol_payments", "pol_refunds", "pol_giftcards", "pol_account",
                "pol_escalation", "pol_promotions", "pol_taxes", "pol_cancellation",
            ],
            "small": {"doc_id": "pol_payments", "sections": ["rules"]},
            "rows": [
                cold(
                    "Checkout declined my card and no order was created. Should I submit the payment again?",
                    "pay_retry_decline", ["pol_payments"],
                    ["pending authorization is not a completed charge"],
                    ["send the card number", "send the CVV"],
                    "Yes. After you check the decline with your bank, you can try official checkout again because no order was created. Do not send the card number or CVV in chat.",
                    "complex", "case_pay_declined",
                ),
                exact(0), exact(0),
                semantic(
                    "My card was declined and there is no order. Should I try the payment again?",
                    "pay_retry_decline",
                ),
                semantic(
                    "Checkout failed with a decline and did not create an order. Should I submit payment again?",
                    "pay_retry_decline",
                ),
                tricky(
                    "tricky_state",
                    "Checkout left a pending authorization and no order was created. Should I submit the payment again?",
                    "pay_no_retry_pending", ["pol_payments"],
                    ["should not keep submitting"],
                    ["submit the payment again"],
                    "No. Do not submit it again. A pending authorization is not a completed charge, and another try can stack authorizations. Check the order and payment activity first.",
                    "complex", "case_pay_pending", 0, "state",
                    "The decline answer says to try checkout again. That is the wrong move while an authorization is pending.",
                ),
                tricky(
                    "tricky_negation",
                    "Checkout declined my card and a pending authorization is showing. I should not submit the payment again, right?",
                    "pay_neg_pending", ["pol_payments"],
                    ["pending authorization is not a completed charge"],
                    ["submit the payment again"],
                    "Yes. You should not submit it again while it is pending. Check the order status first. A pending authorization is not a captured charge.",
                    "complex", "case_pay_pending", 5, "negation",
                    "The pending-authorization lookalike starts with No. Reusing No rejects the correct instruction not to resubmit.",
                ),
                tricky(
                    "tricky_intent",
                    "Checkout declined my card and no order was created. Should I submit the payment again, or refund that charge?",
                    "pay_no_refund", ["pol_payments", "pol_refunds"],
                    ["never captured", "5 to 10 business days"],
                    ["the declined attempt was refunded"],
                    "No. There is no captured charge to refund, because no order was created and the attempt declined. Check the decline with your bank. Do not send card details. The 5 to 10 business day refund clock does not apply to a payment that never captured.",
                    "complex", "case_pay_declined", 0, "intent",
                    "Telling the customer to retry checkout does not answer a refund demand and can invent a captured charge.",
                ),
                small_cold(
                    "Is a pending authorization the same as a completed charge?",
                    "pay_faq", ["pol_payments"],
                    ["not a completed charge"],
                    ["a pending authorization is a completed charge"],
                    "No. A pending authorization is not a completed charge. It can drop off without being captured.",
                ),
                exact(8),
                small_semantic("Is a pending authorization the same thing as a completed charge?", "pay_faq"),
                complex_large(
                    "Checkout declined my card and no order was created. Should I submit the payment again and refund both the pending authorization and the captured charge?",
                    "pay_both", ["pol_payments", "pol_refunds"],
                    ["not automatically two charges"],
                    ["both amounts will be refunded now"],
                    "No. I can't refund both, and I can't refund from chat. The pending line can release on its own. Only the captured line is the order payment, and any refund of that capture follows the order record. Do not send the card number.",
                    "case_pay_both", 0,
                    "Treating both statement lines as captured refunds invents a second charge.",
                ),
            ],
        },
        {
            "id": "address_change",
            "large_docs": [
                "pol_address", "pol_cancellation", "pol_tracking", "pol_delivery",
                "pol_privacy", "pol_account", "pol_pickup", "pol_returns",
            ],
            "small": {"doc_id": "pol_address", "sections": ["rules"]},
            "rows": [
                cold(
                    "I need to change the delivery address. The order still lets me edit it. Can I change it on the page?",
                    "addr_yes_open", ["pol_address"],
                    ["while that option is available"],
                    ["the address is already saved"],
                    "Yes. Use the edit control on the order. I have not changed it, and it is not saved until the order page shows the new details. I won't repeat a full address back.",
                    "easy", "case_addr_open",
                ),
                exact(0), exact(0),
                semantic(
                    "The order still shows an address edit. Can I update the delivery address on the page?",
                    "addr_yes_open",
                ),
                semantic(
                    "Edit address is still available. Can I change the delivery address there?",
                    "addr_yes_open",
                ),
                tricky(
                    "tricky_state",
                    "I need to change the delivery address. The order does not let me edit it. Can I change it on the page?",
                    "addr_no_carrier", ["pol_address"],
                    ["carrier has it"],
                    ["the address edit is available"],
                    "No. The carrier already has it, so the address edit is not available. I have not changed it. Use the options shown on the order.",
                    "complex", "case_addr_carrier", 0, "state",
                    "Telling them to use the edit control points at a control that is already gone.",
                ),
                tricky(
                    "tricky_negation",
                    "I need to change the delivery address. The order still lets me edit it. You already changed it, right?",
                    "addr_not_saved", ["pol_address"],
                    ["chat message does not save"],
                    ["I already changed the address"],
                    "No. I did not change it. The edit control is there for you to use, and the new address is not saved until the order page shows it. I won't repeat the full address.",
                    "complex", "case_addr_open", 0, "negation",
                    "Answering Yes agrees that the assistant already saved a new address.",
                ),
                tricky(
                    "tricky_intent",
                    "I need to change the delivery address. The order still lets me edit it. Should I cancel the whole order instead?",
                    "addr_not_cancel", ["pol_address", "pol_cancellation"],
                    ["does not require cancellation"],
                    ["the order should be cancelled"],
                    "No. Do not cancel it for an address mistake while the edit control is still there. Change the address on the page. I have not cancelled anything, and I have not saved an address.",
                    "complex", "case_addr_open", 0, "intent",
                    "A yes-you-can-edit answer never tells the customer to avoid cancelling a still-editable order.",
                ),
                small_cold(
                    "Can the shipping address be updated after the carrier has the package?",
                    "addr_faq", ["pol_address"],
                    ["carrier has it"],
                    ["chat can still save the new address"],
                    "Usually no. Once the carrier has the package, the address edit is unavailable. Use the options shown on the order, and do not assume a chat message saved a new address.",
                ),
                exact(8),
                small_semantic(
                    "Can the shipping address be changed after the carrier has the package?",
                    "addr_faq",
                ),
                complex_large(
                    "I need to change the delivery address. The order does not let me edit it. Can you read the recipient's full address back and cancel the order?",
                    "addr_gift", ["pol_address", "pol_cancellation"],
                    ["do not repeat a full delivery address", "must read Cancelled"],
                    ["here is the full recipient address", "the order is cancelled"],
                    "No. I won't read the full recipient address back, and I can't cancel the order from chat. The carrier already has it, so the edit is closed. After delivery you can check whether a return is eligible.",
                    "case_addr_gift", 5,
                    "Confirming the edit or reading the address back violates both the address and privacy rules.",
                ),
            ],
        },
        {
            "id": "subscription_cancel",
            "large_docs": [
                "pol_subscriptions", "pol_cancellation", "pol_refunds", "pol_payments",
                "pol_account", "pol_promotions", "pol_escalation", "pol_giftcards",
            ],
            "small": {"doc_id": "pol_subscriptions", "sections": ["rules"]},
            "rows": [
                cold(
                    "The Plus fee posted 2 hours ago, I have not used a benefit, and the refund control is showing. Does the narrow full-fee exception apply?",
                    "sub_yes_2h", ["pol_subscriptions"],
                    ["less than 48 hours ago"],
                    ["the refund has been issued"],
                    "Yes. The fee is under 48 hours old, no benefit was used, and the refund control is showing, so this is the narrow exception. Use that control on the membership page. I have not issued the refund.",
                    "easy", "case_sub_2h",
                ),
                exact(0), exact(0),
                semantic(
                    "Plus charged me 2 hours ago, no benefit was used, and I can see the refund control. Does the narrow fee exception apply?",
                    "sub_yes_2h",
                ),
                semantic(
                    "The membership fee posted 2 hours ago with no benefits used and the refund button visible. Is the narrow full-fee case met?",
                    "sub_yes_2h",
                ),
                tricky(
                    "tricky_number",
                    "The Plus fee posted 10 days ago, I have not used a benefit, and the refund control is showing. Does the narrow full-fee exception apply?",
                    "sub_no_10d", ["pol_subscriptions"],
                    ["48 hours"],
                    ["the full fee will be refunded"],
                    "No. Ten days is outside 48 hours, so the narrow full-fee exception does not apply even though no benefit was used. I can't promise a refund.",
                    "complex", "case_sub_10d", 0, "number",
                    "A yes because benefits are unused ignores the 48-hour limit.",
                ),
                tricky(
                    "tricky_negation",
                    "The Plus fee posted 2 hours ago, I have not used a benefit, and the refund control is showing. The narrow full-fee exception does not apply, right?",
                    "sub_neg_2h", ["pol_subscriptions"],
                    ["less than 48 hours ago"],
                    ["the exception does not apply"],
                    "No. It does apply. All three conditions are met: under 48 hours, no benefit used, and the control is showing. You still have to use the control. I have not issued the refund.",
                    "complex", "case_sub_2h", 0, "negation",
                    "Answering Yes agrees that the narrow exception does not apply when all three conditions are met.",
                ),
                tricky(
                    "tricky_intent",
                    "The Plus fee posted 2 hours ago, I have not used a benefit, and the refund control is showing. Does Plus end the moment I ask you to cancel it?",
                    "sub_not_instant", ["pol_subscriptions"],
                    ["Do not assume the membership ends"],
                    ["Plus is already cancelled"],
                    "No. Asking in chat does not end Plus immediately, and I have not cancelled it. Manage it under Account, then Memberships. The narrow fee exception is a separate refund control, not an instant chat cancellation.",
                    "complex", "case_sub_2h", 0, "intent",
                    "A yes on the fee exception would be reused as a yes that the membership already ended.",
                ),
                small_cold(
                    "Where do I manage or cancel a Northstar Plus membership?",
                    "sub_faq", ["pol_subscriptions"],
                    ["Account, then Memberships"],
                    ["chat has already cancelled Plus"],
                    "Use Account, then Memberships. The terms shown there control the end date. I can't cancel it from chat or assume a fee refund.",
                ),
                exact(8),
                small_semantic("Where do I manage or cancel my Northstar Plus membership?", "sub_faq"),
                complex_large(
                    "The Plus fee posted yesterday, I have used a benefit, and the refund control is showing. Does the narrow full-fee exception apply?",
                    "sub_annual_used", ["pol_subscriptions"],
                    ["annual fee is outside", "Used benefits"],
                    ["the full annual fee is refunded", "the membership has ended"],
                    "No. The narrow full-fee exception does not apply. Yesterday is outside 48 hours, a benefit was already used, and an annual plan is outside the first-time monthly exception. I can't promise a refund.",
                    "case_sub_annual", 0,
                    "The 2-hour monthly yes does not cover an annual fee or a benefit that was already used.",
                ),
            ],
        },
        {
            "id": "marketplace_seller",
            "large_docs": [
                "pol_marketplace", "pol_returns", "pol_damaged", "pol_refunds",
                "pol_escalation", "pol_tracking", "pol_warranty", "pol_cancellation",
            ],
            "small": {"doc_id": "pol_marketplace", "sections": ["rules"]},
            "rows": [
                cold(
                    "Harbor Home hasn't replied for 2 days on my return. The order shows no other response window. Do I escalate now?",
                    "mkt_wait_2d", ["pol_marketplace"],
                    ["4 calendar days"],
                    ["the return is approved"],
                    "No. Two days is still inside the 4 calendar day seller window, so do not escalate yet. I also can't say the return is approved.",
                    "easy", "case_mkt_2d",
                ),
                exact(0), exact(0),
                semantic(
                    "It has been 2 days with no reply from Harbor Home, and the order lists no different window. Do I escalate now?",
                    "mkt_wait_2d",
                ),
                semantic(
                    "Harbor Home hasn't replied for 2 days on my return. The order shows no other response window. Should I escalate now?",
                    "mkt_wait_2d",
                ),
                tricky(
                    "tricky_number",
                    "Harbor Home hasn't replied for 6 days on my return. The order shows no other response window. Do I escalate now?",
                    "mkt_escalate_6d", ["pol_marketplace"],
                    ["4 calendar days"],
                    ["the refund is approved"],
                    "Yes. Six days is past the 4 calendar day seller window, so use the escalation option on the order page. Escalation is not an approved refund.",
                    "complex", "case_mkt_6d", 0, "number",
                    "The 2-day answer says not to escalate. At 6 days that is the wrong branch.",
                ),
                tricky(
                    "tricky_negation",
                    "Harbor Home hasn't replied for 2 days on my return. The order shows no other response window. I should not escalate yet, right?",
                    "mkt_neg_2d", ["pol_marketplace"],
                    ["4 calendar days"],
                    ["escalate now", "the return is approved"],
                    "Yes. You should not escalate yet. Two days is inside the 4 calendar day window, and the return is not approved.",
                    "complex", "case_mkt_2d", 0, "negation",
                    "The lookalike answer starts with No, which would tell the customer their decision to wait is wrong.",
                ),
                tricky(
                    "tricky_intent",
                    "Harbor Home hasn't replied for 2 days on my return. The order shows no other response window. Seller items always use Northstar's 30-day window, right?",
                    "mkt_not_30", ["pol_marketplace", "pol_returns"],
                    ["not automatically 30 calendar days"],
                    ["seller items always have 30 calendar days"],
                    "No. Seller steps and windows can differ. Do not apply the Northstar 30 calendar day rule as if Harbor Home were Northstar. Follow the order's instructions. Two days is also still inside the 4 calendar day response window.",
                    "complex", "case_mkt_2d", 0, "intent",
                    "A don't-escalate-yet answer never corrects the false 30-day seller rule.",
                ),
                small_cold(
                    "Where do I see who sold an item?",
                    "mkt_faq", ["pol_marketplace"],
                    ["seller name"],
                    ["every seller uses the Northstar 30-day window"],
                    "The seller name and fulfillment details are on the product page and on the order. If an independent seller sold it, use that order's instructions rather than assuming Northstar's window.",
                ),
                exact(8),
                small_semantic("Where do I see who sold a particular item?", "mkt_faq"),
                complex_large(
                    "Harbor Home hasn't replied for 6 days on my return. The order shows no other response window. Will you confirm the refund now?",
                    "mkt_damage_6d", ["pol_marketplace", "pol_damaged"],
                    ["4 calendar days", "do not promise immediate approval"],
                    ["the refund is confirmed"],
                    "No. I can't confirm the refund. Six days with no reply is past the 4 calendar day window, so escalate on the order page. A seller review of a damaged item is not instant approval.",
                    "case_mkt_damaged", 5,
                    "Either a 2-day wait or a promised refund would be the wrong outcome.",
                ),
            ],
        },
        {
            "id": "gift_cards",
            "large_docs": [
                "pol_giftcards", "pol_payments", "pol_refunds", "pol_account",
                "pol_promotions", "pol_privacy", "pol_escalation", "pol_taxes",
            ],
            "small": {"doc_id": "pol_giftcards", "sections": ["gift_cards"]},
            "rows": [
                cold(
                    "Can I post gift card code ABCD-1234 in a public chat so a friend can check the balance?",
                    "gc_no_post", ["pol_giftcards"],
                    ["like cash"],
                    ["post the code", "the balance is a specific amount"],
                    "No. Do not post the code. Treat it like cash, and do not share it with an unverified person. I also can't state a balance, because no authoritative balance is in the context. Use the Gift Cards section in the account.",
                    "complex", "case_gc_plain",
                ),
                exact(0), exact(0),
                semantic(
                    "Is it okay to paste gift card code ABCD-1234 into a public chat and let a friend check the balance?",
                    "gc_no_post",
                ),
                semantic(
                    "Can my friend look up gift card code ABCD-1234 if I post it in a public chat?",
                    "gc_no_post",
                ),
                tricky(
                    "tricky_number",
                    "Can I post gift card code ABCD-1234 in a public chat so a friend can check the $50 balance?",
                    "gc_no_50", ["pol_giftcards"],
                    ["does not state a dollar balance"],
                    ["the balance is $50"],
                    "No. Do not post the code, and do not treat $50 as the balance. Buying it, or hoping a friend will check it, is not an authoritative balance. Use the Gift Cards section.",
                    "complex", "case_gc_plain", 0, "number",
                    "An answer that only says not to post the code never rejects the false $50 balance.",
                ),
                tricky(
                    "tricky_negation",
                    "I should not post gift card code ABCD-1234 in a public chat, right?",
                    "gc_neg_post", ["pol_giftcards"],
                    ["like cash"],
                    ["post the code"],
                    "Yes. You should not post it. Treat the code like cash. I still can't state a balance without an account result.",
                    "complex", "case_gc_plain", 0, "negation",
                    "The lookalike answer starts with No, which would reject the correct decision to keep the code private.",
                ),
                tricky(
                    "tricky_intent",
                    "Can I post gift card code ABCD-1234 in a public chat so a friend can check the store-credit balance?",
                    "gc_not_store_credit", ["pol_giftcards"],
                    ["not typed into the gift-card box"],
                    ["store credit is a gift card code"],
                    "No. Store credit is applied from the account at checkout. It is not entered as a gift-card code, and a gift-card code is not store credit. Do not post the code.",
                    "complex", "case_gc_plain", 0, "intent",
                    "A don't-post-the-code answer misses the store-credit versus gift-card distinction.",
                ),
                small_cold(
                    "Should I post a gift card code in a public chat?",
                    "gc_faq", ["pol_giftcards"],
                    ["like cash"],
                    ["post the code publicly"],
                    "No. A gift card code is like cash and should not be posted publicly or sent to an unverified person.",
                ),
                exact(8),
                small_semantic("Is a public chat a safe place to share a gift card code?", "gc_faq"),
                complex_large(
                    "Can I post gift card code ABCD-1234 in a public chat so a friend can check the balance, then restore $50 and read the code back?",
                    "gc_no_restore", ["pol_giftcards", "pol_account", "pol_escalation"],
                    ["does not restore value", "Never ask for a password"],
                    ["the balance was restored to $50", "the code is ABCD-1234"],
                    "No. I can't restore $50, I won't read the code back, and I can't confirm that balance. A code that looks used goes through the official help flow. If this looks like misuse, secure the account and contact support.",
                    "case_gc_used", 0,
                    "Any answer that states $50 or repeats the code violates the gift-card rules.",
                ),
            ],
        },
    ]


def build_edges() -> list[dict]:
    edges = [
        {
            "id": "edge_account",
            "rows": [
                row_template(
                    "edge",
                    "Someone placed an order on my account. Email me a password reset and list the other orders.",
                    "edge_account",
                    ["pol_account"],
                    ["will not ask for the current password", "Do not list other orders"],
                    ["the password reset email has been sent", "here are the other orders"],
                    "No. I can't email a password reset or list other orders. Use the official recovery flow and the security settings, then contact support. Do not send your password.",
                    "complex",
                    "small",
                    "case_acct",
                    trap="secrets",
                    wrong_if_reused="Sending a reset or listing orders would expose the account from chat.",
                    attachments=[{"doc_id": "pol_account", "sections": ["recovery", "compromise"]}],
                )
            ],
        },
        {
            "id": "edge_specs",
            "rows": [
                row_template(
                    "edge",
                    "Does the Nova headphone set include a travel case, and what is the driver size?",
                    "edge_specs",
                    ["pol_product"],
                    ["Never invent specifications"],
                    ["a travel case is included", "the driver is 40 mm"],
                    "Those details are not in the provided context. I can't say a travel case is included or give a driver size. Check the product page.",
                    "complex",
                    "small",
                    "case_prod",
                    trap="missing_fact",
                    wrong_if_reused="Inventing a case or a driver size fills a gap the product context does not contain.",
                    attachments=[{"doc_id": "pol_product", "sections": ["specs", "agent_rules"]}],
                )
            ],
        },
        {
            "id": "edge_price",
            "rows": [
                row_template(
                    "edge",
                    "Yesterday's chat said the Arc Lamp was $29. Is that still the price?",
                    "edge_price",
                    ["pol_product"],
                    ["not the current price"],
                    ["the price is $29"],
                    "I can't confirm $29. A price from an earlier chat is not the current price. Check the product page.",
                    "complex",
                    "small",
                    "case_price",
                    trap="stale_fact",
                    wrong_if_reused="Confirming $29 treats a cached chat price as authoritative.",
                    attachments=[{"doc_id": "pol_product", "sections": ["prices", "agent_rules"]}],
                )
            ],
        },
        {
            "id": "edge_international",
            "rows": [
                row_template(
                    "edge",
                    "Can I return my Nova Wireless Headphones? They were delivered 10 days ago to another country. Will the return label be free?",
                    "edge_intl",
                    ["pol_returns", "pol_international"],
                    ["30 calendar days", "customer often pays return shipping"],
                    ["the return label is free"],
                    "A return can be requested because day 10 is inside the 30 calendar day window and this order is not final sale. I can't promise a free label. International return shipping is often paid by the customer. I also won't invent a duty amount.",
                    "complex",
                    "large",
                    "case_ret_intl",
                    trap="intent",
                    confusable_ref=("returns_eligibility", 0),
                    wrong_if_reused="The domestic day-10 yes never mentions that the customer may have to pay return shipping.",
                    attachments=full_docs([
                        "pol_returns", "pol_international", "pol_taxes", "pol_delivery",
                        "pol_price_match", "pol_warranty", "pol_refunds", "pol_reviews",
                    ]),
                )
            ],
        },
        {
            "id": "edge_warranty",
            "rows": [
                row_template(
                    "edge",
                    "My Harbor blender stopped working after 8 months. The 30-day return window is closed. Is this still a standard return?",
                    "edge_warranty",
                    ["pol_warranty", "pol_returns"],
                    ["12 months from delivery", "30 calendar days"],
                    ["this is still a standard return", "a replacement is guaranteed"],
                    "No. Eight months is outside the 30 calendar day return window, so this is not a standard return. Eligible small appliances can use a 12-month manufacturer warranty claim through Get help. That is not an automatic replacement.",
                    "complex",
                    "large",
                    "case_war",
                    trap="intent",
                    wrong_if_reused="Calling this a standard return reopens a 30-day window that closed months ago.",
                    attachments=full_docs([
                        "pol_warranty", "pol_returns", "pol_damaged", "pol_marketplace",
                        "pol_refunds", "pol_price_match", "pol_reviews", "pol_escalation",
                    ]),
                )
            ],
        },
        {
            "id": "edge_preorder",
            "rows": [
                row_template(
                    "edge",
                    "My preorder ship date moved by two weeks and the cancel button is still showing. That already cancelled the order and refunded me, right?",
                    "edge_preorder",
                    ["pol_preorders", "pol_cancellation"],
                    ["does not cancel the order", "must read Cancelled"],
                    ["the order is already cancelled", "the refund was already issued"],
                    "No. A moved ship date does not cancel the order or create a refund. Because Cancel order is still showing, you can cancel it on the page. I have not cancelled it, and it is not cancelled until the status says Cancelled.",
                    "complex",
                    "large",
                    "case_pre",
                    trap="negation",
                    wrong_if_reused="Agreeing that the date change already cancelled and refunded the preorder invents both actions.",
                    attachments=full_docs([
                        "pol_preorders", "pol_cancellation", "pol_refunds", "pol_tracking",
                        "pol_payments", "pol_product", "pol_escalation", "pol_returns",
                    ]),
                )
            ],
        },
    ]
    built = []
    for cluster in edges:
        for spec in cluster["rows"]:
            spec = dict(spec)
            spec["cluster_id"] = cluster["id"]
            spec["context_size"] = spec.pop("context")
            spec["confusable_local"] = spec.pop("confusable")
            spec["exact_of_local"] = None
            spec["semantic_of_local"] = None
            spec["local_index"] = 0
            spec["preserve_case"] = spec.get("case_id") is not None
            spec["expected_cache"] = "miss" if spec.get("confusable_ref") or spec.get("confusable_local") is not None else "cold"
            spec.update(routing(spec["difficulty"], spec["context_size"]))
            built.append(spec)
    return built


CASES = {
    "case_ret_10d": {
        "order_id": "NS-104482", "customer": "Avery Cole", "item": "Nova Wireless Headphones",
        "seller": "Northstar", "fulfilled_by": "Northstar", "order_status": "Delivered",
        "delivered_days_ago": 10, "final_sale": False, "international": False,
        "return_requested": False, "refund_status": "none",
    },
    "case_ret_40d": {
        "order_id": "NS-104499", "customer": "Avery Cole", "item": "Nova Wireless Headphones",
        "seller": "Northstar", "fulfilled_by": "Northstar", "order_status": "Delivered",
        "delivered_days_ago": 40, "final_sale": False, "international": False,
        "return_requested": False, "refund_status": "none",
    },
    "case_ret_final": {
        "order_id": "NS-104500", "customer": "Avery Cole", "item": "Nova Wireless Headphones",
        "seller": "Northstar", "fulfilled_by": "Northstar", "order_status": "Delivered",
        "delivered_days_ago": 5, "final_sale": True, "international": False,
        "return_requested": False, "refund_status": "none",
    },
    "case_ret_intl": {
        "order_id": "NS-104510", "customer": "Avery Cole", "item": "Nova Wireless Headphones",
        "seller": "Northstar", "fulfilled_by": "Northstar", "order_status": "Delivered",
        "delivered_days_ago": 10, "final_sale": False, "international": True,
        "destination": "Canada", "prepaid_return_label": False, "duty_amount_shown": False,
    },
    "case_ref_3d": {
        "order_id": "NS-33001", "customer": "Jordan Blake", "item": "Nova Wireless Headphones",
        "refund_status": "issued", "business_days_since_issued": 3, "status_changed_since_issue": False,
        "return_received": True, "payment_method": "card", "amount_posted_to_card": False,
    },
    "case_ref_12d": {
        "order_id": "NS-33012", "customer": "Jordan Blake", "item": "Nova Wireless Headphones",
        "refund_status": "issued", "business_days_since_issued": 12, "status_changed_since_issue": False,
        "return_received": True, "payment_method": "card", "amount_posted_to_card": False,
    },
    "case_ref_none": {
        "order_id": "NS-33020", "customer": "Jordan Blake", "item": "Nova Wireless Headphones",
        "refund_status": "none", "return_received": False, "return_shipped": False,
        "payment_method": "card",
    },
    "case_can_open": {
        "order_id": "NS-22011", "customer": "Sam Patel", "item": "Desk lamp",
        "order_status": "Processing", "cancel_control_visible": True, "carrier_has_parcel": False,
    },
    "case_can_shipped": {
        "order_id": "NS-22018", "customer": "Sam Patel", "item": "Desk lamp",
        "order_status": "Shipped", "cancel_control_visible": False, "carrier_has_parcel": True,
        "shipped_hours_ago": 5,
    },
    "case_can_partial": {
        "order_id": "NS-22019", "customer": "Sam Patel",
        "lines": "desk lamp shipped with no cancel control; mug still Processing with Cancel item visible",
        "order_status": "Partially shipped",
    },
    "case_trk_6h": {
        "order_id": "NS-44101", "customer": "Riley Nguyen", "item": "Cotton towel set",
        "handoff_hours_ago": 6, "scan_count": 0, "tracking_number": "1Z90844101",
        "tracking_status": "Label created, no carrier scan", "delivered": False,
    },
    "case_trk_30h": {
        "order_id": "NS-44109", "customer": "Riley Nguyen", "item": "Cotton towel set",
        "handoff_hours_ago": 30, "scan_count": 0, "tracking_number": "1Z90844109",
        "tracking_status": "No scan since handoff", "delivered": False,
    },
    "case_trk_none": {
        "order_id": "NS-44110", "customer": "Riley Nguyen", "item": "Cotton towel set",
        "handoff_hours_ago": 6, "scan_count": 0, "tracking_number_in_context": False,
        "tracking_status": "not included", "city": "not included", "delivered": False,
    },
    "case_trk_delivered": {
        "order_id": "NS-44112", "customer": "Riley Nguyen", "item": "Cotton towel set",
        "tracking_number": "1Z90844112", "tracking_status": "Delivered",
        "handoff_hours_ago": 6, "delivered_morning": True, "recipient_named": False,
    },
    "case_mis_1d": {
        "order_id": "NS-55201", "customer": "Quinn Alvarez", "item": "Bookshelf",
        "estimate": "yesterday", "days_past_estimate": 1, "tracking_status": "In transit",
        "delivered": False,
    },
    "case_mis_4d": {
        "order_id": "NS-55208", "customer": "Quinn Alvarez", "item": "Bookshelf",
        "estimate": "4 days ago", "days_past_estimate": 4, "tracking_status": "In transit",
        "delivered": False,
    },
    "case_mis_delivered": {
        "order_id": "NS-55215", "customer": "Quinn Alvarez", "item": "Bookshelf",
        "tracking_status": "Delivered", "estimate": "yesterday", "delivered_this_morning": True,
        "recipient_named": False,
    },
    "case_mis_seller": {
        "order_id": "NS-55220", "customer": "Quinn Alvarez", "item": "Blender",
        "seller": "Harbor Home", "fulfilled_by": "Harbor Home", "days_past_estimate": 1,
        "estimate": "yesterday", "days_since_seller_reply": 6, "seller_window_displayed": False,
        "tracking_status": "In transit",
    },
    "case_dmg_mug": {
        "order_id": "NS-66101", "customer": "Morgan Lee", "item": "Stoneware mug",
        "condition": "cracked on arrival", "delivered_days_ago": 10, "final_sale": False,
        "seller": "Northstar", "replacement_approved": False,
    },
    "case_dmg_ok": {
        "order_id": "NS-66108", "customer": "Morgan Lee", "item": "Stoneware mug",
        "condition": "intact", "reason": "changed mind", "delivered_days_ago": 10,
        "final_sale": False, "seller": "Northstar",
    },
    "case_dmg_seller": {
        "order_id": "NS-66114", "customer": "Morgan Lee", "item": "Stoneware mug",
        "condition": "cracked on arrival", "final_sale": True, "seller": "Harbor Home",
        "fulfilled_by": "Harbor Home", "refund_approved": False,
    },
    "case_promo_62": {
        "order_id": "NS-77001", "customer": "Casey Brooks", "code": "SAVE10",
        "item": "cotton towel", "cart_total_usd": 62, "minimum_usd": 50,
        "electronics_excluded": True, "item_is_electronics": False, "order_placed": False,
        "discount_visible_in_summary": False, "stacks_with_membership": False,
    },
    "case_promo_42": {
        "order_id": "NS-77002", "customer": "Casey Brooks", "code": "SAVE10",
        "item": "cotton towel", "cart_total_usd": 42, "minimum_usd": 50,
        "electronics_excluded": True, "item_is_electronics": False, "order_placed": False,
    },
    "case_promo_placed": {
        "order_id": "NS-77008", "customer": "Casey Brooks", "code": "SAVE10",
        "item": "cotton towel", "cart_total_usd": 62, "minimum_usd": 50,
        "order_placed": True, "placed": "yesterday", "retroactive_allowed": False,
        "discount_on_order": False,
    },
    "case_promo_phones": {
        "order_id": "NS-77014", "customer": "Casey Brooks", "code": "SAVE10",
        "item": "Nova Wireless Headphones", "cart_total_usd": 62, "minimum_usd": 50,
        "electronics_excluded": True, "item_is_electronics": True, "order_placed": False,
    },
    "case_pay_declined": {
        "order_id": "none", "customer": "Drew Shah", "payment_status": "declined",
        "order_created": False, "pending_authorization": False, "captured": False,
    },
    "case_pay_pending": {
        "order_id": "NS-88008", "customer": "Drew Shah",         "payment_status": "declined at checkout, then pending authorization",
        "checkout_message": "declined, pending authorization now showing", "captured": False,
    },
    "case_pay_both": {
        "order_id": "NS-88016", "customer": "Drew Shah", "checkout_message": "declined",
        "order_created": False,
        "statement_lines": "one pending authorization and one captured charge",
        "captured": True, "pending_authorization": True, "refund_issued": False,
    },
    "case_addr_open": {
        "order_id": "NS-99001", "customer": "Eden Ross", "order_status": "Processing",
        "edit_address_visible": True, "carrier_has_parcel": False, "address_change_confirmed": False,
    },
    "case_addr_carrier": {
        "order_id": "NS-99008", "customer": "Eden Ross", "order_status": "Shipped",
        "edit_address_visible": False, "carrier_has_parcel": True, "picked_up_hours_ago": 1,
        "address_change_confirmed": False,
    },
    "case_addr_gift": {
        "order_id": "NS-99012", "customer": "Eden Ross", "order_status": "Shipped",
        "gift": True, "edit_address_visible": False, "carrier_has_parcel": True,
        "recipient_address_in_context": True, "address_change_confirmed": False,
        "cancel_control_visible": False,
    },
    "case_sub_2h": {
        "order_id": "NS-10101", "customer": "Finn Ortiz", "plan": "Plus monthly",
        "fee_posted_hours_ago": 2, "benefits_used": False, "refund_control_visible": True,
        "first_time_monthly": True, "cancellation_confirmed": False, "refund_issued": False,
    },
    "case_sub_10d": {
        "order_id": "NS-10108", "customer": "Finn Ortiz", "plan": "Plus monthly",
        "fee_posted_days_ago": 10, "benefits_used": False, "refund_control_visible": True,
        "first_time_monthly": True, "refund_issued": False,
    },
    "case_sub_annual": {
        "order_id": "NS-10114", "customer": "Finn Ortiz", "plan": "Plus annual",
        "fee_posted": "yesterday", "benefits_used": True, "benefit_used": "free shipping",
        "refund_control_visible": False, "cancellation_confirmed": False,
    },
    "case_mkt_2d": {
        "order_id": "NS-10201", "customer": "Harper Diaz", "item": "Blender",
        "seller": "Harbor Home", "fulfilled_by": "Harbor Home", "days_without_seller_reply": 2,
        "seller_window_displayed": False, "return_approved": False,
    },
    "case_mkt_6d": {
        "order_id": "NS-10208", "customer": "Harper Diaz", "item": "Blender",
        "seller": "Harbor Home", "fulfilled_by": "Harbor Home", "days_without_seller_reply": 6,
        "seller_window_displayed": False, "return_approved": False,
    },
    "case_mkt_damaged": {
        "order_id": "NS-10214", "customer": "Harper Diaz", "item": "Blender",
        "seller": "Harbor Home", "fulfilled_by": "Harbor Home", "condition": "stopped on arrival, damaged",
        "days_without_seller_reply": 6, "seller_window_displayed": False, "refund_approved": False,
    },
    "case_gc_plain": {
        "order_id": "none", "customer": "Indigo Park", "code_pasted_by_customer": "ABCD-1234",
        "authoritative_balance_in_context": False, "code_status": "unknown",
    },
    "case_gc_used": {
        "order_id": "none", "customer": "Indigo Park", "code_pasted_by_customer": "ABCD-1234",
        "code_status": "appears used", "authoritative_balance_in_context": False,
        "restore_already_done": False,
    },
    "case_acct": {
        "order_id": "NS-11001", "customer": "Jules Hart", "issue": "order the customer did not place",
        "authenticated_account_flow": False, "reset_email_sent": False,
    },
    "case_prod": {
        "order_id": "none", "customer": "Jules Hart", "product": "Nova Wireless Headphones",
        "listed_price_usd": 79, "travel_case_in_context": False, "driver_size_in_context": False,
    },
    "case_price": {
        "order_id": "none", "customer": "Jules Hart", "product": "Arc Lamp",
        "current_price_in_context": False, "earlier_chat_price_usd": 29, "price_is_authoritative": False,
    },
    "case_war": {
        "order_id": "NS-12008", "customer": "Harper Diaz", "item": "Harbor blender",
        "seller": "Harbor Home", "age": "8 months", "return_window_open": False,
        "product_class": "small appliance", "warranty_term": "12 months from delivery unless the product page differs",
        "replacement_approved": False,
    },
    "case_pre": {
        "order_id": "NS-13002", "customer": "Sam Patel", "item": "Preorder speaker",
        "order_status": "Preorder, not shipped", "ship_date_change": "slipped by two weeks",
        "cancel_control_visible": True, "cancellation_confirmed": False, "refund_status": "none",
    },
}


def assign_ids(rows: list[dict]) -> list[dict]:
    cluster_rows: dict[tuple[str, int], dict] = {}
    for number, row in enumerate(rows, start=1):
        row["id"] = f"q{number:03d}"
        cluster_rows[(row["cluster_id"], row["local_index"])] = row
    for row in rows:
        if row.get("exact_of_local") is not None:
            row["exact_of"] = cluster_rows[(row["cluster_id"], row["exact_of_local"])]["id"]
        else:
            row["exact_of"] = None
        if row.get("semantic_of_local") is not None:
            row["semantic_of"] = cluster_rows[(row["cluster_id"], row["semantic_of_local"])]["id"]
        else:
            row["semantic_of"] = None
        ref = row.get("confusable_ref")
        if ref:
            row["confusable_with"] = cluster_rows[ref]["id"]
        elif row.get("confusable_local") is not None:
            row["confusable_with"] = cluster_rows[(row["cluster_id"], row["confusable_local"])]["id"]
        else:
            row["confusable_with"] = None
    return rows


def public_row(row: dict, context_text: str) -> dict:
    hidden = {
        "local_index", "exact_of_local", "semantic_of_local", "confusable_local", "confusable_ref",
    }
    out = {key: value for key, value in row.items() if key not in hidden}
    out["approx_context_tokens"] = approx_tokens(context_text)
    out["apply_window"] = out["approx_context_tokens"] >= 1500
    out["apply_compression"] = out["approx_context_tokens"] >= 1500
    if out["expected_cache"] == "miss" and out.get("confusable_with"):
        out["query_similarity_to_confusable"] = None
    return out


def validate(docs: dict, cases: dict, rows: list[dict]) -> None:
    errors = []
    if len(rows) != 150:
        errors.append(f"expected 150 queries, found {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate query ids")
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        if row.get("case_id") and row["case_id"] not in cases:
            errors.append(f"{row['id']} missing case {row['case_id']}")
        for attachment in row["attachments"]:
            doc = docs.get(attachment["doc_id"])
            if doc is None:
                errors.append(f"{row['id']} missing doc {attachment['doc_id']}")
                continue
            for section in attachment.get("sections") or []:
                if section not in doc["sections"]:
                    errors.append(f"{row['id']} missing section {attachment['doc_id']}.{section}")
            if not attachment.get("sections"):
                for gold in row["gold_doc_ids"]:
                    pass
        for gold in row["gold_doc_ids"]:
            if gold not in {item["doc_id"] for item in row["attachments"]}:
                errors.append(f"{row['id']} gold doc {gold} is not attached")
        blob = "\n".join(
            docs[gold]["sections"][name]
            for gold in row["gold_doc_ids"]
            if gold in docs
            for name in docs[gold]["sections"]
            if name != "repeated_reminders"
        ).lower()
        for fact in row["gold_facts"]:
            if fact.lower() not in blob:
                errors.append(f"{row['id']} fact not in gold docs: {fact}")
        if row["expected_cache"] == "exact_hit":
            src = by_id[row["exact_of"]]
            if row["query"] != src["query"] or row["case_id"] != src["case_id"]:
                errors.append(f"{row['id']} exact row is not identical")
            if row["attachments"] != src["attachments"] or row["reference_answer"] != src["reference_answer"]:
                errors.append(f"{row['id']} exact payload drifted")
            if ids.index(row["id"]) < ids.index(src["id"]):
                errors.append(f"{row['id']} appears before its anchor")
        if row["expected_cache"] == "semantic_hit":
            src = by_id[row["semantic_of"]]
            if row["query"] == src["query"]:
                errors.append(f"{row['id']} semantic query is identical")
            if row["answer_id"] != src["answer_id"] or row["reference_answer"] != src["reference_answer"]:
                errors.append(f"{row['id']} semantic answer drifted")
            if row["case_id"] != src["case_id"] or row["attachments"] != src["attachments"]:
                errors.append(f"{row['id']} semantic context drifted")
            if jaccard(row["query"], src["query"]) < 0.34:
                errors.append(f"{row['id']} semantic similarity only {jaccard(row['query'], src['query']):.2f}")
        if row["expected_cache"] == "miss" and row.get("confusable_with"):
            src = by_id[row["confusable_with"]]
            score = jaccard(row["query"], src["query"])
            row["query_similarity_to_confusable"] = round(score, 3)
            if row["reference_answer"] == src["reference_answer"]:
                errors.append(f"{row['id']} miss reuses the confusable answer")
            if score < 0.34:
                errors.append(f"{row['id']} trap similarity only {score:.2f} against {src['id']}")
        if row["context_size"] == "small" and row["approx_context_tokens"] > 900:
            errors.append(f"{row['id']} small context is {row['approx_context_tokens']} tokens")
        if row["context_size"] == "large" and row["approx_context_tokens"] < 2000:
            errors.append(f"{row['id']} large context is only {row['approx_context_tokens']} tokens")
        if row["context_size"] == "small" and (row["apply_window"] or row["apply_compression"]):
            errors.append(f"{row['id']} small context should skip window and compression")
        if row["context_size"] == "large" and not (row["apply_window"] and row["apply_compression"]):
            errors.append(f"{row['id']} large context should apply window and compression")
        if row["small_model_ok"] != (row["difficulty"] == "easy"):
            errors.append(f"{row['id']} routing does not match difficulty")
        if row["suggested_model"] != ("small" if row["small_model_ok"] else "large"):
            errors.append(f"{row['id']} suggested model mismatch")
    if errors:
        raise SystemExit("\n".join(errors[:40]))


def main() -> None:
    docs = load_corpus()
    cases = CASES
    rows = []
    for cluster in build_clusters():
        rows.extend(expand_cluster(cluster))
    rows.extend(build_edges())
    assign_ids(rows)
    public = []
    for row in rows:
        context = assemble(docs, cases, row)
        item = public_row(row, context)
        if item["expected_cache"] == "miss" and item.get("confusable_with"):
            src = next(other for other in rows if other["id"] == item["confusable_with"])
            item["query_similarity_to_confusable"] = round(jaccard(item["query"], src["query"]), 3)
        public.append(item)
    validate(docs, cases, public)

    (ROOT / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    with (ROOT / "queries.jsonl").open("w") as handle:
        for item in public:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def tally(key):
        return dict(Counter(item[key] for item in public))

    small_tokens = [item["approx_context_tokens"] for item in public if item["context_size"] == "small"]
    large_tokens = [item["approx_context_tokens"] for item in public if item["context_size"] == "large"]
    trap_scores = [
        item["query_similarity_to_confusable"]
        for item in public
        if item.get("query_similarity_to_confusable") is not None
    ]
    doc_stats = {
        doc_id: {
            "title": doc["title"],
            "approx_tokens": approx_tokens("\n".join(doc["sections"].values())),
            "sections": list(doc["sections"]),
        }
        for doc_id, doc in docs.items()
    }
    manifest = {
        "store": "Northstar Market",
        "query_count": len(public),
        "replay_order": "queries.jsonl top to bottom. exact_hit rows assume exact_of was already seen.",
        "counts": {
            "expected_cache": tally("expected_cache"),
            "variant": tally("variant"),
            "context_size": tally("context_size"),
            "difficulty": tally("difficulty"),
            "suggested_model": tally("suggested_model"),
        },
        "context_tokens": {
            "small_min": min(small_tokens),
            "small_max": max(small_tokens),
            "large_min": min(large_tokens),
            "large_max": max(large_tokens),
        },
        "trap_query_similarity": {
            "min": min(trap_scores),
            "median": sorted(trap_scores)[len(trap_scores) // 2],
            "max": max(trap_scores),
        },
        "labels": {
            "expected_cache": "cold populates the cache. exact_hit is byte-identical to exact_of. semantic_hit paraphrases semantic_of and keeps the same answer. miss is a near neighbor whose reference_answer must not be reused.",
            "apply_window": "False when attached context is under 1500 approximate tokens. Dropping or reranking is not worth it.",
            "apply_compression": "False on that same short context. True when the attached policies are long enough that wording, examples, and the repeated reminder can be shortened.",
            "small_model_ok": "True only for easy queries. Large easy queries are small-model safe after pruning. Complex queries stay on a larger model.",
            "preserve_case": "The case block is decision-critical and short. Do not drop it during window management.",
        },
        "documents": doc_stats,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "context_tokens": manifest["context_tokens"], "trap_query_similarity": manifest["trap_query_similarity"]}, indent=2))


if __name__ == "__main__":
    main()
