"""Build the held-out evaluation set: 40 Northstar support queries, never used for tuning.

Usage: .venv/bin/python -m eval.data.evalset.build_eval

Writes eval/data/evalset/queries.jsonl, cases.json and manifest.json in the
same schema as demo_data, so the e2e evaluation can run on it unchanged.

Mix (replay order matters: exact repeats, paraphrases and traps come after their source):
  cold      20  12 on topics the demo barely covers (pickup, preorders, price match,
                warranty, reviews, privacy, taxes, escalation, account, international,
                product, store credit), 8 on familiar topics with new cases
  exact      5  byte-identical repeats          -> should hit the exact cache
  semantic   6  paraphrases, same answer         -> should hit the semantic cache
  traps      9  near neighbours, different answer -> must miss (number, negation,
                intent, state, case, date boundary, source swap, duties vs tax)

Checks: every gold fact appears verbatim in a gold doc, every gold doc is attached,
exact/paraphrase/trap sources come earlier, and no query text repeats a demo query.
"""

import json
import random
from pathlib import Path

from demo_data.build_dataset import approx_tokens, assemble, load_corpus

ROOT = Path(__file__).resolve().parent
DEMO_QUERIES = ROOT.parents[2] / "demo_data" / "queries.jsonl"
SEED = 2026
LARGE_BUNDLE = 8
MIN_WINDOW_TOKENS = 1500

CASES = {
    "ev_pm_4d": {"order_id": "NS-40104", "customer": "Jordan Lee", "item": "Arc Lamp", "seller": "Northstar",
                 "fulfilled_by": "Northstar", "order_status": "Delivered", "delivered_days_ago": 4,
                 "price_match_terms_open": True, "return_requested": False},
    "ev_pm_12d": {"order_id": "NS-40112", "customer": "Jordan Lee", "item": "Arc Lamp", "seller": "Northstar",
                  "fulfilled_by": "Northstar", "order_status": "Delivered", "delivered_days_ago": 12,
                  "price_match_terms_open": True, "return_requested": False},
    "ev_pickup_ready": {"order_id": "NS-40201", "customer": "Sam Rivera", "item": "Arc Lamp",
                        "fulfillment": "Store pickup", "pickup_store": "Northstar Elm Street",
                        "pickup_status": "Ready for pickup", "hold_until": "2026-10-21 (printed on order)",
                        "pickup_code_in_context": False, "refund_status": "none"},
    "ev_pickup_12d": {"order_id": "NS-40202", "customer": "Sam Rivera", "item": "Arc Lamp",
                      "fulfillment": "Store pickup", "seller": "Northstar", "pickup_status": "Picked up",
                      "picked_up_days_ago": 12, "final_sale": False, "condition": "unused, original packaging"},
    "ev_ret_30d": {"order_id": "NS-40301", "customer": "Mia Chen", "item": "Juniper Rain Jacket", "seller": "Northstar",
                   "fulfilled_by": "Northstar", "order_status": "Delivered", "delivered_days_ago": 30,
                   "final_sale": False, "condition": "unused, tags attached"},
    "ev_ret_31d": {"order_id": "NS-40331", "customer": "Mia Chen", "item": "Juniper Rain Jacket", "seller": "Northstar",
                   "fulfilled_by": "Northstar", "order_status": "Delivered", "delivered_days_ago": 31,
                   "final_sale": False, "condition": "unused, tags attached"},
    "ev_ret_30d_final": {"order_id": "NS-40302", "customer": "Mia Chen", "item": "Juniper Rain Jacket",
                         "seller": "Northstar", "fulfilled_by": "Northstar", "order_status": "Delivered",
                         "delivered_days_ago": 30, "final_sale": True, "condition": "unused, tags attached"},
    "ev_war_5m": {"order_id": "NS-40401", "customer": "Priya Nair", "item": "Steelbrew Electric Kettle",
                  "seller": "Northstar", "product_class": "small appliance", "delivered_months_ago": 5,
                  "warranty_term": "not stated on product (general term applies)", "misuse_reported": False},
    "ev_war_14m": {"order_id": "NS-40414", "customer": "Priya Nair", "item": "Steelbrew Electric Kettle",
                   "seller": "Northstar", "product_class": "small appliance", "delivered_months_ago": 14,
                   "warranty_term": "not stated on product (general term applies)", "misuse_reported": False},
    "ev_ref_10bd": {"order_id": "NS-40501", "customer": "Leo Park", "item": "Juniper Rain Jacket",
                    "return_received": True, "refund_status": "Issued", "payment_method": "card",
                    "business_days_since_issued": 10, "status_changed_since_issue": False},
    "ev_rev_9d": {"order_id": "NS-40601", "customer": "Ana Silva", "item": "Ember Mug",
                  "review_posted_days_ago": 9, "review_stars": 1},
    "ev_tax": {"order_id": "NS-40701", "customer": "Omar Haddad", "order_placed_days_ago": 7,
               "sales_tax_charged": True, "tax_exemption_on_account": False, "international": False},
    "ev_tax_intl": {"order_id": "NS-40702", "customer": "Omar Haddad", "order_placed_days_ago": 7,
                    "international": True, "destination": "Canada", "duties_included_in_price": False,
                    "duty_amount_shown": False, "tax_exemption_on_account": False},
    "ev_preorder_shipped": {"order_id": "NS-40801", "customer": "Kai Morgan", "item": "Halo Smart Speaker",
                            "preorder": True, "ship_date_changes": 2, "order_status": "Shipped",
                            "shipped_days_ago": 1, "cancel_control_visible": False},
    "ev_customs": {"order_id": "NS-40901", "customer": "Elena Rossi", "item": "Ember Mug", "international": True,
                   "destination": "Canada", "tracking_status": "Held by customs", "days_in_customs": 3,
                   "duties_included_in_price": False, "duty_amount_shown": False},
    "ev_product": {"order_id": "NS-41001", "customer": "the signed-in customer", "product": "Arc Lamp",
                   "listed_price_usd": 45, "dimensions_in_context": False, "bulb_included_in_context": False},
    "ev_plus_order": {"order_id": "NS-30412", "customer": "Noah Kim", "plan": "Plus monthly",
                      "membership_status": "Active", "order_status": "Processing", "cancel_control_visible": True,
                      "cancellation_confirmed": False},
    "ev_addr_carrier": {"order_id": "NS-41201", "customer": "Grace Liu", "order_status": "Shipped",
                        "carrier_has_parcel": True, "carrier_pickup_hours_ago": 2, "edit_address_visible": False,
                        "address_change_confirmed": False},
    "ev_two_lines": {"order_id": "NS-41301", "customer": "Ivy Brooks", "order_created": True,
                     "statement_lines": "1 pending authorization and 1 captured charge, same amount",
                     "captured": True},
    "ev_mkt_dmg": {"order_id": "NS-41401", "customer": "Zoe Adams", "item": "Blender", "seller": "Harbor Home",
                   "fulfilled_by": "Harbor Home", "issue": "cracked jug", "photos_submitted": True,
                   "days_without_seller_reply": 5, "seller_window_displayed": False, "refund_approved": False},
    "ev_credit": {"order_id": "NS-41501", "customer": "Ben Torres", "refund_destination": "store credit",
                  "store_credit_balance_in_context": False},
    "ev_plus_annual": {"order_id": "NS-41601", "customer": "Ruth Ellis", "plan": "Plus annual",
                       "fee_posted_hours_ago": 20, "benefits_used": False, "refund_control_visible": False,
                       "first_time_monthly": False},
    "ev_privacy_pending": {"order_id": "NS-41701", "customer": "Lena Fischer",
                           "privacy_request_submitted_days_ago": 7, "privacy_request_result_in_context": False},
    "ev_escalated": {"order_id": "NS-41801", "customer": "Theo Grant", "escalated": True, "escalated_days_ago": 1,
                     "refund_status": "requested, not issued"},
}

ROWS = []


def _add(row):
    row["id"] = f"e{len(ROWS) + 1:03d}"
    ROWS.append(row)
    return row["id"]


def cold(key, cluster, query, case_id, gold, facts, must, reference, difficulty,
         sections=None, extra_docs=(), new_topic=False):
    """sections: {doc_id: [section, ...]} gives a small context of just those sections."""
    return _add({
        "key": key, "cluster_id": cluster, "variant": "cold", "new_topic": new_topic, "query": query,
        "case_id": case_id, "answer_id": key, "gold_doc_ids": gold, "gold_facts": facts,
        "must_not_claim": must, "reference_answer": reference, "difficulty": difficulty,
        "sections": sections, "extra_docs": list(extra_docs), "trap": None, "wrong_if_reused": None,
        "expected_cache": "cold", "exact_of": None, "semantic_of": None, "confusable_with": None,
    })


def _source(key):
    return next(r for r in ROWS if r["key"] == key)


def exact(src_key):
    s = _source(src_key)
    return _add({**{k: v for k, v in s.items() if k != "id"}, "key": s["key"] + "_x", "variant": "exact",
                 "expected_cache": "exact_hit", "exact_of": s["id"], "new_topic": s["new_topic"]})


def para(src_key, query):
    s = _source(src_key)
    return _add({**{k: v for k, v in s.items() if k != "id"}, "key": s["key"] + "_p", "variant": "semantic",
                 "query": query, "expected_cache": "semantic_hit", "semantic_of": s["id"]})


def trap(src_key, kind, query, answer_key, facts, must, reference, wrong_if_reused,
         case_id=None, gold=None):
    s = _source(src_key)
    return _add({**{k: v for k, v in s.items() if k != "id"}, "key": answer_key, "variant": f"tricky_{kind}",
                 "query": query, "case_id": case_id if case_id is not None else s["case_id"],
                 "answer_id": answer_key, "gold_doc_ids": gold or s["gold_doc_ids"], "gold_facts": facts,
                 "must_not_claim": must, "reference_answer": reference, "difficulty": "complex",
                 "trap": kind, "wrong_if_reused": wrong_if_reused, "expected_cache": "miss",
                 "confusable_with": s["id"]})


# ---------------------------------------------------------------------------------- rows
cold("pm_4d", "ev_price_match",
     "My Arc Lamp was delivered 4 days ago and I just saw the same lamp listed cheaper on Northstar. Can I get a price match?",
     "ev_pm_4d", ["pol_price_match"],
     ["within 7 days of delivery", "Approval is not automatic"],
     ["the price match is approved", "a specific refund amount"],
     "Yes, you can request one. Day 4 is inside the 7-day price-match window, which is counted from delivery, "
     "and price match is open. Submit it from the order's help flow. It has to be for an identical, in-stock "
     "Northstar offer. Approval is not automatic, so I can't promise the difference will be credited.",
     "easy", new_topic=True)

cold("pickup_missed", "ev_pickup",
     "I couldn't make it to the store yesterday to pick up my Arc Lamp. Can you just refund me now? And what's my pickup code?",
     "ev_pickup_ready", ["pol_pickup"],
     ["The hold time is the time printed on the order", "Quote a pickup code only when that code is in the order context"],
     ["a refund has been issued", "a pickup code", "a 3-day hold"],
     "Your order still shows Ready for pickup at Northstar Elm Street, held until 2026-10-21 as printed on the "
     "order. You can still collect it before then; missing one day didn't cancel it. I can't issue a refund "
     "from chat, and a missed pickup isn't an automatic refund. Check the hold status on the order for your "
     "options. Your pickup code isn't in the information I have. Use the code in your order confirmation, "
     "and bring the identification the store asks for.",
     "complex", new_topic=True)

cold("pw_reset", "ev_account",
     "I forgot my password. Can you send me a reset link?",
     None, ["pol_account"],
     ["account recovery flow on the sign-in page", "It cannot email a reset link on its own"],
     ["a reset email has been sent"],
     "I can't send a reset link from chat. Use the account recovery flow on the sign-in page to reset your "
     "password. Support will never ask for your current password or a one-time passcode.",
     "easy", sections={"pol_account": ["recovery"]}, new_topic=True)

cold("ret_30d", "ev_returns_boundary",
     "My Juniper Rain Jacket was delivered exactly 30 days ago. Is today too late to start a return?",
     "ev_ret_30d", ["pol_returns"],
     ["Day 30 is the last day to start a return"],
     ["the return window has closed", "the return is approved"],
     "No, today is the last day. Returns are counted in calendar days from delivery, and day 30 is the last day "
     "to start one. Since the jacket is unused with tags and isn't final sale, go to Orders, select it and "
     "choose Return item today. The return is only approved once the order page confirms it.",
     "easy")

cold("privacy_delete", "ev_privacy",
     "Please delete all of my personal data and confirm once it's done.",
     None, ["pol_privacy"],
     ["privacy request form linked from privacy settings", "Chat does not complete that request"],
     ["the data has been deleted"],
     "I can't delete your data or confirm a deletion from chat. Submit a deletion request through the privacy "
     "request form linked from your account's privacy settings. The result of that request is what confirms "
     "it.",
     "easy", sections={"pol_privacy": ["rules"]}, new_topic=True)

cold("war_5m", "ev_warranty",
     "My Steelbrew kettle stopped heating 5 months after delivery. What are my options?",
     "ev_war_5m", ["pol_warranty"],
     ["12 months from delivery", "Get help on the order"],
     ["a free replacement is guaranteed", "a standard return is still open"],
     "The 30-day return window has closed, so this isn't a standard return. The kettle is a small appliance "
     "with no different term listed, so the general manufacturer warranty of 12 months from delivery may "
     "cover it. Start a warranty claim with Get help on the order. The manufacturer may handle the service, "
     "and a repair or replacement isn't guaranteed.",
     "easy", new_topic=True)

exact("pm_4d")

cold("ref_10bd", "ev_refund_stale",
     "My refund was issued 10 business days ago and the status still hasn't changed. What should I do?",
     "ev_ref_10bd", ["pol_refunds"],
     ["contact support and include the order number"],
     ["the money has posted to the card", "keep waiting"],
     "Contact support now. Your refund has been marked issued for 10 business days without any status change, "
     "which is the point where it should be checked. Include your order number, NS-40501. I can't say the "
     "money has reached your card.",
     "easy")

cold("rev_9d", "ev_reviews",
     "Can I still edit the review I posted for my Ember Mug? I posted it 9 days ago.",
     "ev_rev_9d", ["pol_reviews"],
     ["edit their own review for 14 days after posting it"],
     ["the 14 days are a return window"],
     "Yes. You can edit your own review for 14 days after posting it, and you're on day 9. Open the review "
     "history in your account to change it.",
     "easy", sections={"pol_reviews": ["rules"]}, new_topic=True)

cold("tax_exempt", "ev_taxes",
     "I'm tax exempt. Can you remove the sales tax from the order I placed last week?",
     "ev_tax", ["pol_taxes"],
     ["An exemption has to be on the account through the exemption flow before purchase"],
     ["the sales tax will be refunded", "a recalculated tax amount"],
     "I can't remove it. An exemption has to be on your account through the exemption flow before purchase, "
     "and your account didn't have one when you placed this order. Stating the status in chat can't apply it "
     "afterwards. The tax charged on the order stands. Set up the exemption on your account for future orders.",
     "easy", extra_docs=["pol_international"], new_topic=True)

para("pw_reset", "I can't remember my password. Could you email me a link to reset it?")

cold("pickup_return", "ev_pickup",
     "I picked up my Arc Lamp at the store 12 days ago and haven't used it. Can I still return it?",
     "ev_pickup_12d", ["pol_pickup", "pol_returns"],
     ["counted from the pickup time shown on the order", "30 calendar days"],
     ["the return is approved", "the pickup hold is the return window"],
     "Yes. For a pickup order, the return window is counted from the pickup time on the order, and day 12 is "
     "inside the 30 calendar days. The lamp is unused, in its original packaging and not final sale, so "
     "start a return from Orders by choosing Return item. It's approved only when the order page confirms it.",
     "easy", new_topic=True)

trap("pm_4d", "number",
     "My Arc Lamp was delivered 12 days ago and I just saw the same lamp listed cheaper on Northstar. Can I get a price match?",
     "pm_12d", ["within 7 days of delivery", "is outside this 7-day window"],
     ["the price match can be submitted", "the 30-day return window applies to price match"],
     "No. A price-match request has to be submitted within 7 days of delivery, and day 12 is outside that "
     "window. The 30-day return window is separate and doesn't extend price match, so I can't promise any "
     "price adjustment.",
     "Reusing the day-4 answer would invite a price match outside the 7-day window.", case_id="ev_pm_12d")

cold("preorder_shipped", "ev_preorders",
     "My Halo Smart Speaker preorder finally shipped yesterday after the date slipped twice, and the cancel button is gone. Can I still cancel since the delays were your fault?",
     "ev_preorder_shipped", ["pol_preorders", "pol_cancellation"],
     ["After the item ships, the shipped rule in the cancellation article applies",
      "Cancellation is available only while a cancel control is shown"],
     ["the order has been cancelled", "the delays reopened cancellation"],
     "Not anymore. Cancellation is only available while a cancel control is shown, and it disappeared once "
     "the speaker shipped. The earlier date changes don't reopen it, and I can't cancel it from chat. Once "
     "it's delivered, you may be able to request a return under the returns policy if the speaker is "
     "eligible.",
     "complex", new_topic=True)

exact("ret_30d")

cold("escalated", "ev_escalation",
     "I escalated my refund case yesterday. Does that mean my refund is approved?",
     "ev_escalated", ["pol_escalation"],
     ["It does not mean the customer's requested outcome is granted"],
     ["the refund is approved", "the refund has been issued"],
     "No. Escalation hands your case over for review. It doesn't mean the refund is approved, and right now "
     "your refund is only requested, not issued. The order record has to show the refund before it's real.",
     "easy", sections={"pol_escalation": ["overview", "when"]}, new_topic=True)

cold("customs", "ev_international",
     "My order to Canada has been held in customs for 3 days. Is it lost, and do I have to pay duties?",
     "ev_customs", ["pol_international", "pol_delivery"],
     ["A customs hold is not, by itself, a lost parcel",
      "Duties and import taxes are usually the customer's responsibility"],
     ["the parcel is lost", "a specific duty amount", "the 48-hour rule is a customs deadline"],
     "A customs hold doesn't mean the parcel is lost, and the 48-hour late-package rule isn't a customs "
     "deadline. Keep watching the tracking and the order page. Duties and import taxes are usually your "
     "responsibility unless the order says the price included them, and yours doesn't. No duty amount is "
     "shown on the order, so I can't tell you how much they'll be.",
     "complex", new_topic=True)

trap("ret_30d", "number",
     "My Juniper Rain Jacket was delivered exactly 31 days ago. Is today too late to start a return?",
     "ret_31d", ["On day 31 the general window is closed"],
     ["today is the last day", "the return can still be started"],
     "Yes, unfortunately. Day 30 was the last day to start a return, and on day 31 the general window is "
     "closed. You can check the order page in case it lists an exception, but the standard return is no "
     "longer available.",
     "Reusing the day-30 answer would tell the customer a closed window is still open.", case_id="ev_ret_31d")

para("war_5m", "My Steelbrew kettle quit heating about 5 months after it arrived. What can I do?")

cold("product_specs", "ev_product",
     "What are the dimensions of the Arc Lamp, and does it come with a bulb?",
     "ev_product", ["pol_product"],
     ["Never invent specifications", "say the information is not available in the provided context"],
     ["a specific dimension", "a bulb is included"],
     "I don't have either detail. The information I have lists the Arc Lamp at $45 but includes no "
     "dimensions and doesn't say whether a bulb is included. Check the product page for both rather than "
     "relying on a guess.",
     "easy", new_topic=True)

exact("privacy_delete")

cold("plus_and_order", "ev_membership_vs_order",
     "Cancel my Plus membership. Will that also cancel my order NS-30412 that's still processing?",
     "ev_plus_order", ["pol_subscriptions", "pol_cancellation"],
     ["Account, then Memberships", "Cancelling a physical order is not cancelling Plus"],
     ["the membership is cancelled", "the order is cancelled"],
     "They're two separate things, and I can't cancel either from chat. To end Plus, go to Account, then "
     "Memberships; the terms shown there control when it ends. Cancelling Plus won't cancel order NS-30412. "
     "That order is still Processing and shows Cancel order, so use that control if you want to cancel it, "
     "and wait until its status says Cancelled.",
     "complex")

trap("rev_9d", "intent",
     "I posted a 1-star review of my Ember Mug 9 days ago. If I change it to 5 stars, can I get a discount?",
     "rev_discount", ["does not offer a discount for changing a star rating"],
     ["a discount for a rating change"],
     "No. Northstar doesn't offer a discount for changing a star rating. You can still edit your review for "
     "14 days after posting it, but it should describe your own experience.",
     "Reusing the plain edit answer would skip the refusal of a discount for a rating change.")

para("ref_10bd", "It's been 10 business days since my refund was issued and nothing has changed. What now?")

cold("addr_carrier", "ev_address",
     "The carrier picked up my package 2 hours ago but I typed the wrong apartment number. Can you fix it?",
     "ev_addr_carrier", ["pol_address"],
     ["Once the order enters fulfillment or a carrier has it, the edit is usually gone"],
     ["the address has been changed"],
     "I can't change it from chat. Your order no longer shows Edit address because the carrier already has "
     "the package, so the change is likely unavailable. Use the carrier or support options shown on the "
     "order to sort out the apartment number.",
     "easy")

trap("pm_4d", "source",
     "My Arc Lamp was delivered 4 days ago and I just saw the same lamp listed cheaper at another store. Can I get a price match?",
     "pm_competitor", ["for an identical in-stock Northstar offer"],
     ["a competitor price qualifies"],
     "Not for that offer. Price match only covers an identical, in-stock offer on Northstar, not another "
     "store's price. You're inside the 7-day window, so if the same lamp is cheaper on Northstar you can "
     "submit that from the order's help flow, but approval isn't automatic.",
     "Reusing the Northstar-listing answer would accept a competitor price.")

cold("two_lines", "ev_payments",
     "My bank shows two lines for the same order, one pending and one posted. Was I charged twice?",
     "ev_two_lines", ["pol_payments"],
     ["Two statement lines, one pending and one captured, are not automatically two charges"],
     ["you were charged twice", "both will be refunded"],
     "Not necessarily. A pending authorization next to a captured charge isn't automatically two charges. "
     "The captured charge is the payment for your order, and the pending authorization may drop off on its "
     "own. Don't pay again. If the pending line doesn't release, check with your bank.",
     "easy")

para("tax_exempt", "I have tax-exempt status. Could you take the sales tax off my order from last week?")

exact("pw_reset")

cold("mkt_damaged", "ev_marketplace_damage",
     "My Harbor Home blender arrived with a cracked jug. I sent photos through Get help 5 days ago and the seller still hasn't replied. The order shows no response window. What now?",
     "ev_mkt_dmg", ["pol_marketplace", "pol_damaged"],
     ["After 4 calendar days with no seller response, escalate on the order page",
      "Do not treat escalation as an approved refund"],
     ["the refund is approved", "a replacement is guaranteed", "throw the blender away"],
     "Escalate now using the option on the order page. With no window shown, the seller has 4 calendar days "
     "to respond, and it's been 5. Escalation asks for a review; it doesn't approve a refund or guarantee a "
     "replacement. Keep the blender and its packaging until the flow tells you what to do.",
     "complex")

trap("ref_10bd", "negation",
     "My refund was issued 10 business days ago and the status still hasn't changed. I shouldn't contact support yet, right?",
     "ref_10bd_neg", ["contact support and include the order number"],
     ["keep waiting", "it is too early for support"],
     "No, you should contact support now. The refund has been marked issued for 10 business days with no "
     "status change, which is the point to have it checked. Include your order number, NS-40501.",
     "Agreeing with the 'right?' would tell the customer to keep waiting past the stale line.")

cold("credit_convert", "ev_store_credit",
     "My return was refunded as store credit. Can you turn it into a gift card code so I can give it to my sister?",
     "ev_credit", ["pol_giftcards"],
     ["Chat cannot convert one into the other"],
     ["the credit has been converted", "a gift card code"],
     "I can't convert it. Store credit is a balance on your account, and checkout applies it from there. It "
     "isn't a gift card code, and chat can't turn one into the other. Your account page shows the current "
     "balance.",
     "easy", sections={"pol_giftcards": ["store_credit"]}, new_topic=True)

trap("war_5m", "date",
     "My Steelbrew kettle stopped heating 14 months after delivery. What are my options?",
     "war_14m", ["12 months from delivery"],
     ["the warranty covers it", "a standard return is open"],
     "Unfortunately, the general manufacturer warranty for small appliances is 12 months from delivery, and "
     "14 months is past it. The kettle's product page doesn't list a longer term, and the 30-day return "
     "window has long closed. I can't promise any coverage. Check the product and order page in case a "
     "different warranty term is listed.",
     "Reusing the 5-month answer would point to a warranty that has expired.", case_id="ev_war_14m")

para("escalated", "I escalated my refund yesterday. Does that mean it got approved?")

cold("plus_annual", "ev_membership_fee",
     "I paid the annual Plus fee yesterday and haven't used any benefits. Can I get it refunded?",
     "ev_plus_annual", ["pol_subscriptions"],
     ["An annual fee is outside that monthly exception"],
     ["the fee will be refunded", "the refund has been issued"],
     "I can't promise that. The only fee-refund exception is for a first-time monthly fee posted less than "
     "48 hours ago with no benefits used and the refund control showing, and an annual fee is outside it. "
     "Your membership page doesn't show a refund control. Check the terms under Account, then Memberships.",
     "complex")

trap("privacy_delete", "state",
     "I submitted the privacy request form last week. Has my data been deleted yet?",
     "privacy_pending", ["do not confirm deletion"],
     ["the data has been deleted"],
     "I can't confirm that. There's no result for your privacy request in what I can see. The request's own "
     "result, reached from your privacy settings, is what confirms a deletion.",
     "Reusing the 'how to request deletion' answer ignores that the request was already submitted.",
     case_id="ev_privacy_pending")

para("pm_4d", "The same Arc Lamp I got 4 days ago is now cheaper on Northstar. Can I ask for a price match?")

trap("ret_30d", "case",
     "My Juniper Rain Jacket was delivered exactly 30 days ago and was marked final sale. Is today too late to start a return?",
     "ret_30d_final", ["stays non-returnable even inside the 30 calendar days"],
     ["today is the last day to return it", "the return can be started"],
     "The timing isn't the problem: a final-sale item stays non-returnable even inside the 30 calendar days, "
     "so a change-of-mind return isn't available. If it arrived damaged or incorrect, you can use Get help on "
     "the order instead.",
     "Reusing the day-30 answer would invite a return of a final-sale item.", case_id="ev_ret_30d_final")

exact("ref_10bd")

trap("tax_exempt", "intent",
     "I'm tax exempt. Can you refund the import duties on my order to Canada from last week?",
     "duties_not_tax", ["Duties on international orders are not sales tax",
                        "Duties and import taxes are usually the customer's responsibility"],
     ["the duties will be refunded", "a tax exemption covers duties"],
     "No. Import duties aren't sales tax, so a sales-tax exemption doesn't cover them. Duties are usually "
     "your responsibility unless the order says the price included them, and yours doesn't. I can't refund "
     "them or quote an amount, since none is shown on the order.",
     "Reusing the sales-tax answer would treat duties as sales tax.",
     case_id="ev_tax_intl", gold=["pol_taxes", "pol_international"])


# ---------------------------------------------------------------------------------- build
def main():
    docs = load_corpus()
    rng = random.Random(SEED)
    demo_queries = {json.loads(line)["query"].strip().lower() for line in DEMO_QUERIES.open()}
    bundles = {}  # answer_id of a cold row -> attachments, reused by its repeats, paraphrases and traps
    out = []
    for r in ROWS:
        src = r["exact_of"] or r["semantic_of"] or r["confusable_with"]
        if src:
            base = next(o for o in out if o["id"] == src)
            attachments = base["attachments"]
        elif r["sections"]:
            attachments = [{"doc_id": d, "sections": s} for d, s in r["sections"].items()]
        else:
            required = list(dict.fromkeys(r["gold_doc_ids"] + r["extra_docs"]))
            others = [d for d in docs if d not in required]
            attachments = [{"doc_id": d} for d in required + rng.sample(others, LARGE_BUNDLE - len(required))]
            rng.shuffle(attachments)
        row = {k: v for k, v in r.items() if k not in ("key", "sections", "extra_docs")}
        row["attachments"] = attachments
        tokens = approx_tokens(assemble(docs, CASES, row))
        row["approx_context_tokens"] = tokens
        row["context_size"] = "large" if tokens >= MIN_WINDOW_TOKENS else "small"
        row["apply_window"] = tokens >= MIN_WINDOW_TOKENS
        row["small_model_ok"] = row["difficulty"] == "easy"
        row["suggested_model"] = "small" if row["small_model_ok"] else "large"
        out.append(row)

    # --- validation ---
    ids = [r["id"] for r in out]
    for i, r in enumerate(out):
        att = {a["doc_id"] for a in r["attachments"]}
        assert set(r["gold_doc_ids"]) <= att, (r["id"], "gold doc not attached")
        gold_text = " ".join(t for d in r["gold_doc_ids"] for t in docs[d]["sections"].values())
        for f in r["gold_facts"]:
            assert f in gold_text, (r["id"], "gold fact not verbatim in gold docs", f)
        for ref in (r["exact_of"], r["semantic_of"], r["confusable_with"]):
            assert ref is None or ids.index(ref) < i, (r["id"], "source must come earlier")
        assert r["query"].strip().lower() not in demo_queries or r["exact_of"], (r["id"], "repeats a demo query")
        assert r["case_id"] is None or r["case_id"] in CASES, (r["id"], "unknown case")
        if r["exact_of"]:
            s = out[ids.index(r["exact_of"])]
            assert (s["query"], s["case_id"], s["attachments"]) == (r["query"], r["case_id"], r["attachments"])

    (ROOT / "queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    (ROOT / "cases.json").write_text(json.dumps(CASES, indent=2) + "\n")
    count = lambda f: dict(sorted({v: sum(1 for r in out if r[f] == v) for v in {r[f] for r in out}}.items()))
    manifest = {
        "store": "Northstar Market", "purpose": "held-out evaluation; never used for tuning",
        "query_count": len(out), "replay_order": "queries.jsonl top to bottom",
        "counts": {f: count(f) for f in ("expected_cache", "variant", "context_size", "difficulty")},
        "new_topic_cold": sum(1 for r in out if r["variant"] == "cold" and r["new_topic"]),
        "multi_doc_gold": sum(1 for r in out if len(r["gold_doc_ids"]) > 1),
        "gold_doc_first_in_bundle": sum(1 for r in out if r["attachments"][0]["doc_id"] in r["gold_doc_ids"]),
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
