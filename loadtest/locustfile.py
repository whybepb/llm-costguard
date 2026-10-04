"""Locust load profile for the CostGuard proxy (run it via loadtest/run.sh, which starts a mock-upstream proxy).

Traffic mix, modelled on the ShopNest support workload:
  repeat      40%  the canonical wording of a popular question       -> exact-cache hits after the first
  paraphrase  30%  a reworded variant of a popular question           -> semantic-cache candidates
  context     20%  a store-policy question carrying 4-6 KB documents  -> context trimming + compression path
                   (half of them send no_cache=true so rerank/compression run under load instead of being cached)
  novel       10%  a question with a fresh order number               -> always a miss (and an entity-guard probe)

Each request records the proxy's own `x-costguard-overhead-ms` and `x-costguard-cache` headers. At the end the
samples are written to $COSTGUARD_LOADTEST_SAMPLES (JSON) for run.sh to summarise.
"""
from __future__ import annotations

import json
import os
import random

from locust import HttpUser, between, events, task

MODE = os.environ.get("COSTGUARD_LOADTEST_MODE", "balanced")
WAIT_MIN = float(os.environ.get("COSTGUARD_LOADTEST_WAIT_MIN", "0.1"))
WAIT_MAX = float(os.environ.get("COSTGUARD_LOADTEST_WAIT_MAX", "0.5"))
CONTEXT_NO_CACHE = float(os.environ.get("COSTGUARD_LOADTEST_CONTEXT_NO_CACHE", "0.5"))

# (category, canonical wording, paraphrases)
CLUSTERS: list[tuple[str, str, list[str]]] = [
    ("order", "Where is my order?", ["Can you tell me where my order is?", "Where's my package right now?",
                                     "I want to know the status of my order"]),
    ("shipping", "How long does standard shipping take?", ["What is the delivery time for standard shipping?",
                                                           "How many days does regular delivery take?",
                                                           "When will standard shipping arrive?"]),
    ("returns", "How do I return a jacket?", ["What's the process to send back a jacket?",
                                              "I need to return a coat I bought, how?",
                                              "Can you explain how to return clothing?"]),
    ("refund", "When will I get my refund?", ["How long until my refund arrives?",
                                              "When does the refund hit my account?",
                                              "What is the refund processing time?"]),
    ("payment", "Which payment methods do you accept?", ["What ways can I pay?", "Do you take PayPal or cards?",
                                                         "What payment options are available?"]),
    ("account", "How do I reset my password?", ["I forgot my password, how do I change it?",
                                                "Can you help me reset my account password?",
                                                "How can I recover my login password?"]),
    ("product", "Is the X200 blender dishwasher safe?", ["Can I put the X200 blender in the dishwasher?",
                                                         "Is the X200 blender jar safe for dishwashers?",
                                                         "Dishwasher safe - X200 blender?"]),
    ("order", "Can I cancel my order?", ["How do I cancel an order I just placed?", "Is it possible to cancel my order?",
                                         "I want to cancel my purchase"]),
    ("shipping", "Do you ship internationally?", ["Can you deliver outside the country?",
                                                  "Do you offer international shipping?",
                                                  "Will you ship my order abroad?"]),
    ("other", "What are your customer service hours?", ["When is support available?",
                                                        "What time can I reach customer service?",
                                                        "What hours is your help desk open?"]),
]
# Zipf-like popularity: a few questions dominate, as in real support traffic.
WEIGHTS = [1.0 / (i + 1) for i in range(len(CLUSTERS))]

KB_DOCS = [
    "Returns policy: Most items can be returned within 30 days of delivery for a full refund to the original payment "
    "method. Items must be unused and in their original packaging with all tags attached. Electronics that have been "
    "opened can be returned within 15 days and may be subject to a 10% restocking fee unless they are defective. "
    "Final-sale items, gift cards and personalised products cannot be returned.",
    "Refund timing: Once a return reaches our warehouse it is inspected within 2 business days. Approved refunds are "
    "issued to the original payment method; card refunds take 5 to 7 business days to appear, PayPal refunds 3 to 5 "
    "business days, and store credit is available immediately. Shipping fees are refunded only if the item was "
    "damaged, defective or sent in error.",
    "Shipping options: Standard shipping takes 4 to 6 business days and is free on orders over $50. Express shipping "
    "takes 2 business days and costs $12.99. Next-day delivery is available in selected cities for $24.99 when the "
    "order is placed before 1 pm local time. Orders are processed Monday to Friday, excluding public holidays.",
    "International shipping: We ship to 38 countries. International delivery takes 7 to 14 business days. Import "
    "duties and taxes are calculated at checkout for most destinations; where they are not, the recipient pays them "
    "on delivery. Large furniture and items containing lithium batteries cannot be shipped internationally.",
    "Warranty: Electronics carry a 1-year limited manufacturer warranty covering defects in materials and "
    "workmanship. Small kitchen appliances such as blenders carry a 2-year warranty. Accidental damage, normal wear "
    "and tear and unauthorised repairs are not covered. To start a claim, contact support with your order number and "
    "a photo or video of the problem.",
    "Exchanges: Apparel and footwear can be exchanged for a different size or colour within 30 days at no cost. "
    "Start an exchange from the Orders page in your account. We ship the replacement as soon as the original item is "
    "scanned by the carrier. Exchanges for a different product are processed as a return plus a new order.",
    "Order changes: Orders can be cancelled or edited within 1 hour of being placed from the Orders page. After that "
    "the order enters fulfilment and can no longer be changed; you can refuse the delivery or return the items "
    "instead. Address changes after dispatch must be requested directly with the carrier.",
    "Payment methods: We accept Visa, Mastercard, American Express, PayPal, Apple Pay and ShopNest gift cards. "
    "Buy-now-pay-later is available on orders between $50 and $1,500. We never store full card numbers. Payments "
    "that fail fraud checks are cancelled automatically and the hold is released within 3 business days.",
]
KB_QUESTIONS = [
    ("returns", "Can I return opened headphones and is there a fee?"),
    ("refund", "How long does a PayPal refund take after you receive my return?"),
    ("shipping", "How much is express shipping and how fast is it?"),
    ("shipping", "Can you ship a blender to Germany and who pays the import duties?"),
    ("product", "Does the warranty on my blender cover accidental damage?"),
    ("order", "Can I still change the address on an order I placed this morning?"),
    ("payment", "Can I use buy now pay later for a $40 order?"),
]

SAMPLES: list[list] = []   # [kind, cache_status, overhead_ms_header, client_ms, status_code]


class CostGuardUser(HttpUser):
    wait_time = between(WAIT_MIN, WAIT_MAX)

    def _ask(self, kind: str, query: str, category: str, context: list[str] | None = None,
             no_cache: bool = False) -> None:
        body = {"model": "strong", "temperature": 0, "messages": [{"role": "user", "content": query}],
                "costguard": {"mode": MODE, "category": category, "context": context or [], "no_cache": no_cache}}
        with self.client.post("/v1/chat/completions", json=body, name=f"chat:{kind}", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif not r.headers.get("x-costguard-request-id"):
                r.failure("missing x-costguard headers")
            else:
                r.success()

    def _cluster(self) -> tuple[str, str, list[str]]:
        return random.choices(CLUSTERS, weights=WEIGHTS, k=1)[0]

    @task(4)
    def repeat(self) -> None:
        cat, canonical, _ = self._cluster()
        self._ask("repeat", canonical, cat)

    @task(3)
    def paraphrase(self) -> None:
        cat, _, paras = self._cluster()
        self._ask("paraphrase", random.choice(paras), cat)

    @task(2)
    def with_context(self) -> None:
        cat, q = random.choice(KB_QUESTIONS)
        docs = random.sample(KB_DOCS, k=random.randint(4, 6))   # deliberately generous retrieval
        self._ask("context", q, cat, docs, no_cache=random.random() < CONTEXT_NO_CACHE)

    @task(1)
    def novel(self) -> None:
        n = random.randint(10000, 99999)
        q = random.choice(["Where is order #{n}?", "Has order {n} shipped yet?", "Can you cancel order #{n}?"])
        self._ask("novel", q.format(n=n), "order")


@events.request.add_listener
def _record(request_type, name, response_time, response_length, response=None, exception=None, **kw) -> None:
    kind = name.split(":", 1)[-1]
    status = getattr(response, "status_code", 0) if response is not None else 0
    headers = getattr(response, "headers", None) or {}
    try:
        overhead = float(headers.get("x-costguard-overhead-ms")) if headers.get("x-costguard-overhead-ms") else None
    except (TypeError, ValueError):
        overhead = None
    SAMPLES.append([kind, headers.get("x-costguard-cache"), overhead, float(response_time), status])


@events.quitting.add_listener
def _dump(environment, **kw) -> None:
    out = os.environ.get("COSTGUARD_LOADTEST_SAMPLES")
    if out:
        with open(out, "w") as f:
            json.dump({"fields": ["kind", "cache", "overhead_ms", "client_ms", "status"], "samples": SAMPLES}, f)
