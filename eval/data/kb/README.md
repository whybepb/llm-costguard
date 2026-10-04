# ShopNest store-policy knowledge base

**Fictional company; written for this demo.** ShopNest, its policies, prices, phone numbers, e-mail addresses and the
"Nestra" house brand are invented for the LLM CostGuard capstone. Any resemblance to a real retailer is accidental.
The `.example` e-mail domain is reserved and never resolves.

What this folder is:

- System content for the RAG path: `eval/kb.py` chunks these files into ~150-250-token passages and retrieves the
  top-8 for a question, the way a typical "stuff 8 chunks" support bot would.
- **Not** the graded eval set. The hand-written, graded questions live in `eval/data/evalset/`.

How it was written:

- One fact sheet, so numbers agree across documents (30-day return window, 10 days for electronics, refund timelines
  per payment method, fees and thresholds).
- Deliberately messy in the way real help-centre exports are: repeated boilerplate footers ("For more information...",
  "We value your business..."), near-duplicate passages (refund timelines restated in several docs, the security
  warning repeated) and a couple of tables. That redundancy is what the context optimiser and compressor remove.

Bump `kb_version` in `configs/policy.yaml` whenever these files change, so the semantic cache gets a new partition.
