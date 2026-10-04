# ShopNest eval set (hand-written, graded)

This folder is the **graded evaluation set**. The course requires it to be hand-written and domain-specific, with no
copy-pasted public benchmarks. Everything else in `eval/data/` is workload, not grading material: the replay trace
is built from Bitext and the KB.

## Who writes what

- Each of the 6 team members writes **10–15 rows** in their own file, `<name>.jsonl` (for example `prathmesh.jsonl`),
  with `"author": "<name>"`.
- `seed.jsonl` is **AI-written scaffolding** (`"author": "seed"`). It exists so the pipeline, the CI gate and the
  trace builder work before the team's rows land.
  - Don't present it as hand-written.
  - Replace it, or keep it clearly labelled as seed data in the report.
  - `eval/run_ab.py` and `eval/ci_gate.py` record which authors' rows they used.
- Copy `TEMPLATE.jsonl.example` to start. Loaders ignore files whose names start with `_` or `TEMPLATE`.

## Row format (one JSON object per line)

```json
{"id": "ret-101", "category": "returns", "query": "...", "reference": "...", "needs_context": true,
 "type": "answerable", "pair_id": null, "should_cache_hit": null, "author": "yourname"}
```

| Field | Rules |
|---|---|
| `id` | Unique across all files. Use a category prefix plus a number from your own range (seed uses 001–099; take 1xx, 2xx, … per person). |
| `category` | One of `order`, `shipping`, `returns`, `refund`, `payment`, `account`, `product`, `other`. |
| `query` | What a real customer would type. Typos and informal wording are welcome. |
| `reference` | The correct answer, 1–4 sentences, containing the facts a grader should check. |
| `needs_context` | `true` if answering needs the store-policy KB (`eval/data/kb/*.md`). The reference **must** then agree with the KB: quote its numbers and don't invent policy. |
| `type` | `answerable` (normal), `hard` (multi-step, comparison or calculation), or `trap_pair` (see below). |
| `pair_id` | Only for `trap_pair`: both rows of a pair share it, e.g. `"tp-yourname-01"`. |
| `should_cache_hit` | `false` for trap rows, otherwise `null`. |
| `author` | Your name. |
| `same_as` | Optional. The id of a KB question (`kbq-005`) or another eval row that asks **the same question**, so a cache hit between them counts as correct rather than as a false hit. Leave it out otherwise. |

## Trap pairs (the most valuable rows)

A trap pair is two rows that **look almost identical but need different answers**. A semantic cache that serves
one row's answer for the other row has produced a **false hit**. The CI gate fails on any trap false hit.

Good trap patterns:
- negation ("…after opening the seal" vs "…if I have not opened the seal")
- a different order number, amount or date (SN-48213 vs SN-48231)
- a different product (Nestra headphones vs Nestra mattress)
- refund vs exchange
- domestic vs international
- cancel vs track

Write both rows of a pair one after the other, with the same `pair_id`, and make sure both references are correct
and actually differ.

## Hard items

Aim for 2–3 per person:
- multi-step ("cracked phone arrived yesterday — what exactly do I do?")
- comparison ("return or exchange for shoes that don't fit?")
- calculation ("₹1,500 change-of-mind return, not Plus: how much back?")

These are the rows the router must not downshift blindly.

## Checklist before committing

1. `./.venv/bin/python -c "from eval.build_trace import load_evalset; print(len(load_evalset()))"` loads without
   errors. The validator checks categories, types, unique ids and that each trap pair has exactly 2 rows.
2. Every `needs_context: true` reference agrees with `eval/data/kb/`.
3. No real customer data, no real people, and no copied benchmark questions.
4. Then rebuild the CI subset (`python -m eval.build_trace --ci-subset`) and update the CI baseline
   (`python -m eval.ci_gate --update-baseline`) in the same PR.
