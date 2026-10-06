"""LLM judge for semantic cache candidates: does the cached answer fit the new query?

Runs only on candidates that already passed cosine, context, numbers and negation.
Reads ANTHROPIC_API_KEY from the environment. Verdicts are memoized on disk so
re-running the replay doesn't re-bill the same pairs.
"""

import json
import os
from pathlib import Path

import anthropic

from config import CFG, prompt as load_prompt

MODEL = CFG["models"]["cache_judge"]
SYSTEM = load_prompt("cache_judge_system")


class Judge:
    def __init__(self, verdicts_path):
        self.client = None if os.environ.get("COSTGUARD_OFFLINE") == "1" else anthropic.Anthropic()
        self.path = Path(verdicts_path)
        self.verdicts = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.calls = 0  # API calls made this run (memoized verdicts don't count)
        self.input_tokens = 0
        self.output_tokens = 0
        self.used = []  # every verdict consulted this run, memoized or not, for cost accounting

    def _ask(self, cached, new):
        if self.client is None:
            raise RuntimeError(f"offline and no memoized judge verdict for: {cached!r} vs {new!r}")
        msg = self.client.messages.create(
            model=MODEL,
            max_tokens=5,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Message A:\n{cached}\n\nMessage B:\n{new}"}],
        )
        self.calls += 1
        self.input_tokens += msg.usage.input_tokens
        self.output_tokens += msg.usage.output_tokens
        text = "".join(b.text for b in msg.content if b.type == "text").strip().upper()
        return {"verdict": "SAME" if text.startswith("SAME") else "DIFFERENT", "raw": text,
                "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}

    def same(self, new_query, cached_query):
        key = json.dumps([cached_query, new_query])
        if key not in self.verdicts:
            self.verdicts[key] = self._ask(cached_query, new_query)
            self.path.write_text(json.dumps(self.verdicts, indent=1))
        self.used.append(self.verdicts[key])
        return self.verdicts[key]["verdict"] == "SAME"
