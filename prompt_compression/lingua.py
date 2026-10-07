"""LLMLingua-2 word-level compression of the windowed context.

LLMLingua-2 labels each token keep/drop without looking at the query, so a section
compresses the same way for every query; results are memoized per (text, rate).
Digits and negation/qualifier words are always kept: they change answers.
The case block is never compressed.
"""

import re

from llmlingua import PromptCompressor

from window.manager import approx_tokens

MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
PROTECTED = ["not", "no", "never", "without", "only", "unless", "cannot", "except", "before", "after"]

_STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on", "at",
         "for", "and", "or", "it", "its", "this", "that", "as", "by", "with", "from", "can", "do"}


class Lingua:
    def __init__(self, device="cpu"):
        self.compressor = PromptCompressor(model_name=MODEL, use_llmlingua2=True, device_map=device)
        self._memo = {}

    def compress_text(self, text, rate):
        key = (text, rate)
        if key not in self._memo:
            out = self.compressor.compress_prompt(
                [text], rate=rate, force_tokens=PROTECTED, force_reserve_digit=True, drop_consecutive=True
            )
            self._memo[key] = out["compressed_prompt"]
        return self._memo[key]

    def compress(self, chunks, rate):
        out = []
        for c in chunks:
            header, body = c["text"].split("\n", 1)
            small = self.compress_text(body, rate)
            out.append({**c, "text": header + "\n" + small, "tokens": approx_tokens(small)})
        return out


def _words(text):
    return re.findall(r"[a-z0-9$]+", text.lower())


def fact_kept_loose(fact, text):
    """Fact's content words (stopwords removed; digits and negations kept) appear in order."""
    need = [w for w in _words(fact) if w not in _STOP] or _words(fact)
    it = iter(_words(text))
    return all(any(w == t for t in it) for w in need)
