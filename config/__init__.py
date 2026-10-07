"""Load config/costguard.toml and the prompt files in config/prompts/."""

import hashlib
import json
import tomllib
from pathlib import Path

DIR = Path(__file__).resolve().parent
CFG = tomllib.loads((DIR / "costguard.toml").read_text())


def prompt(name):
    return (DIR / "prompts" / f"{name}.txt").read_text()


def price(model):
    return tuple(CFG["prices"][model])


def config_version():
    """Short hash of what an answer depends on: the answer prompt and the answer models."""
    payload = {"answer_system": prompt("answer_system"), "models": [CFG["models"]["small"], CFG["models"]["large"]],
               "max_tokens": CFG["llm"]["max_tokens"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
