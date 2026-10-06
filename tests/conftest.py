"""Keep tests away from real spend: the Anthropic adapter's spend ledger goes to a temp file in every test."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_spend_ledger(tmp_path, monkeypatch):
    import costguard.providers.anthropic_provider as ap
    monkeypatch.setenv("COSTGUARD_SPEND_LEDGER", str(tmp_path / "spend_ledger.jsonl"))
    monkeypatch.setattr(ap, "_LEDGERS", {})
