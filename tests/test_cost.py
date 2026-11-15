"""Token ledger, cost maths and token-budget truncation."""

from __future__ import annotations

from config import Price
from utils import CostLedger, format_cost, truncate_to_tokens


def test_ledger_accumulates_per_step():
    ledger = CostLedger(model="m", price=Price(1.0, 2.0))
    ledger.record("map", 100, 50)
    ledger.record("map", 100, 50)
    ledger.record("combine", 10, 5)

    assert ledger.calls == 3
    assert ledger.input_tokens == 210
    assert ledger.output_tokens == 105
    assert ledger.total_tokens == 315
    assert ledger.steps["map"].calls == 2


def test_cost_uses_per_million_pricing():
    ledger = CostLedger(model="m", price=Price(input_per_mtok=0.30, output_per_mtok=2.50))
    ledger.record("map", 1_000_000, 1_000_000)
    assert ledger.cost_usd == 2.80


def test_unknown_pricing_reports_none():
    ledger = CostLedger(model="mystery", price=None)
    ledger.record("map", 500, 500)
    assert ledger.cost_usd is None
    assert format_cost(ledger) == "unknown"
    assert "unknown" in ledger.summary()


def test_rows_end_with_total():
    ledger = CostLedger(model="m", price=Price(1.0, 1.0))
    ledger.record("map", 1, 2)
    ledger.record("combine", 3, 4)
    rows = ledger.rows()
    assert rows[-1]["step"] == "TOTAL"
    assert rows[-1]["total_tokens"] == 10


def test_truncate_to_tokens_respects_limit():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    counter = lambda value: len(value.split())  # noqa: E731 - tiny local stub
    trimmed = truncate_to_tokens(text, 4, counter)
    assert counter(trimmed) <= 4
    assert text.startswith(trimmed)
    assert truncate_to_tokens("short", 10, counter) == "short"
