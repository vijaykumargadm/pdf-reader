"""Token accounting, cost estimation and small formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import Price


@dataclass
class StepUsage:
    """Tokens spent by one logical step (a chunk, the combine, the analysis...)."""

    label: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostLedger:
    """Running tally of every model call made during one document run."""

    model: str
    price: Price | None = None
    steps: dict[str, StepUsage] = field(default_factory=dict)

    def record(self, label: str, input_tokens: int, output_tokens: int) -> None:
        step = self.steps.setdefault(label, StepUsage(label=label))
        step.input_tokens += max(0, int(input_tokens or 0))
        step.output_tokens += max(0, int(output_tokens or 0))
        step.calls += 1

    @property
    def input_tokens(self) -> int:
        return sum(step.input_tokens for step in self.steps.values())

    @property
    def output_tokens(self) -> int:
        return sum(step.output_tokens for step in self.steps.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def calls(self) -> int:
        return sum(step.calls for step in self.steps.values())

    @property
    def cost_usd(self) -> float | None:
        """Estimated spend, or None when the model has no price entry."""
        if self.price is None:
            return None
        return (
            self.input_tokens / 1_000_000 * self.price.input_per_mtok
            + self.output_tokens / 1_000_000 * self.price.output_per_mtok
        )

    def rows(self) -> list[dict[str, object]]:
        """Flat records, ready for a DataFrame or CSV export."""
        rows: list[dict[str, object]] = []
        for step in self.steps.values():
            rows.append(
                {
                    "step": step.label,
                    "calls": step.calls,
                    "input_tokens": step.input_tokens,
                    "output_tokens": step.output_tokens,
                    "total_tokens": step.total_tokens,
                }
            )
        rows.append(
            {
                "step": "TOTAL",
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            }
        )
        return rows

    def summary(self) -> str:
        cost = "unknown (no price entry)" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        return (
            f"{self.calls} calls | {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out | est. {cost} | {self.model}"
        )


def format_cost(ledger: CostLedger) -> str:
    """Human-facing cost string used by the CLI and the UI."""
    return "unknown" if ledger.cost_usd is None else f"${ledger.cost_usd:.4f}"


def truncate_to_tokens(text: str, limit: int, counter) -> str:
    """Trim text to roughly ``limit`` tokens without cutting mid-word."""
    if counter(text) <= limit:
        return text
    # Binary search on characters is cheap and good enough for budgeting.
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if counter(text[:mid]) <= limit:
            low = mid
        else:
            high = mid - 1
    cut = text[:low]
    return cut[: cut.rfind(" ")] if " " in cut else cut
