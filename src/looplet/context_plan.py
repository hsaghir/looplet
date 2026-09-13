"""Advisory context planning for host-owned prompt budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ContextPlan"]


@dataclass(frozen=True, slots=True)
class ContextPlan:
    """Inspectable selection and budget decision before prompt rendering."""

    sections: tuple[str, ...]
    estimated_tokens: int
    budget_tokens: int | None = None
    sources: tuple[str, ...] = ()
    dropped_sections: tuple[str, ...] = ()

    @property
    def within_budget(self) -> bool:
        return self.budget_tokens is None or self.estimated_tokens <= self.budget_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": list(self.sections),
            "estimated_tokens": self.estimated_tokens,
            "budget_tokens": self.budget_tokens,
            "sources": list(self.sources),
            "dropped_sections": list(self.dropped_sections),
            "within_budget": self.within_budget,
        }
