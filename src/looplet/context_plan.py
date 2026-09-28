"""Advisory context planning for host-owned prompt budgets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Literal

from looplet.memory import PersistentMemorySource
from looplet.scaffolding import estimate_prompt_tokens

__all__ = ["ContextPlan", "ContextSourceDecision", "ContextSourceSelector", "ScopedContextSource"]


@dataclass(frozen=True, slots=True)
class ScopedContextSource:
    """Host-owned source with a per-turn scope and explicit provenance."""

    source_id: str
    source: PersistentMemorySource
    scope: Callable[..., bool] | None = None
    origin: str = ""
    trust: str = "untrusted"
    retention: Literal["turn", "run"] = "turn"


@dataclass(frozen=True, slots=True)
class ContextSourceDecision:
    """A source selection decision without the source's raw content."""

    source_id: str
    origin: str
    trust: str
    retention: str
    estimated_tokens: int
    included: bool
    reason: str
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "origin": self.origin,
            "trust": self.trust,
            "retention": self.retention,
            "estimated_tokens": self.estimated_tokens,
            "included": self.included,
            "reason": self.reason,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class ContextPlan:
    """Inspectable selection and budget decision before prompt rendering."""

    sections: tuple[str, ...]
    estimated_tokens: int
    budget_tokens: int | None = None
    sources: tuple[str, ...] = ()
    dropped_sections: tuple[str, ...] = ()
    source_decisions: tuple[ContextSourceDecision, ...] = ()

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
            "source_decisions": [decision.to_dict() for decision in self.source_decisions],
            "within_budget": self.within_budget,
        }


class ContextSourceSelector:
    """Select scoped sources within a budget, retaining content for one run only."""

    def __init__(
        self, sources: list[ScopedContextSource], *, budget_tokens: int | None = None
    ) -> None:
        if budget_tokens is not None and budget_tokens < 0:
            raise ValueError("Scoped context budget must be non-negative")
        self._sources: dict[str, ScopedContextSource] = {}
        self._retained: dict[str, str] = {}
        self._budget_tokens = budget_tokens
        for source in sources:
            if not source.source_id or not source.trust:
                raise ValueError("Scoped context sources require an ID and trust label")
            if source.retention not in ("turn", "run"):
                raise ValueError(f"Unknown context retention: {source.retention}")
            existing = self._sources.get(source.source_id)
            if existing is not None and existing != source:
                raise ValueError(f"Conflicting context source ID: {source.source_id}")
            self._sources[source.source_id] = source

    def select(self, *, task: Any, state: Any, step_num: int) -> tuple[str, ContextPlan]:
        """Evaluate scopes before loading, then fit eligible sources in declared order."""
        chunks: list[str] = []
        included_ids: list[str] = []
        decisions: list[ContextSourceDecision] = []
        for source in self._sources.values():
            if source.source_id in self._retained:
                text = self._retained[source.source_id]
            elif source.scope is not None and not source.scope(
                task=task, state=state, step_num=step_num
            ):
                decisions.append(self._decision(source, 0, False, "out_of_scope"))
                continue
            else:
                loaded = source.source.load(state)
                text = str(loaded).strip() if loaded is not None else ""
                if not text:
                    decisions.append(self._decision(source, 0, False, "empty"))
                    continue

            chunk = (
                f"[source={source.source_id!r} origin={source.origin!r} trust={source.trust!r}]\n"
                f"{text}"
            )
            tokens = estimate_prompt_tokens(chunk)
            candidate = "\n\n".join((*chunks, chunk))
            if (
                self._budget_tokens is not None
                and estimate_prompt_tokens(candidate) > self._budget_tokens
            ):
                decisions.append(self._decision(source, tokens, False, "budget", text))
                continue
            if source.retention == "run":
                self._retained[source.source_id] = text
            chunks.append(chunk)
            included_ids.append(source.source_id)
            decisions.append(self._decision(source, tokens, True, "included", text))

        rendered = "\n\n".join(chunks)
        plan = ContextPlan(
            sections=("scoped_context",) if chunks else (),
            estimated_tokens=estimate_prompt_tokens(rendered) if rendered else 0,
            budget_tokens=self._budget_tokens,
            sources=tuple(included_ids),
            dropped_sections=("scoped_context",)
            if any(decision.reason == "budget" for decision in decisions)
            else (),
            source_decisions=tuple(decisions),
        )
        return rendered, plan

    @staticmethod
    def _decision(
        source: ScopedContextSource, tokens: int, included: bool, reason: str, text: str = ""
    ) -> ContextSourceDecision:
        return ContextSourceDecision(
            source_id=source.source_id,
            origin=source.origin,
            trust=source.trust,
            retention=source.retention,
            estimated_tokens=tokens,
            included=included,
            reason=reason,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else "",
        )
