"""
High-level abstraction tier for deriving strategic conclusions from relations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Tuple

from models.types import AbstractConclusion, Fact, Relation


ConclusionRule = Callable[[Tuple[Fact, ...], Tuple[Relation, ...]], Iterable[AbstractConclusion]]


def _collect_conclusions(
    rules: Sequence[ConclusionRule],
    facts: Tuple[Fact, ...],
    relations: Tuple[Relation, ...],
) -> Tuple[AbstractConclusion, ...]:
    """
    Execute conclusion rules and aggregate the resulting abstractions.
    """

    conclusions: list[AbstractConclusion] = []
    for rule in rules:
        conclusions.extend(rule(facts, relations))
    return tuple(conclusions)


@dataclass
class AbstractLayer:
    """
    Tier-3 component that transforms relations into legal, ethical, or policy views.

    Attributes:
        rules: Sequence of conclusion rules. Each rule receives both the facts and
               the relations to allow multi-hop reasoning before producing
               `AbstractConclusion` objects.
    """

    rules: Tuple[ConclusionRule, ...] = field(default_factory=tuple)

    def register_rule(self, rule: ConclusionRule) -> None:
        """
        Append a conclusion rule to the evaluation pipeline.
        """

        self.rules = (*self.rules, rule)

    def abstract(
        self,
        facts: Tuple[Fact, ...],
        relations: Tuple[Relation, ...],
    ) -> Tuple[AbstractConclusion, ...]:
        """
        Generate high-level conclusions from facts and relations.

        Args:
            facts: Immutable collection emitted by the sensory layer.
            relations: Immutable collection produced by the inference layer.

        Returns:
            Tuple containing high-level `AbstractConclusion` objects.
        """

        if not self.rules:
            return ()
        return _collect_conclusions(self.rules, facts, relations)


__all__ = ["AbstractLayer", "ConclusionRule"]

