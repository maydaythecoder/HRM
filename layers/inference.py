"""
Mid-level inference tier for deducing relationships between observed facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Tuple

from models.types import Fact, Relation


InferenceRule = Callable[[Tuple[Fact, ...]], Iterable[Relation]]


def _collect_relations(
    rules: Sequence[InferenceRule],
    facts: Tuple[Fact, ...],
) -> Tuple[Relation, ...]:
    """
    Execute inference rules and aggregate their outputs into a tuple.
    """

    relations: list[Relation] = []
    for rule in rules:
        relations.extend(rule(facts))
    return tuple(relations)


@dataclass
class InferenceLayer:
    """
    Tier-2 component that enriches facts with inferred causal relationships.

    Attributes:
        rules: Ordered collection of inference rules. Each rule is a callable that
               accepts the full set of facts and yields zero or more `Relation`
               objects. Rules can be chained to implement increasingly complex
               logic while keeping each component testable in isolation.
    """

    rules: Tuple[InferenceRule, ...] = field(default_factory=tuple)

    def register_rule(self, rule: InferenceRule) -> None:
        """
        Dynamically append a rule to the inference pipeline.
        """

        self.rules = (*self.rules, rule)

    def infer(self, facts: Tuple[Fact, ...]) -> Tuple[Relation, ...]:
        """
        Apply inference rules to the supplied facts.

        Args:
            facts: Immutable collection of facts produced by the sensory layer.

        Returns:
            Tuple containing inferred `Relation` objects.
        """

        if not self.rules:
            return ()
        return _collect_relations(self.rules, facts)


__all__ = ["InferenceLayer", "InferenceRule"]

