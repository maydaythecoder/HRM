"""
Typed containers representing the information exchanged between reasoning tiers.

These structures enable loose coupling between layers while preserving clear
contracts for what each tier consumes and produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Tuple


class FactType(Enum):
    """
    Enumerates the sensory primitives captured by the first tier.
    """

    OBJECT = auto()
    EVENT = auto()
    CONTEXT = auto()


@dataclass(frozen=True, slots=True)
class Fact:
    """
    Represents an observed, low-level data point extracted from raw inputs.

    Attributes:
        label: Canonical name of the observed element (e.g., \"car\").
        description: Human-readable description, retained for explainability.
        fact_type: Category describing the fact (object, event, context, etc.).
        metadata: Arbitrary structured attributes for downstream enrichment.
    """

    label: str
    description: str
    fact_type: FactType
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Relation:
    """
    Captures causal or correlative links inferred between facts.

    Attributes:
        subject: Origin fact participating in the relationship.
        predicate: Relationship label (e.g., \"caused\", \"impacted\").
        obj: Destination fact participating in the relationship.
        weight: Optional confidence score (0..1) expressing inference strength.
    """

    subject: Fact
    predicate: str
    obj: Fact
    weight: Optional[float] = None


@dataclass(frozen=True, slots=True)
class AbstractConclusion:
    """
    Expresses the highest-tier reasoning outputs such as legal or ethical views.

    Attributes:
        theme: High-level domain of the conclusion (legal, ethical, policy, etc.).
        summary: Concise statement summarizing the reasoning outcome.
        supporting_relations: Relations that substantiate the conclusion.
        recommended_actions: Optional follow-up steps for decision-makers.
    """

    theme: str
    summary: str
    supporting_relations: Tuple[Relation, ...] = ()
    recommended_actions: Tuple[str, ...] = ()


def ensure_tuple(items: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """
    Utility to coerce optional iterables into tuples for immutability guarantees.
    """

    if items is None:
        return ()
    return tuple(items)


__all__: List[str] = [
    "AbstractConclusion",
    "Fact",
    "FactType",
    "Relation",
    "ensure_tuple",
]

