"""
Typed containers representing the information exchanged between reasoning tiers.

These structures enable loose coupling between layers while preserving clear
contracts for what each tier consumes and produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    from torch import Tensor
else:
    try:
        from torch import Tensor
    except ImportError:
        Tensor = object


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


@dataclass(frozen=True, slots=True)
class FactEmbedding:
    """
    Tensor representation of a Fact for neural processing.

    Attributes:
        embedding: Dense vector representation of the fact (shape: [embedding_dim]).
        fact_type_idx: Integer index of the fact type (0=OBJECT, 1=EVENT, 2=CONTEXT).
        metadata_embedding: Optional dense vector for metadata (shape: [metadata_dim]).
        label: Original label for traceability.
    """

    embedding: Tensor
    fact_type_idx: int
    metadata_embedding: Optional[Tensor] = None
    label: str = ""


@dataclass(frozen=True, slots=True)
class RelationEmbedding:
    """
    Tensor representation of a Relation for neural processing.

    Attributes:
        subject_idx: Index of the subject fact in the fact sequence.
        obj_idx: Index of the object fact in the fact sequence.
        predicate_embedding: Dense vector for the predicate (shape: [predicate_dim]).
        weight: Confidence score (0..1) from the neural model.
        predicate_label: Original predicate string for traceability.
    """

    subject_idx: int
    obj_idx: int
    predicate_embedding: Tensor
    weight: float
    predicate_label: str = ""


@dataclass(frozen=True, slots=True)
class ConclusionEmbedding:
    """
    Tensor representation of an AbstractConclusion for neural processing.

    Attributes:
        theme_embedding: Dense vector for the theme (shape: [theme_dim]).
        summary_embedding: Dense vector for the summary (shape: [summary_dim]).
        theme_label: Original theme string for traceability.
        supporting_relation_indices: Indices of supporting relations in the relation sequence.
    """

    theme_embedding: Tensor
    summary_embedding: Tensor
    theme_label: str = ""
    supporting_relation_indices: Tuple[int, ...] = ()


def fact_to_embedding(
    fact: Fact,
    embedding: Tensor,
    fact_type_mapping: Dict[FactType, int],
    metadata_embedding: Optional[Tensor] = None,
) -> FactEmbedding:
    """
    Convert a Fact to its tensor representation.

    Args:
        fact: The fact to convert.
        embedding: Dense embedding vector for the fact.
        fact_type_mapping: Mapping from FactType enum to integer index.
        metadata_embedding: Optional embedding for metadata.

    Returns:
        FactEmbedding instance.
    """
    fact_type_idx = fact_type_mapping.get(fact.fact_type, 0)
    return FactEmbedding(
        embedding=embedding,
        fact_type_idx=fact_type_idx,
        metadata_embedding=metadata_embedding,
        label=fact.label,
    )


def relation_to_embedding(
    relation: Relation,
    subject_idx: int,
    obj_idx: int,
    predicate_embedding: Tensor,
    weight: float,
) -> RelationEmbedding:
    """
    Convert a Relation to its tensor representation.

    Args:
        relation: The relation to convert.
        subject_idx: Index of subject fact in the fact sequence.
        obj_idx: Index of object fact in the fact sequence.
        predicate_embedding: Dense embedding for the predicate.
        weight: Confidence score from the model.

    Returns:
        RelationEmbedding instance.
    """
    return RelationEmbedding(
        subject_idx=subject_idx,
        obj_idx=obj_idx,
        predicate_embedding=predicate_embedding,
        weight=weight,
        predicate_label=relation.predicate,
    )


def conclusion_to_embedding(
    conclusion: AbstractConclusion,
    theme_embedding: Tensor,
    summary_embedding: Tensor,
    supporting_relation_indices: Tuple[int, ...],
) -> ConclusionEmbedding:
    """
    Convert an AbstractConclusion to its tensor representation.

    Args:
        conclusion: The conclusion to convert.
        theme_embedding: Dense embedding for the theme.
        summary_embedding: Dense embedding for the summary.
        supporting_relation_indices: Indices of supporting relations.

    Returns:
        ConclusionEmbedding instance.
    """
    return ConclusionEmbedding(
        theme_embedding=theme_embedding,
        summary_embedding=summary_embedding,
        theme_label=conclusion.theme,
        supporting_relation_indices=supporting_relation_indices,
    )


def get_fact_type_mapping() -> Dict[FactType, int]:
    """
    Get standard mapping from FactType enum to integer indices.

    Returns:
        Dictionary mapping FactType to integer index.
    """
    return {
        FactType.OBJECT: 0,
        FactType.EVENT: 1,
        FactType.CONTEXT: 2,
    }


__all__: List[str] = [
    "AbstractConclusion",
    "ConclusionEmbedding",
    "Fact",
    "FactEmbedding",
    "FactType",
    "Relation",
    "RelationEmbedding",
    "conclusion_to_embedding",
    "ensure_tuple",
    "fact_to_embedding",
    "get_fact_type_mapping",
    "relation_to_embedding",
]

