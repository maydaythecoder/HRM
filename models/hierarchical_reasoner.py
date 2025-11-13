"""
Coordinator for the hierarchical reasoning pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from layers.abstract import AbstractLayer, ConclusionRule
from layers.inference import InferenceLayer, InferenceRule
from layers.sensory import Observation, SensoryLayer
from models.types import AbstractConclusion, Fact, Relation


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """
    Snapshot of the complete reasoning trace produced by the pipeline.

    Attributes:
        facts: Output of the sensory layer.
        relations: Output of the inference layer.
        conclusions: Output of the abstract layer.
    """

    facts: Tuple[Fact, ...]
    relations: Tuple[Relation, ...]
    conclusions: Tuple[AbstractConclusion, ...]


@dataclass
class HierarchicalReasoner:
    """
    Orchestrates the three-tier reasoning workflow.
    """

    sensory: SensoryLayer
    inference: InferenceLayer
    abstract: AbstractLayer

    def analyze(self, observations: Sequence[Observation]) -> ReasoningResult:
        """
        Run the full pipeline against the provided observations.
        """

        facts = self.sensory.extract(observations)
        relations = self.inference.infer(facts)
        conclusions = self.abstract.abstract(facts, relations)
        return ReasoningResult(facts=facts, relations=relations, conclusions=conclusions)


def build_reasoner(
    *,
    sensory: SensoryLayer | None = None,
    inference_rules: Iterable[InferenceRule] | None = None,
    conclusion_rules: Iterable[ConclusionRule] | None = None,
) -> HierarchicalReasoner:
    """
    Factory for constructing a `HierarchicalReasoner` with optional custom rules.
    """

    sensory_layer = sensory or SensoryLayer()
    inference_layer = InferenceLayer(tuple(inference_rules or ()))
    abstract_layer = AbstractLayer(tuple(conclusion_rules or ()))
    return HierarchicalReasoner(
        sensory=sensory_layer,
        inference=inference_layer,
        abstract=abstract_layer,
    )


__all__ = ["HierarchicalReasoner", "ReasoningResult", "build_reasoner"]

