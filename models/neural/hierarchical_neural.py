"""
Coordinator for the neural hierarchical reasoning pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from layers.sensory import Observation
from models.neural.abstract_reasoner import AbstractReasoner
from models.neural.relation_network import RelationNetwork
from models.neural.sensory_encoder import SensoryEncoder
from models.types import (
    AbstractConclusion,
    ConclusionEmbedding,
    Fact,
    FactEmbedding,
    Relation,
    RelationEmbedding,
)


@dataclass(frozen=True, slots=True)
class NeuralReasoningResult:
    """
    Snapshot of the complete neural reasoning trace.

    Attributes:
        fact_embeddings: Output embeddings from the sensory encoder.
        relation_embeddings: Output embeddings from the relation network.
        conclusion_embeddings: Output embeddings from the abstract reasoner.
        facts: Optional reconstructed Fact objects.
        relations: Optional reconstructed Relation objects.
        conclusions: Optional reconstructed AbstractConclusion objects.
    """

    fact_embeddings: Tuple[FactEmbedding, ...]
    relation_embeddings: Tuple[RelationEmbedding, ...]
    conclusion_embeddings: Tuple[ConclusionEmbedding, ...]
    facts: Optional[Tuple[Fact, ...]] = None
    relations: Optional[Tuple[Relation, ...]] = None
    conclusions: Optional[Tuple[AbstractConclusion, ...]] = None


class HierarchicalNeuralReasoner:
    """
    Orchestrates the three-tier neural reasoning workflow.

    Coordinates the sensory encoder, relation network, and abstract reasoner
    to perform end-to-end hierarchical reasoning on observations.

    Attributes:
        sensory_encoder: Neural encoder for converting observations to facts.
        relation_network: Neural network for learning relations between facts.
        abstract_reasoner: Neural reasoner for generating abstract conclusions.
        device: Device to run computations on.
        predicate_vocab: Optional vocabulary of predicate strings.
        theme_vocab: Optional vocabulary of theme strings.
    """

    def __init__(
        self,
        sensory_encoder: SensoryEncoder,
        relation_network: RelationNetwork,
        abstract_reasoner: AbstractReasoner,
        device: Optional[torch.device] = None,
        predicate_vocab: Optional[Sequence[str]] = None,
        theme_vocab: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the hierarchical neural reasoner.

        Args:
            sensory_encoder: Sensory encoder module.
            relation_network: Relation network module.
            abstract_reasoner: Abstract reasoner module.
            device: Device to run on. If None, uses CUDA if available, else CPU.
            predicate_vocab: Optional vocabulary of predicate strings.
            theme_vocab: Optional vocabulary of theme strings.
        """
        self.sensory_encoder = sensory_encoder
        self.relation_network = relation_network
        self.abstract_reasoner = abstract_reasoner

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.sensory_encoder.to(device)
        self.relation_network.to(device)
        self.abstract_reasoner.to(device)

        self.predicate_vocab = predicate_vocab
        self.theme_vocab = theme_vocab

    def analyze(
        self,
        observations: Sequence[Observation],
        relation_threshold: float = 0.5,
        reconstruct_objects: bool = False,
    ) -> NeuralReasoningResult:
        """
        Run the full neural pipeline against the provided observations.

        Args:
            observations: Sequence of observation dictionaries.
            relation_threshold: Confidence threshold for including relations.
            reconstruct_objects: If True, reconstructs Fact, Relation, and
                               AbstractConclusion objects from embeddings.

        Returns:
            NeuralReasoningResult containing embeddings and optionally
            reconstructed objects.
        """
        self.sensory_encoder.eval()
        self.relation_network.eval()
        self.abstract_reasoner.eval()

        with torch.no_grad():
            fact_embeddings = self.sensory_encoder.encode_facts(observations)

            relation_embeddings = self.relation_network.predict_relations(
                fact_embeddings,
                predicate_vocab=self.predicate_vocab,
                threshold=relation_threshold,
            )

            conclusion_embeddings = self.abstract_reasoner.generate_conclusions(
                fact_embeddings,
                relation_embeddings,
                theme_vocab=self.theme_vocab,
                relation_threshold=relation_threshold,
            )

        facts = None
        relations = None
        conclusions = None

        if reconstruct_objects:
            facts = self._reconstruct_facts(observations, fact_embeddings)
            relations = self._reconstruct_relations(facts, relation_embeddings)
            conclusions = self._reconstruct_conclusions(
                relations,
                conclusion_embeddings,
            )

        return NeuralReasoningResult(
            fact_embeddings=fact_embeddings,
            relation_embeddings=relation_embeddings,
            conclusion_embeddings=conclusion_embeddings,
            facts=facts,
            relations=relations,
            conclusions=conclusions,
        )

    def _reconstruct_facts(
        self,
        observations: Sequence[Observation],
        fact_embeddings: Sequence[FactEmbedding],
    ) -> Tuple[Fact, ...]:
        """Reconstruct Fact objects from observations and embeddings."""
        from models.types import FactType

        facts = []
        for obs, fe in zip(observations, fact_embeddings):
            raw_type = str(obs.get("type", "context")).upper()
            fact_type = (
                FactType[raw_type]
                if raw_type in FactType.__members__
                else FactType.CONTEXT
            )

            fact = Fact(
                label=fe.label,
                description=str(obs.get("description", "")),
                fact_type=fact_type,
                metadata=dict(obs.get("metadata", {})),
            )
            facts.append(fact)

        return tuple(facts)

    def _reconstruct_relations(
        self,
        facts: Sequence[Fact],
        relation_embeddings: Sequence[RelationEmbedding],
    ) -> Tuple[Relation, ...]:
        """Reconstruct Relation objects from facts and relation embeddings."""
        relations = []
        for re in relation_embeddings:
            if re.subject_idx < len(facts) and re.obj_idx < len(facts):
                relation = Relation(
                    subject=facts[re.subject_idx],
                    predicate=re.predicate_label,
                    obj=facts[re.obj_idx],
                    weight=re.weight,
                )
                relations.append(relation)

        return tuple(relations)

    def _reconstruct_conclusions(
        self,
        relations: Sequence[Relation],
        conclusion_embeddings: Sequence[ConclusionEmbedding],
    ) -> Tuple[AbstractConclusion, ...]:
        """Reconstruct AbstractConclusion objects from relations and conclusion embeddings."""
        conclusions = []
        for ce in conclusion_embeddings:
            supporting = tuple(
                relations[i]
                for i in ce.supporting_relation_indices
                if i < len(relations)
            )

            conclusion = AbstractConclusion(
                theme=ce.theme_label,
                summary=f"Summary embedding dimension: {ce.summary_embedding.shape[0]}",
                supporting_relations=supporting,
                recommended_actions=(),
            )
            conclusions.append(conclusion)

        return tuple(conclusions)

    def train(self) -> None:
        """Set all modules to training mode."""
        self.sensory_encoder.train()
        self.relation_network.train()
        self.abstract_reasoner.train()

    def eval(self) -> None:
        """Set all modules to evaluation mode."""
        self.sensory_encoder.eval()
        self.relation_network.eval()
        self.abstract_reasoner.eval()


def build_neural_reasoner(
    embedding_dim: int = 256,
    hidden_dim: int = 128,
    num_predicate_classes: int = 10,
    num_theme_classes: int = 5,
    device: Optional[torch.device] = None,
    predicate_vocab: Optional[Sequence[str]] = None,
    theme_vocab: Optional[Sequence[str]] = None,
) -> HierarchicalNeuralReasoner:
    """
    Factory function for constructing a HierarchicalNeuralReasoner.

    Args:
        embedding_dim: Dimension for fact embeddings.
        hidden_dim: Hidden dimension for neural layers.
        num_predicate_classes: Number of predicate types.
        num_theme_classes: Number of theme types.
        device: Device to run on.
        predicate_vocab: Optional vocabulary of predicate strings.
        theme_vocab: Optional vocabulary of theme strings.

    Returns:
        Configured HierarchicalNeuralReasoner instance.
    """
    sensory_encoder = SensoryEncoder(embedding_dim=embedding_dim)
    relation_network = RelationNetwork(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_predicate_classes=num_predicate_classes,
    )
    abstract_reasoner = AbstractReasoner(
        fact_embedding_dim=embedding_dim,
        relation_embedding_dim=num_predicate_classes,
        hidden_dim=hidden_dim,
        num_theme_classes=num_theme_classes,
    )

    return HierarchicalNeuralReasoner(
        sensory_encoder=sensory_encoder,
        relation_network=relation_network,
        abstract_reasoner=abstract_reasoner,
        device=device,
        predicate_vocab=predicate_vocab,
        theme_vocab=theme_vocab,
    )


__all__ = ["HierarchicalNeuralReasoner", "NeuralReasoningResult", "build_neural_reasoner"]

