"""
Neural abstract reasoner for generating high-level conclusions.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.types import ConclusionEmbedding, FactEmbedding, RelationEmbedding


class AbstractReasoner(nn.Module):
    """
    Neural abstract reasoner that generates conclusions from facts and relations.

    Uses multi-head attention to aggregate information from facts and relations,
    then generates theme and summary embeddings for abstract conclusions.

    Attributes:
        fact_embedding_dim: Dimension of fact embeddings.
        relation_embedding_dim: Dimension of relation predicate embeddings.
        hidden_dim: Hidden dimension for attention and MLP layers.
        num_heads: Number of attention heads.
        num_theme_classes: Number of distinct theme types.
        summary_dim: Dimension for summary embeddings.
        attention: Multi-head attention layer.
        theme_classifier: MLP for classifying conclusion themes.
        summary_generator: MLP for generating summary embeddings.
        relation_selector: MLP for selecting supporting relations.
        dropout: Dropout layer for regularization.
    """

    def __init__(
        self,
        fact_embedding_dim: int = 256,
        relation_embedding_dim: int = 10,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_theme_classes: int = 5,
        summary_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the abstract reasoner.

        Args:
            fact_embedding_dim: Dimension of input fact embeddings.
            relation_embedding_dim: Dimension of relation predicate embeddings.
            hidden_dim: Hidden dimension for internal layers.
            num_heads: Number of attention heads.
            num_theme_classes: Number of distinct theme types.
            summary_dim: Dimension for summary embeddings.
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        self.fact_embedding_dim = fact_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_theme_classes = num_theme_classes
        self.summary_dim = summary_dim

        combined_dim = fact_embedding_dim + relation_embedding_dim

        self.fact_projection = nn.Linear(fact_embedding_dim, hidden_dim)
        self.relation_projection = nn.Linear(relation_embedding_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.theme_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_theme_classes),
        )

        self.summary_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, summary_dim),
        )

        self.relation_selector = nn.Sequential(
            nn.Linear(hidden_dim + relation_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        fact_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        relation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate conclusion embeddings from facts and relations.

        Args:
            fact_embeddings: Tensor of shape [batch_size, num_facts, fact_embedding_dim].
            relation_embeddings: Tensor of shape [batch_size, num_relations, relation_embedding_dim].
            relation_indices: Optional tensor of shape [batch_size, num_relations, 2] with
                            (subject_idx, obj_idx) pairs for each relation.

        Returns:
            Tuple of:
                - theme_logits: Tensor of shape [batch_size, num_theme_classes]
                - summary_embeddings: Tensor of shape [batch_size, summary_dim]
                - relation_scores: Tensor of shape [batch_size, num_relations] with
                                  scores for supporting relations
        """
        batch_size = fact_embeddings.size(0)

        fact_proj = self.fact_projection(fact_embeddings)
        relation_proj = self.relation_projection(relation_embeddings)

        combined_context = torch.cat([fact_proj, relation_proj], dim=1)

        attended, _ = self.attention(
            combined_context,
            combined_context,
            combined_context,
        )

        attended = self.layer_norm(attended + combined_context)
        attended = self.dropout(attended)

        pooled = torch.mean(attended, dim=1)

        theme_logits = self.theme_classifier(pooled)
        summary_embeddings = self.summary_generator(pooled)

        if relation_indices is not None:
            relation_scores = self._compute_relation_scores(
                pooled,
                relation_embeddings,
                relation_indices,
            )
        else:
            num_relations = relation_embeddings.size(1)
            relation_scores = torch.ones(
                (batch_size, num_relations),
                device=fact_embeddings.device,
            )

        return theme_logits, summary_embeddings, relation_scores

    def _compute_relation_scores(
        self,
        context_embedding: torch.Tensor,
        relation_embeddings: torch.Tensor,
        relation_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores for which relations support the conclusion.

        Args:
            context_embedding: Pooled context embedding of shape [batch_size, hidden_dim].
            relation_embeddings: Relation embeddings of shape [batch_size, num_relations, relation_embedding_dim].
            relation_indices: Relation indices of shape [batch_size, num_relations, 2].

        Returns:
            Relation scores tensor of shape [batch_size, num_relations].
        """
        batch_size, num_relations, _ = relation_embeddings.shape

        context_expanded = context_embedding.unsqueeze(1).expand(-1, num_relations, -1)
        combined = torch.cat([context_expanded, relation_embeddings], dim=-1)

        scores = self.relation_selector(combined).squeeze(-1)
        return scores

    def generate_conclusions(
        self,
        fact_embeddings: Sequence[FactEmbedding],
        relation_embeddings: Sequence[RelationEmbedding],
        theme_vocab: Optional[Sequence[str]] = None,
        relation_threshold: float = 0.5,
    ) -> Tuple[ConclusionEmbedding, ...]:
        """
        Generate conclusion embeddings from fact and relation embeddings.

        Args:
            fact_embeddings: Sequence of FactEmbedding objects.
            relation_embeddings: Sequence of RelationEmbedding objects.
            theme_vocab: Optional vocabulary of theme strings.
            relation_threshold: Threshold for selecting supporting relations.

        Returns:
            Tuple of ConclusionEmbedding objects.
        """
        if not fact_embeddings or not relation_embeddings:
            return ()

        device = next(self.parameters()).device

        fact_tensor = torch.stack([fe.embedding for fe in fact_embeddings]).unsqueeze(0).to(device)
        relation_tensor = torch.stack(
            [re.predicate_embedding for re in relation_embeddings]
        ).unsqueeze(0).to(device)

        relation_indices = torch.tensor(
            [[re.subject_idx, re.obj_idx] for re in relation_embeddings],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        theme_logits, summary_embeddings, relation_scores = self.forward(
            fact_tensor,
            relation_tensor,
            relation_indices,
        )

        theme_probs = F.softmax(theme_logits, dim=-1)
        theme_idx = torch.argmax(theme_probs, dim=-1).item()

        theme_label = (
            theme_vocab[theme_idx]
            if theme_vocab and theme_idx < len(theme_vocab)
            else f"theme_{theme_idx}"
        )

        summary_embedding = summary_embeddings[0]
        relation_score_vec = relation_scores[0]

        supporting_indices = [
            i
            for i, score in enumerate(relation_score_vec)
            if score.item() >= relation_threshold
        ]

        return (
            ConclusionEmbedding(
                theme_embedding=theme_probs[0],
                summary_embedding=summary_embedding,
                theme_label=theme_label,
                supporting_relation_indices=tuple(supporting_indices),
            ),
        )


__all__ = ["AbstractReasoner"]

