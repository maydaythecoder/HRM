"""
Neural relation network for learning relations between facts.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from models.types import FactEmbedding, RelationEmbedding


class RelationNetwork(nn.Module):
    """
    Neural network for learning relations between facts using graph-based reasoning.

    Uses a Graph Convolutional Network (GCN) to model interactions between facts
    and predict relations with confidence scores.

    Attributes:
        embedding_dim: Dimension of input fact embeddings.
        hidden_dim: Hidden dimension for GCN layers.
        num_layers: Number of GCN layers.
        num_predicate_classes: Number of distinct predicate types to predict.
        gcn_layers: Stack of GCN convolution layers.
        relation_classifier: MLP for classifying relation predicates.
        relation_scorer: MLP for scoring relation confidence.
        dropout: Dropout layer for regularization.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_predicate_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the relation network.

        Args:
            embedding_dim: Dimension of input fact embeddings.
            hidden_dim: Hidden dimension for GCN layers.
            num_layers: Number of GCN layers to stack.
            num_predicate_classes: Number of distinct predicate types.
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_predicate_classes = num_predicate_classes

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(embedding_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicate_classes),
        )

        self.relation_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        fact_embeddings: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict relations between facts.

        Args:
            fact_embeddings: Tensor of shape [num_facts, embedding_dim].
            edge_index: Optional edge index for graph structure. If None, creates
                       a fully connected graph.

        Returns:
            Tuple of:
                - predicate_logits: Tensor of shape [num_edges, num_predicate_classes]
                - relation_weights: Tensor of shape [num_edges] with confidence scores
        """
        num_facts = fact_embeddings.size(0)

        if edge_index is None:
            edge_index = self._create_fully_connected_graph(num_facts, fact_embeddings.device)

        x = fact_embeddings

        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        edge_features = self._extract_edge_features(x, edge_index)

        predicate_logits = self.relation_classifier(edge_features)
        relation_weights = self.relation_scorer(edge_features).squeeze(-1)

        return predicate_logits, relation_weights

    def _create_fully_connected_graph(
        self,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a fully connected graph edge index.

        Args:
            num_nodes: Number of nodes in the graph.
            device: Device to create tensor on.

        Returns:
            Edge index tensor of shape [2, num_edges].
        """
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])

        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    def _extract_edge_features(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features for each edge by concatenating source and target node features.

        Args:
            node_features: Tensor of shape [num_nodes, hidden_dim].
            edge_index: Edge index tensor of shape [2, num_edges].

        Returns:
            Edge features tensor of shape [num_edges, hidden_dim * 2].
        """
        source_indices = edge_index[0]
        target_indices = edge_index[1]

        source_features = node_features[source_indices]
        target_features = node_features[target_indices]

        edge_features = torch.cat([source_features, target_features], dim=-1)
        return edge_features

    def predict_relations(
        self,
        fact_embeddings: Sequence[FactEmbedding],
        predicate_vocab: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
    ) -> Tuple[RelationEmbedding, ...]:
        """
        Predict relations from fact embeddings and return RelationEmbedding objects.

        Args:
            fact_embeddings: Sequence of FactEmbedding objects.
            predicate_vocab: Optional vocabulary of predicate strings. If None,
                           uses indices as labels.
            threshold: Confidence threshold for including relations.

        Returns:
            Tuple of RelationEmbedding objects.
        """
        if not fact_embeddings:
            return ()

        embeddings_tensor = torch.stack([fe.embedding for fe in fact_embeddings])

        device = next(self.parameters()).device
        embeddings_tensor = embeddings_tensor.to(device)

        predicate_logits, relation_weights = self.forward(embeddings_tensor)

        num_facts = len(fact_embeddings)
        edge_index = self._create_fully_connected_graph(num_facts, device)

        predicate_probs = F.softmax(predicate_logits, dim=-1)
        predicate_indices = torch.argmax(predicate_probs, dim=-1)

        relations = []
        for i, (edge, weight, pred_idx) in enumerate(
            zip(edge_index.t(), relation_weights, predicate_indices)
        ):
            if weight.item() >= threshold:
                subject_idx = edge[0].item()
                obj_idx = edge[1].item()

                predicate_label = (
                    predicate_vocab[pred_idx.item()]
                    if predicate_vocab and pred_idx.item() < len(predicate_vocab)
                    else f"predicate_{pred_idx.item()}"
                )

                predicate_embedding = predicate_probs[i]

                relations.append(
                    RelationEmbedding(
                        subject_idx=subject_idx,
                        obj_idx=obj_idx,
                        predicate_embedding=predicate_embedding,
                        weight=weight.item(),
                        predicate_label=predicate_label,
                    )
                )

        return tuple(relations)


__all__ = ["RelationNetwork"]

