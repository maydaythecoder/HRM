"""
Neural sensory encoder for converting observations to fact embeddings.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from layers.sensory import Observation
from models.types import Fact, FactEmbedding, FactType, get_fact_type_mapping


class SensoryEncoder(nn.Module):
    """
    Neural encoder that converts observations into fact embeddings.

    Uses a transformer-based text encoder (e.g., BERT) to encode observation
    descriptions and metadata into dense embeddings suitable for downstream
    reasoning layers.

    Attributes:
        embedding_dim: Dimension of output fact embeddings.
        text_encoder: Pre-trained transformer model for text encoding.
        tokenizer: Tokenizer for the text encoder.
        fact_type_embedding: Embedding layer for fact type indices.
        metadata_encoder: Optional MLP for encoding metadata.
        projection: Linear layer to project to target embedding dimension.
        dropout: Dropout layer for regularization.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        model_name: str = "distilbert-base-uncased",
        metadata_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the sensory encoder.

        Args:
            embedding_dim: Target dimension for fact embeddings.
            model_name: HuggingFace model identifier for text encoding.
            metadata_dim: Dimension for metadata embeddings. If None, metadata
                         is not separately encoded.
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim

        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        encoder_dim = self.text_encoder.config.hidden_size

        self.fact_type_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=32,
        )

        if metadata_dim is not None:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
            )
        else:
            self.metadata_encoder = None

        combined_dim = encoder_dim + 32 + (32 if metadata_dim else 0)
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        observations: Sequence[Observation],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode observations into fact embeddings.

        Args:
            observations: Sequence of observation dictionaries.

        Returns:
            Tuple of:
                - fact_embeddings: Tensor of shape [batch_size, embedding_dim]
                - fact_type_indices: Tensor of shape [batch_size] with type indices
                - metadata_embeddings: Optional tensor of shape [batch_size, metadata_dim]
        """
        batch_size = len(observations)

        texts = [
            f"{obs.get('label', '')} {obs.get('description', '')}" for obs in observations
        ]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        text_outputs = self.text_encoder(**encoded)

        text_embeddings = text_outputs.last_hidden_state[:, 0, :]

        fact_type_indices = []
        metadata_vectors = []

        fact_type_mapping = get_fact_type_mapping()

        for obs in observations:
            raw_type = str(obs.get("type", "context")).upper()
            fact_type = FactType[raw_type] if raw_type in FactType.__members__ else FactType.CONTEXT
            fact_type_indices.append(fact_type_mapping[fact_type])

            if self.metadata_dim is not None:
                metadata = obs.get("metadata", {})
                if isinstance(metadata, Mapping):
                    metadata_vec = self._encode_metadata(metadata)
                else:
                    metadata_vec = torch.zeros(self.metadata_dim, device=device)
                metadata_vectors.append(metadata_vec)
            else:
                metadata_vectors.append(None)

        fact_type_tensor = torch.tensor(fact_type_indices, device=device, dtype=torch.long)
        fact_type_embeds = self.fact_type_embedding(fact_type_tensor)

        combined = torch.cat([text_embeddings, fact_type_embeds], dim=-1)

        if self.metadata_encoder is not None and metadata_vectors[0] is not None:
            metadata_tensor = torch.stack(metadata_vectors)
            metadata_embeds = self.metadata_encoder(metadata_tensor)
            combined = torch.cat([combined, metadata_embeds], dim=-1)

        fact_embeddings = self.projection(combined)
        fact_embeddings = self.dropout(fact_embeddings)

        metadata_output = (
            torch.stack(metadata_vectors) if metadata_vectors[0] is not None else None
        )

        return fact_embeddings, fact_type_tensor, metadata_output

    def _encode_metadata(self, metadata: Mapping[str, object]) -> torch.Tensor:
        """
        Encode metadata dictionary into a fixed-size vector.

        Args:
            metadata: Dictionary of metadata key-value pairs.

        Returns:
            Tensor of shape [metadata_dim] or [self.metadata_dim].
        """
        device = next(self.parameters()).device

        if self.metadata_dim is None:
            return torch.zeros(1, device=device)

        features = []
        for key, value in sorted(metadata.items()):
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                features.append(float(hash(value) % 1000) / 1000.0)
            else:
                features.append(0.0)

        while len(features) < self.metadata_dim:
            features.append(0.0)

        features = features[: self.metadata_dim]

        return torch.tensor(features, device=device, dtype=torch.float32)

    def encode_facts(
        self,
        observations: Sequence[Observation],
    ) -> Tuple[FactEmbedding, ...]:
        """
        Encode observations and return FactEmbedding objects.

        Args:
            observations: Sequence of observation dictionaries.

        Returns:
            Tuple of FactEmbedding objects.
        """
        fact_embeddings, fact_type_indices, metadata_embeddings = self.forward(observations)

        fact_type_mapping = get_fact_type_mapping()
        reverse_mapping = {v: k for k, v in fact_type_mapping.items()}

        result = []
        for i, obs in enumerate(observations):
            fact_type_idx = fact_type_indices[i].item()
            fact_type = reverse_mapping[fact_type_idx]

            fact = Fact(
                label=str(obs.get("label", "unknown")),
                description=str(obs.get("description", "")),
                fact_type=fact_type,
                metadata=dict(obs.get("metadata", {})),
            )

            embedding = fact_embeddings[i]
            metadata_embedding = (
                metadata_embeddings[i] if metadata_embeddings is not None else None
            )

            result.append(
                FactEmbedding(
                    embedding=embedding,
                    fact_type_idx=fact_type_idx,
                    metadata_embedding=metadata_embedding,
                    label=fact.label,
                )
            )

        return tuple(result)


__all__ = ["SensoryEncoder"]

