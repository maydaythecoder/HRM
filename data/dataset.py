"""
PyTorch Dataset implementation for hierarchical reasoning model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from data.transforms import ObservationTransform
from layers.sensory import Observation
from models.types import Fact, Relation, AbstractConclusion


class HRMDataset(Dataset):
    """
    PyTorch Dataset for hierarchical reasoning model training.

    Handles loading and preprocessing of observations along with optional
    ground truth facts, relations, and conclusions for supervised learning.

    Attributes:
        observations: List of observation dictionaries.
        facts: Optional ground truth facts for each sample.
        relations: Optional ground truth relations for each sample.
        conclusions: Optional ground truth conclusions for each sample.
        transform: Optional transform to apply to observations.
    """

    def __init__(
        self,
        observations: Sequence[Sequence[Observation]],
        facts: Optional[Sequence[Sequence[Fact]]] = None,
        relations: Optional[Sequence[Sequence[Relation]]] = None,
        conclusions: Optional[Sequence[Sequence[AbstractConclusion]]] = None,
        transform: Optional[ObservationTransform] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            observations: Sequence of observation sequences (one per sample).
            facts: Optional ground truth facts for each sample.
            relations: Optional ground truth relations for each sample.
            conclusions: Optional ground truth conclusions for each sample.
            transform: Optional transform to apply to observations.
        """
        self.observations = list(observations)
        self.facts = list(facts) if facts is not None else None
        self.relations = list(relations) if relations is not None else None
        self.conclusions = list(conclusions) if conclusions is not None else None

        self.transform = transform or ObservationTransform()

        if self.facts is not None and len(self.facts) != len(self.observations):
            raise ValueError(
                f"Facts length ({len(self.facts)}) must match observations length ({len(self.observations)})"
            )

        if self.relations is not None and len(self.relations) != len(self.observations):
            raise ValueError(
                f"Relations length ({len(self.relations)}) must match observations length ({len(self.observations)})"
            )

        if (
            self.conclusions is not None
            and len(self.conclusions) != len(self.observations)
        ):
            raise ValueError(
                f"Conclusions length ({len(self.conclusions)}) must match observations length ({len(self.observations)})"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.observations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - observations: Transformed observations
                - facts: Optional ground truth facts
                - relations: Optional ground truth relations
                - conclusions: Optional ground truth conclusions
        """
        obs_list = self.observations[idx]
        transformed_obs = [self.transform(obs) for obs in obs_list]

        sample: Dict[str, Any] = {
            "observations": transformed_obs,
        }

        if self.facts is not None:
            sample["facts"] = self.facts[idx]

        if self.relations is not None:
            sample["relations"] = self.relations[idx]

        if self.conclusions is not None:
            sample["conclusions"] = self.conclusions[idx]

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader to handle variable-length sequences.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Batched dictionary with observations and optional ground truth.
    """
    observations = [sample["observations"] for sample in batch]

    result: Dict[str, Any] = {
        "observations": observations,
    }

    if "facts" in batch[0]:
        result["facts"] = [sample["facts"] for sample in batch]

    if "relations" in batch[0]:
        result["relations"] = [sample["relations"] for sample in batch]

    if "conclusions" in batch[0]:
        result["conclusions"] = [sample["conclusions"] for sample in batch]

    return result


__all__ = ["HRMDataset", "collate_fn"]

