"""
Data transformation utilities for preprocessing observations.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from layers.sensory import Observation


class ObservationTransform:
    """
    Transform for preprocessing observations before neural processing.

    Handles normalization, padding, and encoding of observation data.
    """

    def __init__(
        self,
        max_metadata_keys: int = 10,
        normalize_metadata: bool = True,
    ) -> None:
        """
        Initialize the observation transform.

        Args:
            max_metadata_keys: Maximum number of metadata keys to process.
            normalize_metadata: Whether to normalize metadata values.
        """
        self.max_metadata_keys = max_metadata_keys
        self.normalize_metadata = normalize_metadata

    def __call__(self, observation: Observation) -> Dict[str, Any]:
        """
        Transform a single observation.

        Args:
            observation: Input observation dictionary.

        Returns:
            Transformed observation dictionary.
        """
        transformed = {
            "label": str(observation.get("label", "unknown")),
            "description": str(observation.get("description", "")),
            "type": str(observation.get("type", "context")),
        }

        metadata = observation.get("metadata", {})
        if isinstance(metadata, Mapping):
            transformed["metadata"] = self._transform_metadata(metadata)
        else:
            transformed["metadata"] = {}

        return transformed

    def _transform_metadata(self, metadata: Mapping[str, Any]) -> Dict[str, str]:
        """
        Transform metadata dictionary.

        Args:
            metadata: Original metadata dictionary.

        Returns:
            Transformed metadata dictionary.
        """
        transformed = {}
        sorted_items = sorted(metadata.items())[: self.max_metadata_keys]

        for key, value in sorted_items:
            if isinstance(value, (int, float)):
                if self.normalize_metadata:
                    transformed[str(key)] = str(float(value) / 1000.0)
                else:
                    transformed[str(key)] = str(value)
            elif isinstance(value, str):
                transformed[str(key)] = value
            else:
                transformed[str(key)] = str(value)

        return transformed


__all__ = ["ObservationTransform"]

