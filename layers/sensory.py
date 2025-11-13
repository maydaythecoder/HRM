"""
Low-level perception tier responsible for converting raw observations into facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence, Tuple

from models.types import Fact, FactType


Observation = Mapping[str, object]
FactResolver = Callable[[Observation], Fact]


def _default_fact_resolver(observation: Observation) -> Fact:
    """
    Default conversion from loosely structured observations into `Fact` objects.

    The resolver expects the following keys:
        - `label`: concise identifier for the observed entity.
        - `description`: human-readable summary of what was observed.
        - `type`: string matching one of the `FactType` enum values (case-insensitive).
        - `metadata`: optional mapping with additional structured attributes.
    """

    label = str(observation.get("label", "unknown"))
    description = str(observation.get("description", label))
    raw_type = str(observation.get("type", "context")).upper()
    fact_type = FactType[raw_type] if raw_type in FactType.__members__ else FactType.CONTEXT
    metadata = observation.get("metadata")

    structured_metadata = (
        dict(metadata) if isinstance(metadata, Mapping) else {}
    )

    return Fact(
        label=label,
        description=description,
        fact_type=fact_type,
        metadata={str(k): str(v) for k, v in structured_metadata.items()},
    )


@dataclass
class SensoryLayer:
    """
    Tier-1 component tasked with translating raw sensory inputs into structured facts.

    Attributes:
        resolver: Strategy function that maps an observation into a `Fact`. The
                  default resolver handles dictionary-like inputs with the keys
                  documented in `_default_fact_resolver`.
    """

    resolver: FactResolver = _default_fact_resolver

    def extract(self, observations: Sequence[Observation]) -> Tuple[Fact, ...]:
        """
        Convert incoming observations into immutable `Fact` instances.

        Args:
            observations: Sequence of loosely-typed inputs from upstream systems.

        Returns:
            Tuple of `Fact` objects ready for downstream inference.
        """

        facts: Iterable[Fact] = (self.resolver(obs) for obs in observations)
        return tuple(facts)


__all__ = ["Observation", "SensoryLayer"]

