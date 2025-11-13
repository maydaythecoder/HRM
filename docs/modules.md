# Module Responsibilities

- `layers.sensory`: Adapts raw Python mappings to structured `Fact` instances using pluggable resolver strategies.
- `layers.inference`: Hosts inference rules that transform facts into `Relation` objects representing causality and compliance.
- `layers.abstract`: Evaluates `Fact` and `Relation` collections to produce high-level `AbstractConclusion` outputs.
- `models.hierarchical_reasoner`: Coordinates the tier execution order and exposes a composable factory for custom rule sets.
- `examples.car_accident`: Provides a runnable scenario illustrating how the tiers collaborate on a real-world incident.

