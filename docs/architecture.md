# Hierarchical Reasoning Architecture

This document outlines the layered structure used to simulate hierarchical reasoning.

## Tier Overview

1. **Sensory Layer** — Converts raw observations into immutable `Fact` objects while preserving metadata needed for downstream inferences.
2. **Inference Layer** — Applies a rule pipeline to derive `Relation` objects that capture causality, impact, and policy breaches.
3. **Abstract Layer** — Consumes facts and relations to generate `AbstractConclusion` outputs for legal, ethical, and policy interpretations.

## Data Flow

```
observations -> SensoryLayer.extract -> facts
facts -> InferenceLayer.infer -> relations
(facts, relations) -> AbstractLayer.abstract -> conclusions
```

Each tier exposes a narrow contract so new rules can be introduced without destabilising the rest of the system.

