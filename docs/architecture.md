# Hierarchical Reasoning Architecture

This document outlines the layered structure used to simulate hierarchical reasoning. The system supports both rule-based and neural network implementations, sharing the same data contracts and hierarchical structure.

## Tier Overview

The HRM executes structured reasoning in three tiers:

1. **Sensory Layer** — Converts raw observations into immutable `Fact` objects (rule-based) or dense embeddings (neural) while preserving metadata needed for downstream inferences.
2. **Inference Layer** — Applies rule pipelines (rule-based) or Graph Neural Networks (neural) to derive `Relation` objects that capture causality, impact, and policy breaches.
3. **Abstract Layer** — Consumes facts and relations to generate `AbstractConclusion` outputs for legal, ethical, and policy interpretations using rule functions (rule-based) or attention mechanisms (neural).

## Implementation Approaches

### Rule-Based Architecture

The rule-based implementation uses deterministic Python functions:

``` txt
observations -> SensoryLayer.extract -> facts
facts -> InferenceLayer.infer -> relations
(facts, relations) -> AbstractLayer.abstract -> conclusions
```

**Components:**

- `layers.sensory.SensoryLayer`: Dictionary-based fact extraction
- `layers.inference.InferenceLayer`: Python rule functions
- `layers.abstract.AbstractLayer`: Python conclusion functions
- `models.hierarchical_reasoner.HierarchicalReasoner`: Coordinator

**Characteristics:**

- Fully interpretable and deterministic
- No training required
- Easy to debug and modify
- Best for scenarios with clear logical rules

### Neural Network Architecture

The neural implementation uses deep learning models:

``` txt
observations -> SensoryEncoder -> fact_embeddings
fact_embeddings -> RelationNetwork -> relation_embeddings
(fact_embeddings, relation_embeddings) -> AbstractReasoner -> conclusion_embeddings
```

**Components:**

- `models.neural.sensory_encoder.SensoryEncoder`: Transformer-based text encoder (DistilBERT)
- `models.neural.relation_network.RelationNetwork`: Graph Convolutional Network (GCN)
- `models.neural.abstract_reasoner.AbstractReasoner`: Multi-head attention mechanism
- `models.neural.hierarchical_neural.HierarchicalNeuralReasoner`: Coordinator

**Characteristics:**

- Learns from data through supervised training
- Handles ambiguity and uncertainty
- Scales to large datasets
- Can be fine-tuned for specific domains

## Data Flow

### Rule-Based Flow

``` txt
Raw Observations (dict)
    ↓
[SensoryLayer]
    ↓
Facts (immutable dataclasses)
    ↓
[InferenceLayer with Rules]
    ↓
Relations (with weights)
    ↓
[AbstractLayer with Rules]
    ↓
AbstractConclusions
```

### Neural Flow

``` txt
Raw Observations (dict)
    ↓
[SensoryEncoder: Transformer]
    ↓
Fact Embeddings (tensors)
    ↓
[RelationNetwork: GCN]
    ↓
Relation Embeddings + Weights (tensors)
    ↓
[AbstractReasoner: Attention]
    ↓
Conclusion Embeddings (tensors)
    ↓
[Optional: Reconstruction]
    ↓
AbstractConclusions (dataclasses)
```

## Shared Data Contracts

Both implementations use the same core data structures defined in `models.types`:

- **`Fact`**: Immutable representation of observed data points
- **`Relation`**: Causal or correlative links between facts
- **`AbstractConclusion`**: High-level reasoning outputs

The neural implementation adds tensor representations:

- **`FactEmbedding`**: Tensor representation of facts
- **`RelationEmbedding`**: Tensor representation of relations
- **`ConclusionEmbedding`**: Tensor representation of conclusions

## Interoperability

The neural implementation can reconstruct the original dataclass objects from embeddings, allowing seamless integration with rule-based components. This enables:

- Hybrid approaches combining rules and neural models
- Gradual migration from rules to neural models
- Comparison and validation between approaches

## Choosing an Implementation

**Use Rule-Based When:**

- You have clear, well-defined logical rules
- Interpretability is critical
- Deterministic outputs are required
- No training data is available

**Use Neural When:**

- You have labeled training data
- Handling ambiguity and uncertainty is important
- Scaling to large datasets is needed
- Learning complex patterns from data is required

Each tier exposes a narrow contract so new rules or models can be introduced without destabilising the rest of the system.
