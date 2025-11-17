# Neural Architecture for Hierarchical Reasoning Model

This document describes the neural network implementation of the Hierarchical Reasoning Model (HRM), which replaces the rule-based system with learnable neural components.

## Overview

The neural HRM maintains the same three-tier hierarchical structure as the rule-based version, but uses deep learning models to learn representations and relationships from data:

1. **Sensory Encoder**: Converts observations to dense fact embeddings using transformer-based text encoders
2. **Relation Network**: Learns relations between facts using Graph Neural Networks (GNNs)
3. **Abstract Reasoner**: Generates high-level conclusions using multi-head attention mechanisms

## Architecture Details

### Sensory Encoder (`SensoryEncoder`)

The sensory encoder transforms raw observations into dense vector representations suitable for neural processing.

**Components:**

- Pre-trained transformer encoder (default: DistilBERT) for text encoding
- Fact type embedding layer (3 types: OBJECT, EVENT, CONTEXT)
- Optional metadata encoder (MLP) for structured attributes
- Projection layer to target embedding dimension

**Input:** Sequence of observation dictionaries
**Output:** Fact embeddings tensor `[num_facts, embedding_dim]`

**Key Features:**

- Handles variable-length text descriptions
- Encodes metadata as additional features
- Produces fixed-size embeddings for downstream processing

### Relation Network (`RelationNetwork`)

The relation network learns to identify and score relationships between facts using graph-based reasoning.

**Components:**

- Graph Convolutional Network (GCN) layers for fact interaction
- Relation classifier (MLP) for predicate prediction
- Relation scorer (MLP) for confidence estimation

**Input:** Fact embeddings `[num_facts, embedding_dim]`
**Output:**

- Predicate logits `[num_edges, num_predicate_classes]`
- Relation weights `[num_edges]` (confidence scores)

**Key Features:**

- Constructs fully connected graph of facts
- Learns edge features by concatenating source and target node embeddings
- Predicts both relation type and confidence simultaneously

### Abstract Reasoner (`AbstractReasoner`)

The abstract reasoner generates high-level conclusions by attending to facts and relations.

**Components:**

- Multi-head self-attention mechanism
- Theme classifier (MLP) for conclusion categorization
- Summary generator (MLP) for conclusion summarization
- Relation selector (MLP) for identifying supporting relations

**Input:**

- Fact embeddings `[batch_size, num_facts, fact_embedding_dim]`
- Relation embeddings `[batch_size, num_relations, relation_embedding_dim]`

**Output:**

- Theme logits `[batch_size, num_theme_classes]`
- Summary embeddings `[batch_size, summary_dim]`
- Relation scores `[batch_size, num_relations]`

**Key Features:**

- Uses attention to aggregate information from all facts and relations
- Generates structured conclusions with themes and summaries
- Identifies which relations support each conclusion

## Neural Data Flow

``` txt
Observations
    ↓
[Sensory Encoder]
    ↓
Fact Embeddings
    ↓
[Relation Network]
    ↓
Relation Embeddings + Weights
    ↓
[Abstract Reasoner]
    ↓
Conclusion Embeddings
```

## Training

The model uses multi-task learning with three loss components:

1. **Fact Loss**: MSE between predicted and target fact embeddings
2. **Relation Loss**: Cross-entropy for predicate classification + MSE for confidence scores
3. **Conclusion Loss**: Cross-entropy for theme classification + MSE for summary embeddings

Total loss is a weighted sum: `L_total = w_fact * L_fact + w_relation * L_relation + w_conclusion * L_conclusion`

## Usage

### Basic Inference

```python
from models.neural.hierarchical_neural import build_neural_reasoner

reasoner = build_neural_reasoner(
    embedding_dim=256,
    hidden_dim=128,
    num_predicate_classes=10,
    num_theme_classes=5,
)

observations = [
    {"label": "event_1", "description": "...", "type": "EVENT"},
    # ... more observations
]

result = reasoner.analyze(observations, reconstruct_objects=True)
```

 Training

```bash
python training/train.py --config training/config.yaml
```

## Configuration

Key hyperparameters (see `training/config.yaml`):

- `embedding_dim`: Dimension of fact embeddings (default: 256)
- `hidden_dim`: Hidden dimension for neural layers (default: 128)
- `num_predicate_classes`: Number of relation predicate types (default: 10)
- `num_theme_classes`: Number of conclusion theme types (default: 5)
- `dropout`: Dropout probability (default: 0.1)
- `learning_rate`: Initial learning rate (default: 1e-4)

## Comparison with Rule-Based System

| Aspect | Rule-Based | Neural |
|--------|-----------|--------|
| Fact Extraction | Dictionary mapping | Learned embeddings |
| Relation Inference | Hand-coded rules | GNN-based learning |
| Conclusion Generation | Rule functions | Attention-based |
| Training | None | Supervised learning |
| Interpretability | High | Medium (attention weights) |
| Adaptability | Manual updates | Data-driven |

## Extending the Neural Model

### Adding New Predicate Types

1. Update `num_predicate_classes` in config
2. Provide `predicate_vocab` when building reasoner
3. Retrain with labeled data

### Adding New Theme Types

1. Update `num_theme_classes` in config
2. Provide `theme_vocab` when building reasoner
3. Retrain with labeled data

### Custom Encoders

Replace `SensoryEncoder` with custom implementation:

```python
class CustomSensoryEncoder(nn.Module):
    # ... implementation
```

Then pass to `HierarchicalNeuralReasoner` constructor.

## Performance Considerations

- **GPU Acceleration**: Model automatically uses CUDA if available
- **Batch Processing**: Supports batched inference for efficiency
- **Memory**: GCN creates fully connected graph (O(n²) edges for n facts)
- **Inference Speed**: Transformer encoding is the bottleneck; consider smaller models for production

## Future Enhancements

- Sparse graph construction for large fact sets
- Pre-training on large-scale reasoning datasets
- Hybrid rule-neural approaches for interpretability
- Quantization and model compression for deployment
