# Module Responsibilities

This document outlines the responsibilities of each module in the HRM system, covering both rule-based and neural implementations.

## Rule-Based Modules

### Core Layers

- **`layers.sensory`**: Adapts raw Python mappings to structured `Fact` instances using pluggable resolver strategies. Provides `SensoryLayer` class with configurable fact extraction logic.

- **`layers.inference`**: Hosts inference rules that transform facts into `Relation` objects representing causality and compliance. Provides `InferenceLayer` class that executes a pipeline of inference rule functions.

- **`layers.abstract`**: Evaluates `Fact` and `Relation` collections to produce high-level `AbstractConclusion` outputs. Provides `AbstractLayer` class that executes conclusion rule functions.

### Coordination

- **`models.hierarchical_reasoner`**: Coordinates the tier execution order and exposes a composable factory (`build_reasoner`) for custom rule sets. Manages the full pipeline from observations to conclusions.

### Data Structures

- **`models.types`**: Defines shared data classes (`Fact`, `Relation`, `AbstractConclusion`) and enums (`FactType`) used across all tiers. Also includes tensor-compatible types (`FactEmbedding`, `RelationEmbedding`, `ConclusionEmbedding`) for neural processing.

## Neural Network Modules

### Neural Layers

- **`models.neural.sensory_encoder`**: Neural encoder that converts observations to dense fact embeddings using transformer-based text encoders (default: DistilBERT). Handles text encoding, fact type embedding, and optional metadata encoding.

- **`models.neural.relation_network`**: Graph Neural Network (GCN) that learns relations between facts. Constructs fully connected graphs, applies graph convolutions, and predicts relation predicates and confidence scores.

- **`models.neural.abstract_reasoner`**: Neural reasoner that generates conclusions using multi-head attention. Aggregates information from facts and relations, classifies themes, generates summaries, and selects supporting relations.

### Neural Coordination

- **`models.neural.hierarchical_neural`**: Coordinates the three neural layers into a complete pipeline. Provides `HierarchicalNeuralReasoner` class and `build_neural_reasoner` factory function. Handles device management, training/eval mode switching, and optional reconstruction of dataclass objects.

## Data Pipeline Modules

- **`data.dataset`**: PyTorch `Dataset` implementation (`HRMDataset`) for loading and preprocessing observations with optional ground truth facts, relations, and conclusions. Includes collate function for variable-length sequences.

- **`data.transforms`**: Data transformation utilities (`ObservationTransform`) for preprocessing observations before neural processing. Handles normalization, padding, and encoding of observation data.

## Training Modules

- **`training.train`**: Training script with config management, multi-task loss functions, optimizer setup, validation loops, checkpointing, and TensorBoard logging. Includes `HRMLoss` class for combining fact, relation, and conclusion losses.

- **`training.config.yaml`**: YAML configuration file for training hyperparameters, model architecture, data settings, and logging options.

## Example Modules

- **`examples.car_accident`**: Provides a runnable scenario (rule-based) illustrating how the tiers collaborate on a real-world car accident incident. Demonstrates causal chain rules, obligation breach detection, and legal/ethical conclusion generation.

- **`examples.car_accident_neural`**: Neural network version of the car accident scenario, demonstrating the neural pipeline with the same observations but using learned embeddings and relations.

## Test Modules

- **`tests.test_neural_models`**: Unit tests for all neural components including `SensoryEncoder`, `RelationNetwork`, `AbstractReasoner`, and `HierarchicalNeuralReasoner`. Includes integration test comparing rule-based and neural approaches.

## Module Dependencies

``` txt
Rule-Based:
  models.types
    ↑
  layers.sensory, layers.inference, layers.abstract
    ↑
  models.hierarchical_reasoner
    ↑
  examples.car_accident

Neural:
  models.types
    ↑
  models.neural.sensory_encoder, relation_network, abstract_reasoner
    ↑
  models.neural.hierarchical_neural
    ↑
  data.dataset, data.transforms
    ↑
  training.train
    ↑
  examples.car_accident_neural
```

## Extension Points

Each module is designed for extension:

- **Sensory layers**: Add custom resolvers or encoders for specialized data formats
- **Inference layers**: Add new rule functions or neural architectures
- **Abstract layers**: Add new conclusion rules or attention mechanisms
- **Data pipeline**: Add custom transforms or dataset classes
- **Training**: Customize loss functions, optimizers, or training loops
