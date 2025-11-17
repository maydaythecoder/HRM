# HRM

Hierarchical Reasoning Model (HRM) is a modular Python sandbox for experimenting with layered reasoning systems. It mirrors the multi-tier design of the official Sapient HRM release while remaining lightweight enough for rapid iteration.

## Overview

HRM executes structured reasoning in three tiers:

- **Sensory tier** converts raw observations into immutable facts with traceable metadata.
- **Inference tier** applies rule-driven logic to build causal and compliance relations between facts.
- **Abstract tier** interprets relations to produce legal, ethical, or strategic conclusions.

The design emphasises composability, so each tier can be swapped or extended without refactoring the rest of the stack.

## Key Features

- **Dual Implementation**: Both rule-based and neural network implementations available
- **Typed data contracts** for facts, relations, and conclusions to keep reasoning stages decoupled yet compatible.
- **Rule pipelines** that make it easy to add or reorder inference and abstraction logic.
- **Neural architectures** using transformers, GNNs, and attention mechanisms for learned reasoning.
- **Executable demos** showcasing end-to-end reasoning on real-world inspired scenarios such as a car accident.
- **Documentation set** describing architecture, modules, and reasoning workflows for fast onboarding.

## Environment Setup

1. Install Python 3.11 or later to leverage dataclass `slots`.
2. Create and activate a virtual environment:
   - `python3.11 -m venv venv`
   - `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies:
   - For rule-based system: `pip install types-dataclasses` (optional, for static analysis)
   - For neural system: `pip install -r requirements.txt`

## Usage

### Quick Start

Run the bundled car accident scenario to see the hierarchical pipeline in action.

```bash
cd HRM
python3.11 -m venv venv
./venv/bin/python -m examples.car_accident
```

The script prints the extracted facts, inferred relations, and high-level conclusions. Modify the observations or swap rule functions to explore alternative reasoning behaviours.

### Neural Network Usage

Run the neural network implementation:

```bash
python -m examples.car_accident_neural
```

Or use the neural reasoner programmatically:

```python
from models.neural.hierarchical_neural import build_neural_reasoner

reasoner = build_neural_reasoner(
    embedding_dim=256,
    hidden_dim=128,
    num_predicate_classes=10,
    num_theme_classes=5,
)

result = reasoner.analyze(observations, reconstruct_objects=True)
```

### Training

Train the neural model with your own data:

```bash
python training/train.py --config training/config.yaml
```

## Project Structure

```txt
layers/
  sensory.py      # Tier 1: observation -> Fact (rule-based)
  inference.py    # Tier 2: facts -> Relation via rules
  abstract.py     # Tier 3: facts + relations -> AbstractConclusion

models/
  types.py                 # Shared data classes and enums
  hierarchical_reasoner.py # Rule-based coordinator
  neural/
    sensory_encoder.py     # Neural fact extraction
    relation_network.py    # GNN-based relation learning
    abstract_reasoner.py   # Attention-based conclusion generation
    hierarchical_neural.py # Neural coordinator

data/
  dataset.py      # PyTorch Dataset implementation
  transforms.py   # Data preprocessing utilities

training/
  train.py        # Training script
  config.yaml     # Training configuration

examples/
  car_accident.py  # End-to-end demo scenario (rule-based)

tests/
  test_neural_models.py  # Unit tests for neural components

docs/
  architecture.md      # Tier overview and data flow diagrams
  neural-architecture.md  # Neural network architecture details
  modules.md           # Module responsibilities
  reasoning-guide.md   # Workflow for running and extending the system

learning/
  overview.md   # Curriculum-style extension notes and future experiments

requirements.txt  # Python dependencies
.gitignore
README.md
```

## Documentation

- `docs/architecture.md` explains how information flows across tiers (rule-based).
- `docs/neural-architecture.md` describes the neural network implementation with transformers, GNNs, and attention.
- `docs/modules.md` itemises responsibilities of each module.
- `docs/reasoning-guide.md` provides a step-by-step playbook for extending rules and analysing outputs.
- `learning/overview.md` captures ideas for curriculum-based enhancements and progressive reasoning training.

Refer to these guides when customising rule sets or porting the pipeline to new domains such as supply-chain analysis or medical triage.

## Implementation Approaches

### Rule-Based System

The original rule-based implementation uses hand-coded logic for each tier:
- **Sensory**: Dictionary-based fact extraction
- **Inference**: Python functions that implement inference rules
- **Abstract**: Python functions that generate conclusions

Best for: Interpretability, deterministic outputs, scenarios with clear logical rules.

### Neural Network System

The neural implementation uses deep learning models:
- **Sensory**: Transformer-based text encoders (DistilBERT)
- **Inference**: Graph Neural Networks (GCN) for relation learning
- **Abstract**: Multi-head attention for conclusion generation

Best for: Learning from data, handling ambiguity, scaling to large datasets.

Both implementations share the same data structures and can be used interchangeably.

## Contributing

Contributions are welcome. Focus areas include:

- Adding new sensory resolvers for specialised data formats.
- Implementing additional inference or abstraction rules (risk analysis, compliance auditing, etc.).
- Enhancing neural architectures (new encoder models, attention mechanisms, etc.).
- Automating curriculum evaluation in the `learning/` package.
- Enhancing documentation with new examples or visualisations.

Open a pull request or start a discussion to coordinate enhancements.
