# Reasoning Guide

This guide provides step-by-step instructions for using both the rule-based and neural implementations of the Hierarchical Reasoning Model.

## Quick Start: Rule-Based System

1. **Seed Observations**  
   Populate a list of observation dictionaries with `label`, `description`, `type`, and optional `metadata`:

```python
observations = [
    {
        "label": "event_1",
        "description": "A collision occurred at the intersection",
        "type": "EVENT",
        "metadata": {"location": "5th Ave", "severity": "major"}
    },
    # ... more observations
]
```

2 **Configure Rules**

- Register causal rules in `InferenceLayer` to map facts into impact chains.
- Register conclusion rules in `AbstractLayer` to interpret the derived relations.

```python
from layers.inference import InferenceRule
from layers.abstract import ConclusionRule

def my_causal_rule(facts: Tuple[Fact, ...]) -> Iterable[Relation]:
    # Your inference logic here
    pass

def my_conclusion_rule(facts: Tuple[Fact, ...], relations: Tuple[Relation, ...]) -> Iterable[AbstractConclusion]:
    # Your conclusion logic here
    pass
```

3 **Run Pipeline**  
   Use `build_reasoner` to assemble the layers and invoke `analyze(observations)`:

```python
from models.hierarchical_reasoner import build_reasoner

reasoner = build_reasoner(
    inference_rules=(my_causal_rule,),
    conclusion_rules=(my_conclusion_rule,)
)

result = reasoner.analyze(observations)
```

4 **Inspect Outputs**  

- `facts`: normalised observations with traceable metadata.
- `relations`: causal and compliance links, optionally weighted.
- `conclusions`: domain-specific interpretations and recommended actions.

```python
for fact in result.facts:
    print(f"{fact.label}: {fact.description}")

for relation in result.relations:
    print(f"{relation.subject.label} {relation.predicate} {relation.obj.label}")

for conclusion in result.conclusions:
    print(f"[{conclusion.theme}] {conclusion.summary}")
```

5 **Extend**  
   Introduce additional rule functions for other domains (e.g., medical triage, supply-chain incidents) while reusing the existing data contracts.

## Quick Start: Neural Network System

1 **Prepare Observations**  
   Use the same observation format as the rule-based system:

```python
observations = [
    {
        "label": "event_1",
        "description": "A collision occurred at the intersection",
        "type": "EVENT",
        "metadata": {"location": "5th Ave", "severity": "major"}
    },
    # ... more observations
]
```

2 **Build Neural Reasoner**  
   Create a neural reasoner with desired architecture:

```python
from models.neural.hierarchical_neural import build_neural_reasoner

reasoner = build_neural_reasoner(
    embedding_dim=256,
    hidden_dim=128,
    num_predicate_classes=10,
    num_theme_classes=5,
    predicate_vocab=["caused", "impacted", "breached"],
    theme_vocab=["legal", "ethical", "policy"]
)
```

3 **Run Inference**  
   Analyze observations with the neural pipeline:

```python
result = reasoner.analyze(
    observations,
    relation_threshold=0.5,
    reconstruct_objects=True  # Convert embeddings back to dataclasses
)
```

4 **Inspect Outputs**  
   Access both embeddings and reconstructed objects:

```python
# Embeddings (tensors)
for fe in result.fact_embeddings:
    print(f"{fe.label}: embedding shape {fe.embedding.shape}")

# Reconstructed objects (if reconstruct_objects=True)
if result.facts:
    for fact in result.facts:
        print(f"{fact.label}: {fact.description}")
```

5 **Train Custom Model**  
   Train on your own data:

```python
# Prepare training data
from data.dataset import HRMDataset
from torch.utils.data import DataLoader

dataset = HRMDataset(
    observations=[...],  # Your training observations
    facts=[...],         # Optional ground truth
    relations=[...],     # Optional ground truth
    conclusions=[...]    # Optional ground truth
)

# Train
python training/train.py --config training/config.yaml
```

## Advanced Usage

### Hybrid Approach

Combine rule-based and neural components:

```python
# Use neural encoder for facts
from models.neural.sensory_encoder import SensoryEncoder
encoder = SensoryEncoder()
fact_embeddings = encoder.encode_facts(observations)

# Use rule-based inference
from models.hierarchical_reasoner import build_reasoner
rule_reasoner = build_reasoner()
# Convert embeddings back to facts, then use rules
```

### Custom Neural Architectures

Extend neural components:

```python
from models.neural.sensory_encoder import SensoryEncoder
import torch.nn as nn

class CustomSensoryEncoder(SensoryEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom layers
        self.custom_layer = nn.Linear(self.embedding_dim, 128)
```

### Domain-Specific Extensions

**Medical Triage:**

- Add medical fact types (SYMPTOM, DIAGNOSIS, TREATMENT)
- Create inference rules for symptom-diagnosis relations
- Generate triage priority conclusions

**Supply Chain:**

- Add supply chain fact types (SUPPLIER, PRODUCT, ORDER)
- Create inference rules for supply chain dependencies
- Generate risk assessment conclusions

**Legal Compliance:**

- Add compliance fact types (REGULATION, VIOLATION, PENALTY)
- Create inference rules for compliance checking
- Generate compliance report conclusions

## Best Practices

1. **Start with Rule-Based**: Use rule-based system to prototype and validate logic before training neural models.

2. **Data Preparation**: For neural training, ensure consistent observation formats and consider data augmentation.

3. **Hyperparameter Tuning**: Experiment with embedding dimensions, hidden sizes, and learning rates based on your data.

4. **Evaluation**: Compare rule-based and neural outputs on the same observations to validate neural model behavior.

5. **Interpretability**: Use attention weights and relation scores to understand neural model decisions.

6. **Incremental Training**: Start with pre-trained components (e.g., DistilBERT) and fine-tune for your domain.

## Troubleshooting

**Rule-Based Issues:**

- Check that observation types match `FactType` enum values
- Verify rule functions return correct types
- Ensure metadata keys are consistent

**Neural Issues:**

- Verify CUDA availability if using GPU: `torch.cuda.is_available()`
- Check embedding dimensions match across layers
- Ensure batch sizes are appropriate for your memory
- Monitor training loss for overfitting

**Integration Issues:**

- Verify tensor device consistency (CPU vs GPU)
- Check that vocabularies match model configuration
- Ensure observation formats are consistent

## Next Steps

- Review `docs/neural-architecture.md` for detailed neural architecture information
- Check `examples/car_accident.py` and `examples/car_accident_neural.py` for complete examples
- Explore `learning/overview.md` for curriculum-based training approaches
