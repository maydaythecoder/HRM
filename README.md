# HRM

Hierarchical Reasoning Model (HRM) is a modular Python sandbox for experimenting with layered reasoning systems. It mirrors the multi-tier design of the official Sapient HRM release while remaining lightweight enough for rapid iteration.

## Overview

HRM executes structured reasoning in three tiers:

- **Sensory tier** converts raw observations into immutable facts with traceable metadata.
- **Inference tier** applies rule-driven logic to build causal and compliance relations between facts.
- **Abstract tier** interprets relations to produce legal, ethical, or strategic conclusions.

The design emphasises composability, so each tier can be swapped or extended without refactoring the rest of the stack.

## Key Features

- **Typed data contracts** for facts, relations, and conclusions to keep reasoning stages decoupled yet compatible.
- **Rule pipelines** that make it easy to add or reorder inference and abstraction logic.
- **Executable demos** showcasing end-to-end reasoning on real-world inspired scenarios such as a car accident.
- **Documentation set** describing architecture, modules, and reasoning workflows for fast onboarding.

## Environment Setup

1. Install Python 3.11 or later to leverage dataclass `slots`.
2. Create and activate a virtual environment:
   - `python3.11 -m venv venv`
   - `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)
3. Install optional tooling as needed (e.g., `pip install types-dataclasses` for static analysis).

## Usage

### Quick Start

Run the bundled car accident scenario to see the hierarchical pipeline in action.

```bash
cd HRM
python3.11 -m venv venv
./venv/bin/python -m examples.car_accident
```

The script prints the extracted facts, inferred relations, and high-level conclusions. Modify the observations or swap rule functions to explore alternative reasoning behaviours.

## Project Structure

```txt
layers/
  sensory.py      # Tier 1: observation -> Fact
  inference.py    # Tier 2: facts -> Relation via rules
  abstract.py     # Tier 3: facts + relations -> AbstractConclusion

models/
  types.py                 # Shared data classes and enums
  hierarchical_reasoner.py # Coordinator and result snapshot utilities

examples/
  car_accident.py  # End-to-end demo scenario

docs/
  architecture.md      # Tier overview and data flow diagrams
  modules.md           # Module responsibilities
  reasoning-guide.md   # Workflow for running and extending the system

learning/
  overview.md   # Curriculum-style extension notes and future experiments

.gitignore
README.md
```

## Documentation

- `docs/architecture.md` explains how information flows across tiers.
- `docs/modules.md` itemises responsibilities of each module.
- `docs/reasoning-guide.md` provides a step-by-step playbook for extending rules and analysing outputs.
- `learning/overview.md` captures ideas for curriculum-based enhancements and progressive reasoning training.

Refer to these guides when customising rule sets or porting the pipeline to new domains such as supply-chain analysis or medical triage.

## Contributing

Contributions are welcome. Focus areas include:

- Adding new sensory resolvers for specialised data formats.
- Implementing additional inference or abstraction rules (risk analysis, compliance auditing, etc.).
- Automating curriculum evaluation in the `learning/` package.
- Enhancing documentation with new examples or visualisations.

Open a pull request or start a discussion to coordinate enhancements.
