# Hierarchical Reasoning Prototype

Modular Python implementation of a three-tier reasoning stack inspired by the HRM architecture. The system ingests raw observations, infers causal relationships, and derives high-level conclusions for scenarios such as traffic incidents.

## Project Layout

- `layers/` — Sensory, inference, and abstract reasoning tiers.
- `models/` — Coordination logic and shared data types.
- `examples/` — Executable scenarios (`car_accident.py`).
- `docs/` — Architecture, module, and reasoning guides.
- `learning/` — Notes for curriculum-oriented extensions.

## Quick Start

```bash
cd HRM
python3.11 -m venv venv
./venv/bin/python -m examples.car_accident
```

Expected output lists the extracted facts, inferred relations, and legal/ethical conclusions for the demo event. Update rules or observations to explore new domains.
