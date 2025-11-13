# Learning Extensions

This guide documents approaches for extending the hierarchical reasoning system
with curriculum-style training experiments.

## Curriculum Stages

- Define a stage name that captures the reasoning behaviour being trained.
- Prepare a curated set of observations that highlight the targeted behaviour.
- Decide which inference rules should be evaluated or introduced in the stage.

## Suggested Workflow

1. Draft stage metadata and sample observations in YAML or JSON for reuse.
2. Run the observations through the existing pipeline to capture baseline output.
3. Introduce new rules or modify metadata, then compare conclusions for drift.
4. Document results in `docs/reasoning-guide.md` to preserve institutional memory.

## Future Additions

- Automated rule discovery from labelled datasets.
- Curriculum scheduling utilities to evaluate incremental learning strategies.
- Benchmarks covering additional domains such as supply-chain or medical triage.
