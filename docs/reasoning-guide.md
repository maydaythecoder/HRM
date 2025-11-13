# Reasoning Guide

1. **Seed Observations**  
   Populate a list of observation dictionaries with `label`, `description`, `type`, and optional `metadata`.

2. **Configure Rules**  
   - Register causal rules in `InferenceLayer` to map facts into impact chains.
   - Register conclusion rules in `AbstractLayer` to interpret the derived relations.

3. **Run Pipeline**  
   Use `build_reasoner` to assemble the layers and invoke `analyze(observations)`.

4. **Inspect Outputs**  
   - `facts`: normalised observations with traceable metadata.
   - `relations`: causal and compliance links, optionally weighted.
   - `conclusions`: domain-specific interpretations and recommended actions.

5. **Extend**  
   Introduce additional rule functions for other domains (e.g., medical triage, supply-chain incidents) while reusing the existing data contracts.

