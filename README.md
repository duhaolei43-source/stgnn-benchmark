# stgnn-benchmark

Benchmark for spatio-temporal traffic forecasting under dynamic graphs and structural shifts.

Focus:
- DDS (Diurnal Dependency Shift): time-of-day graphs A^(k)
- SPR (Structural Perturbation Robustness):
  - SCS: sensor configuration shift (feature masking)
  - TMC: topology mismatch/corruption (adjacency perturbations)
- Measurement of models' Adaptability and Generalization ability.
  - Unified Adaptability Score

Settings:
- Graph visibility: G (given) / L (learned) / N (no-graph)
- Adaptation: zero-shot / light adaptation / retrain

Reproduction:
- Official runs via Colab notebooks in `colab/`.
