## Summary (What changed)
- 

## Motivation (Why)
- 

## Scope
- [ ] DDS (Diurnal Dependency Shift)
- [ ] SCS (Sensor-Configuration Shift)
- [ ] TMC (Topology Mismatch/Corruption)
- [ ] Metrics / Adaptability Score (AS)
- [ ] Protocol / Docs
- [ ] Tests / CI
- [ ] Refactor / Chore

## How to test (Reproduce)
Provide the minimal commands/steps to verify this PR.

- [ ] Import sanity:
  - `python -c "import bench; print('ok')"`
- [ ] CLI sanity (if applicable):
  - `python -m bench.cli`
- [ ] Config used (if applicable):
  - `configs/...`

## Files changed
List the key files and what changed in each.
- 

## Protocol & reproducibility checklist
- [ ] No data leakage (graphs A^(k) built from TRAIN split only)
- [ ] Top-m constraint enforced for any adjacency used
- [ ] Seeds handled / deterministic where applicable
- [ ] Results/artifacts are NOT committed (respect .gitignore)

## Notes / Risks / TODO
- 

