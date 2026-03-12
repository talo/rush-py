# EXESS QM/MM Simulation Example

Demonstrates QM/MM molecular dynamics simulations with EXESS, including building a minimal system from scratch.

## Quick Start

```bash
export RUSH_TOKEN="your-token"
export RUSH_PROJECT="your-project"

python 06_exess_qmmm.py
```

## What This Example Covers

1. Basic QM/MM simulation with `exess.qmmm`
2. Manually constructing `Topology` and `Residues` for a two-water system
3. Configuring QM/MM fragment regions
4. Extracting and inspecting the simulation trajectory

## Input Data

- `6a5j_t.json`, `6a5j_r.json` — from `tests/data/` (protein complex)

## Tutorial

See the full tutorial: [EXESS QM/MM](../../docs/tutorials/exess-qmmm.md)
