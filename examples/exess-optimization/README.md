# EXESS Geometry Optimization Example

Demonstrates QM and ML geometry optimization with EXESS, including how to work with the optimization trajectory output.

## Quick Start

```bash
export RUSH_TOKEN="your-token"
export RUSH_PROJECT="your-project"

python 01_exess_optimization.py
```

## What This Example Covers

1. QM geometry optimization with `exess.optimization`
2. ML (AIMNet) optimization by setting `qm_fragments=[]` and `mm_fragments=[]`
3. Using `OptimizationKeywords` (Cartesian coordinates, LBFGS algorithm)
4. Extracting trajectory and energy/gradient info from the output

## Input Data

- `benzene_t.json` — from `tests/data/` (benzene topology)

## Tutorial

See the full tutorial: [EXESS Geometry Optimization](../../docs/tutorials/exess-optimization.md)
