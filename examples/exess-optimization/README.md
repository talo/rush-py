# EXESS Geometry Optimization Example

Demonstrates QM geometry optimization with EXESS, including how to work with the optimization trajectory output.

## Quick Start

```bash
export RUSH_TOKEN="your-token"
export RUSH_PROJECT="your-project"

python 04_exess_optimization.py
```

## What This Example Covers

1. QM geometry optimization with `exess_geo_opt`
2. Using `OptimizationKeywords` (Cartesian coordinates, LBFGS algorithm)
3. Extracting trajectory and energy/gradient info from the output

## Input Data

- `ethene_twisted_t.json` — twisted ethene (C₂H₄) with 90° dihedral angle, optimizes to planar geometry

## Tutorial

See the full tutorial: [EXESS Geometry Optimization](../../docs/tutorials/exess-optimization.md)
