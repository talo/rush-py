# NN-xTB Energy and Forces Example

Demonstrates how to run an NN-xTB energy and forces calculation using rush-py and inspect the results.

## Quick Start

```bash
export RUSH_TOKEN="your-token"
export RUSH_PROJECT="your-project"

python 07_nnxtb_energy.py
```

## What This Example Covers

1. Running an NN-xTB calculation with `nnxtb()` and `compute_forces=True`
2. Fetching the parsed result with `fetch_outputs`
3. Printing energy (in meV, eV, and kcal/mol) and per-atom forces

## Input Data

- `1kuw_t.json` — Small protein system in TRC topology format

## Tutorial

See the full tutorial: [NN-xTB Energy and Forces](../../docs/tutorials/07-nnxtb-energy.md)
