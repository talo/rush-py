# smol-similarity nearest-neighbor example

Demonstrates how to run `smol_similarity.smol_similarity_sumo()` using object-backed JSON inputs and inspect ranked similarity matches.

## Quick Start

```bash
export RUSH_TOKEN="your-token"
export RUSH_PROJECT="your-project"

python 08_smol_similarity_sumo.py
```

## What This Example Covers

1. Submitting a `smol_similarity_sumo` run with partition and query object files
2. Fetching parsed per-query outputs
3. Printing ranked SMILES matches and tanimoto similarities

## Input Data

- `data/input_smis.json` — Query SMILES list
- `data/smi_partition1.json` — Library partition SMILES list

## Tutorial

See the full tutorial: [smol-similarity nearest-neighbor search](../../docs/tutorials/08-smol-similarity-smol_similarity_sumo.md)
