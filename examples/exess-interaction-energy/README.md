# EXESS Interaction Energy Example

Demonstrates fragment-based interaction energy calculations with EXESS, including an end-to-end pipeline from PDB preparation to interaction energy.

## Quick Start

```bash
export RUSH_TOKEN="your-token"
export RUSH_PROJECT="your-project"

python 04_exess_interaction_energy.py
```

## What This Example Covers

1. Computing interaction energy between a ligand fragment and its environment
2. Using `FragKeywords` to control fragmentation (Trimer level, cutoffs)
3. Preparing a complex from PDB using `prepare_complex`
4. Finding nearby fragments with `get_fragments_near_fragment`

## Input Data

- `tyk2_ejm_31_t.json` — from `examples/exess-interaction-energy/data/` (TYK2 protein-ligand complex topology)
- `1hsg.pdb` — from `examples/exess-interaction-energy/data/` (HIV protease complex, for end-to-end example)

## Tutorial

See the full tutorial: [EXESS Interaction Energy](../../docs/tutorials/exess-interaction-energy.md)
