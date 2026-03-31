# Hyper Solvation Example

Demonstrates how to run `hyper.hyper_solvate_sumo()` with rush-py and handle batched per-item outputs.

## Quick Start

```bash
export RUSH_TOKEN="your-token"
export RUSH_PROJECT="your-project"

python 08_hyper_solvate.py
```

## What This Example Covers

1. Submitting a Hyper solvation run with explicit `HyperConfig`
2. Collecting and fetching a `TRCBatchResultRef`
3. Handling `ItemError` per batch item
4. Saving successful TRC output to the Rush workspace

## Input Data

- `data/valid_trc.json` — Minimal TRC structure used as the solvation input

## Tutorial

See the full tutorial: [Hyper Solvation](../../docs/tutorials/08-hyper-solvate.md)
