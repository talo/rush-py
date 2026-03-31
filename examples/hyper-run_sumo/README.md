# Hyper Run Sumo Example

Demonstrates `hyper.hyper_run_sumo()` for a short MD run using prebuilt simulation config and topology artifacts.

## Quick Start

```bash
export RUSH_TOKEN="your-token"
export RUSH_PROJECT="your-project"

python 10_hyper_run_sumo.py
```

## What This Example Covers

1. Submitting one `RunInput` job with explicit `HyperRunConfig`
2. Fetching trajectory/checkpoint payloads as bytes
3. Saving run artifacts to the Rush workspace

## Input Data

- `data/sim_config.json`
- `data/methanol_topology.json`
- `data/methanol_trc.json`

## Tutorial

See: [Tutorial 10: Hyper Run Sumo](../../docs/tutorials/10-hyper-run_sumo.md)
