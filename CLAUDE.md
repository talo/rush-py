# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rush-py is the Python client library for QDX's Rush platform — a distributed computational chemistry and ML service. It wraps Rush's GraphQL API and generates REX DSL (a functional language) for job specification, providing domain-specific abstractions for molecular structures.

## Commands

Note that some developers will use the provided nix flake to enter a devshell before starting Claude. If this is the case, Claude should use the below commands unprefixed with `uv run`, and if a devshell refresh is required, should either request that the developer do so.

```bash
# Install dependencies
uv sync --dev

# Run all tests (requires .env with RUSH_TOKEN, RUSH_ENDPOINT, RUSH_PROJECT)
uv run pytest                          # auto-skips slow tests if queues busy
uv run pytest -m "not slow"            # fast tests only (converters, merge, fetch)
uv run pytest -m slow --force-run-slow # slow tests only, ignore queue status
uv run pytest --force-run-slow         # all tests, ignore queue status

# Via run_tests.sh (sources .env automatically)
./run_tests.sh              # auto-skip slow if queues busy
./run_tests.sh --quick      # fast tests only
./run_tests.sh --slow       # slow tests only, force run
./run_tests.sh --all        # all tests, force run

# Run a single test
uv run pytest tests/test_exess_energy.py
uv run pytest tests/test_exess_energy.py::test_name

# Linting / formatting
uv run ruff check
uv run ruff format

# Type checking
uv run ty check

# Build docs
cd docs && uv run sphinx-build -b html . _build/html
```

Tests hit the live Rush staging API — they require valid credentials in `.env` or environment variables (`RUSH_TOKEN`, `RUSH_ENDPOINT`, `RUSH_PROJECT`). Slow tests (any that submit jobs) are auto-marked and skipped when Rush queues have >2 queued/admitted jobs. Per-test timeout: 300s (fast), 600s (slow).

## Architecture

**Core data model (`mol.py`):** `Topology` (atoms, bonds, geometry, charges) + `Residues` + `Chains` = `TRC` — the central structure passed through all workflows.

**Client layer (`client.py`):** GraphQL client that handles authentication, REX DSL code generation, run submission/polling with exponential backoff, and file upload/download via object storage. Reads credentials from `.env` files (project dir or `~/.rush/.env`).

**Computation modules** each follow the same pattern — accept a `TRC` or file input, build REX DSL code via the client, submit to the Rush API, poll for results, and return parsed output:
- `exess` subpackage — Quantum chemistry (energy, optimization, QMMM, CHELPG, interaction energy)
- `nnxtb.py` — Tight-binding quantum chemistry
- `boltz.py` — Protein folding
- `auto3d.py` — Conformation generation
- `pbsa.py` — Solvation energy (Poisson-Boltzmann)
- `prepare` subpackage — Structure preparation
- `mmseqs2.py` — Sequence search

**File format converters (`convert/`):** Read/write PDB, mmCIF, SDF, JSON with auto-detection via `load_structure()`/`save_structure()`.

**FRIED module (`fried/`):** Fragment-based interaction energy decomposition — fragments ligands, runs EXESS on fragments, and visualizes contributions.

## Key Conventions

- Python >= 3.12, managed with `uv`
- Build backend: `uv_build`
- Formatting and linting: `ruff`
- Type checking: `ty`
- Source layout: `src/rush/` (installed as `rush` package)
