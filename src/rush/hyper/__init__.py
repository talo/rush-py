"""Hyper module for the Rush Python client."""

# --- Shared result error type ---
from ._common import ItemError

# --- Solvation ---
from ._hyper_solvate_sumo import HyperConfig, TRCBatchResultRef, hyper_solvate_sumo

# --- Minimization ---
from ._hyper_minimize_sumo import (
    HyperMinimizeConfig,
    MinimizeInput,
    hyper_minimize_sumo,
)

# --- Molecular dynamics run ---
from ._hyper_run_sumo import (
    HyperRunConfig,
    RunInput,
    RunOutput,
    RunOutputPaths,
    RunOutputRef,
    RunResultRef,
    hyper_run_sumo,
)

__all__ = [
    # Shared
    "ItemError",
    # Solvation
    "HyperConfig",
    "TRCBatchResultRef",
    "hyper_solvate_sumo",
    # Minimization
    "HyperMinimizeConfig",
    "MinimizeInput",
    "hyper_minimize_sumo",
    # Molecular dynamics run
    "HyperRunConfig",
    "RunInput",
    "RunOutput",
    "RunOutputRef",
    "RunOutputPaths",
    "RunResultRef",
    "hyper_run_sumo",
]
