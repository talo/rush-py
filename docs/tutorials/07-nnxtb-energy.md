# Tutorial 7: NN-xTB Energy and Forces

**What you get:** A fast, near-DFT-quality energy and per-atom forces for a molecular system — computed in seconds rather than minutes.

| | |
|---|---|
| **Time** | ~10 seconds |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.12+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Why This Matters

When screening hundreds or thousands of molecular structures — conformers, ligand poses, reaction intermediates — running full DFT on every one can be prohibitively expensive. You need a method that gives you quantum-level information (energies, forces) at a fraction of the cost.

NN-xTB fills this niche. It reparameterizes the GFN2-xTB tight-binding Hamiltonian with a neural network, achieving near-DFT accuracy while running orders of magnitude faster. This makes it practical to:

- **Rank** conformers or poses by energy before committing to DFT
- **Filter** out high-energy structures early in a pipeline
- **Compute forces** for large systems where DFT gradients would be too slow

In this tutorial, you will compute the NN-xTB energy and per-atom forces for a small protein system, then parse and inspect the results.

---

## Quick Start

```python
from rush import nnxtb
from rush.client import RunOpts

run = nnxtb.energy(
    "topology.json",
    compute_forces=True,
    run_opts=RunOpts(
        name="Tutorial: NN-xTB Energy",
        tags=["rush-py", "tutorial", "nnxtb"],
    ),
)
```

That's it — `run` is a `RushRun` handle for the JSON output with the energy and forces. Use `run.fetch()` when you want a parsed `nnxtb.Result` in memory, or `run.save()` if you want the raw JSON output file in the workspace.

### Input File

The input is a TRC (topology representation) JSON file with atomic coordinates and element symbols. This is the same format used by EXESS. See {doc}`../exess/topologies` for the full TRC format specification, or convert from PDB:

```python
import json
from pathlib import Path

from rush import from_pdb

trc = from_pdb(Path("molecule.pdb").read_text())
Path("molecule_t.json").write_text(json.dumps(trc.topology.to_json(), indent=2))
```

---

## Reading the Output

Fetch the parsed output directly with `run.fetch()`. It returns an
`nnxtb.Result` dataclass:

```python
# Parse into a structured result
result = run.fetch()

print(f"Energy: {result.energy_mev:.2f} meV")
```

`nnxtb.Result` has three fields:
- `energy_mev`
- `forces_mev_per_angstrom`
- `frequencies_inv_cm`

### Energy

The energy is returned in **millielectronvolts (meV)**, not Hartrees as in EXESS. Common conversions:

| Unit | Conversion |
|---|---|
| 1 eV | 1000 meV |
| 1 eV | 23.06 kcal/mol |
| 1 eV | 96.49 kJ/mol |
| 1 Hartree | 27211 meV |

### Forces

When `compute_forces=True` (the default), you get a list of (x, y, z) force vectors in meV/A, one per atom:

```python
if result.forces_mev_per_angstrom:
    for i, (fx, fy, fz) in enumerate(result.forces_mev_per_angstrom):
        magnitude = (fx**2 + fy**2 + fz**2) ** 0.5
        print(f"Atom {i}: ({fx:.2f}, {fy:.2f}, {fz:.2f}) meV/A  |F| = {magnitude:.2f}")
```

Large force magnitudes indicate atoms that are far from equilibrium — useful for identifying strained regions in a structure.

---

## Computing Vibrational Frequencies

NN-xTB can also compute vibrational frequencies, which are useful for thermochemistry corrections and IR spectra prediction. This takes more compute than energy/forces:

```python
result = nnxtb.energy(
    "topology.json",
    compute_forces=True,
    compute_frequencies=True,
    run_opts=RunOpts(name="Tutorial: NN-xTB Frequencies"),
).fetch()

if result.frequencies_inv_cm:
    print(f"Number of vibrational modes: {len(result.frequencies_inv_cm)}")
    print(f"Lowest frequency: {min(result.frequencies_inv_cm):.1f} cm^-1")
    print(f"Highest frequency: {max(result.frequencies_inv_cm):.1f} cm^-1")

    # Imaginary frequencies (negative values) indicate a saddle point
    imaginary = [f for f in result.frequencies_inv_cm if f < 0]
    if imaginary:
        print(f"Warning: {len(imaginary)} imaginary frequencies detected")
```

---

## Setting Charge and Multiplicity

For charged or open-shell systems, specify the spin multiplicity:

```python
# Doublet radical (multiplicity = 2)
run = nnxtb.energy(
    "radical_topology.json",
    multiplicity=2,
    run_opts=RunOpts(name="Tutorial: NN-xTB Doublet"),
)
```

The charge is currently read from the topology file. Multiplicity defaults to 1 (singlet).

---

## Batch Workflow: Asynchronous Submission

For screening workflows, submit many jobs asynchronously and collect results later:

```python
from rush import nnxtb
from rush.client import RunOpts

# Submit a batch of structures
topologies = ["conf_001.json", "conf_002.json", "conf_003.json"]
runs = []

for topo in topologies:
    run = nnxtb.energy(
        topo,
        compute_forces=False,   # Forces not needed for ranking
        run_opts=RunOpts(tags=["screening"]),
    )
    runs.append(run)

# Collect all results
for topo, run in zip(topologies, runs):
    result = run.fetch()
    print(f"{topo}: {result.energy_mev:.2f} meV")
```

---

## Notes

- **Default parameters** — Forces are computed by default; frequencies are not. For energy-only calculations (fastest), explicitly set `compute_forces=False`.
- **GPU resources** — The default `RunSpec(gpus=1, storage=100)` is sufficient for most systems. NN-xTB is designed to be efficient on a single GPU.
- **Input format** — NN-xTB uses the same TRC topology format as EXESS. Sample topologies are available in `tests/data/` in the rush-py repository.
- **Energy units** — NN-xTB reports energy in meV, not Hartrees. Keep this in mind when comparing with EXESS results.

---

## See Also

- {doc}`NN-xTB Overview <../nnxtb/overview>` — method details, supported features, and limitations
- {doc}`NN-xTB Running Reference <../nnxtb/running>` — full API reference and parameter documentation
- {doc}`EXESS Single Point Energy <02-exess-spe>` — for high-accuracy DFT/HF calculations
- {doc}`EXESS Topologies <../exess/topologies>` — input file format specification
