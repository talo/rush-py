# Tutorial 2: EXESS Single Point Energy

**What you get:** A single-point energy calculation from first principles — the foundational QM calculation that determines the total electronic energy of a molecule at a fixed geometry.

| | |
|---|---|
| **Time** | ~30 seconds |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.10+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Why This Matters

A single-point energy (SPE) calculation is the most fundamental operation in quantum chemistry. Given a set of atomic coordinates, it solves the electronic Schrödinger equation and returns:

- **Total energy** — a single number (in Hartrees) representing the electronic + nuclear repulsion energy
- **Electronic properties** — molecular orbital energies and optional charges (Mulliken, CHELPG)

SPE calculations are the building block for everything else:

- **Benchmarking** — compare methods/basis sets on the same geometry
- **Energy differences** — reaction energies, conformational preferences, binding energies
- **Property extraction** — dipole moments, charges, densities (see {doc}`03-exess-exports`)
- **Starting point** — geometry optimizations and dynamics are sequences of SPE calculations under the hood

Because there's no geometry optimization loop, SPE is fast — you get your answer in seconds for small molecules.

---

## Quick Start

```python
from rush import exess
from rush.client import RunOpts

# Run a single-point energy calculation on water
res = exess.energy(
    "water_topology.json",
    method="RestrictedHF",
    basis="STO-3G",  # Minimal basis set for tutorials only – use cc-pVDZ or larger for production
    run_opts=RunOpts(
        name="Tutorial: Single Point Energy",
        tags=["rush-py", "tutorial", "exess", "spe"],
    ),
    collect=True,
)

# Save outputs and extract energy
files = exess.save_energy_outputs(res)
```

That's it — `res` contains the energy result, and `files` has the saved JSON output on disk.

> ⚠️ **Tutorial Basis Set Warning**
>
> This tutorial uses **STO-3G**, a minimal basis set chosen purely for speed so examples
> run quickly. **STO-3G is not suitable for research or production calculations.** For
> meaningful results, use at least `cc-pVDZ` or larger. See the electronic structure
> methods reference for available basis sets.

### Input File

The input is a TRC (topology representation) JSON file with atomic coordinates and element symbols. See {doc}`../exess/topologies` for the full TRC format specification. You can also convert from PDB using `rush.convert.pdb.from_pdb()` (see {doc}`01-exess-chelpg` for an example).

---

## Reading the Output

The JSON output contains the total energy in Hartrees (atomic units). Common conversions:

| Unit | Conversion |
|---|---|
| 1 Hartree | 627.509 kcal/mol |
| 1 Hartree | 2625.500 kJ/mol |
| 1 Hartree | 27.211 eV |

---

## Notes

- **Basis set choice** — This tutorial uses `STO-3G` for speed, but for real results use `cc-pVDZ` or larger. `RestrictedHF/STO-3G` is fast but low accuracy — good for learning and testing.
- **Default parameters** — `exess.energy()` uses sensible SCF defaults (convergence threshold, DIIS history, etc.). See the [`SCFKeywords` class](https://github.com/talo/rush-py/blob/main/src/rush/exess.py#L190){target="_blank"} in `exess.py` for the complete list of defaults.
- **No fragmentation** — for a small molecule like water, the whole system is treated in one calculation. For larger systems, see fragmentation in {doc}`05-exess-interaction-energy`
- **Geometry matters** — SPE gives you the energy at *exactly* the coordinates you provide. If the geometry is unrealistic, the energy will be too. Use {doc}`04-exess-optimization` first if you need a relaxed structure

---

## Try It Yourself

The complete example is available in the rush-py repository:

👉 **[examples/exess-spe/](https://github.com/talo/rush-py/tree/main/examples/exess-spe){target="_blank"}**

You can download the files directly or clone the repository to run the full example.

---

## See Also

- {doc}`CHELPG charges <01-exess-chelpg>` — compute partial charges from the wavefunction
- {doc}`Exporting properties <03-exess-exports>` — extract electron density, ESP, and other properties from an energy calculation
- {doc}`Geometry optimization <04-exess-optimization>` — find the minimum-energy structure before computing properties
- {doc}`EXESS Documentation <../exess/index>`
- [Full example script](https://github.com/talo/rush-py/tree/main/examples/exess-spe){target="_blank"}
