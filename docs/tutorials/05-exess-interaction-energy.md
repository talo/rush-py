# Tutorial 5: EXESS Interaction Energy

**What you get:** A single number (in Hartrees) quantifying how strongly a ligand interacts with its protein pocket — computed from first principles, no force field fitting required.

| | |
|---|---|
| **Time** | ~2–5 minutes (cloud compute) |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.10+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Why This Matters

In structure-based drug design, you constantly ask: *how tightly does this molecule sit in the binding pocket?* Docking scores approximate this with empirical potentials. Molecular mechanics (MM-GBSA, MM-PBSA) improve on it. But quantum mechanics gives you the most physically rigorous answer — it captures polarization, charge transfer, and dispersion effects that classical methods miss.

The catch? QM on an entire protein is impossible. EXESS solves this with **fragment-based quantum mechanics**: it breaks your system into amino-acid-sized pieces, computes QM energies for each piece and their pairwise/three-body interactions, then assembles the result. You get a QM-quality interaction energy without needing a supercomputer.

:::{admonition} What this is (and isn't)
:class: note
This interaction energy is **not** a binding free energy — it doesn't include entropy, solvation, or conformational sampling. Think of it as a high-quality electronic interaction energy for a single pose. It's most useful for **rank-ordering** poses or comparing ligands in the same pocket.
:::

## Quick Start: Get an Interaction Energy in 10 Lines

```python
import json
from rush import exess
from rush.client import RunOpts, download_object

out = exess.interaction_energy(
    "tyk2_ejm_31_t.json",       # QDX Topology file for TYK2 + ligand EJM-31
    93,                          # Fragment index of the ligand
    method="RestrictedHF",       # Explicit: minimal method for testing
    basis="STO-3G",              # Explicit: minimal basis for testing
    frag_keywords=exess.FragKeywords(
        level="Trimer",          # Include up to 3-body interactions
        dimer_cutoff=10.0,       # Å — pairs within this distance
        trimer_cutoff=5.0,       # Å — trimers within this distance
        cutoff_type="Centroid",
        distance_metric="Min",
    ),
    run_opts=RunOpts(name="Tutorial: Interaction Energy"),
    collect=True,
)

# Extract the result
json_bytes = download_object(out[0]["path"])
result = json.loads(json_bytes.decode())
print(f"Interaction energy: {result['qmmbe']['expanded_hf_energy']} Eh")
```

**Example output:**
```
Interaction energy: -0.08234 Eh
```

That negative value means the ligand is stabilized by its protein environment. The magnitude tells you the strength — more negative = stronger interaction.

:::{admonition} About the default method
:class: warning
By default, EXESS uses **RestrictedHF/STO-3G** — the simplest possible quantum chemistry method with a minimal basis set. This runs fast and is great for testing your workflow, but the absolute energies are not quantitatively meaningful. For production work, use at least `method="RestrictedHF", basis="cc-pVDZ"` or a correlated method like RI-MP2. The *relative* trends (which ligand binds more tightly) are often preserved even at low levels of theory.
:::

### Where to get the input file

The `tyk2_ejm_31_t.json` topology file is in the rush-py repo's [`tests/data/`](https://github.com/talo/rush-py/tree/main/tests/data) folder. It's a QDX Topology — a JSON format that encodes atomic coordinates, fragment definitions, charges, and connectivity.

---

## Understanding the Parameters

| Parameter | What it controls | Rule of thumb |
|---|---|---|
| `93` (fragment index) | Which fragment is your ligand | Check your topology to find the right index |
| `level="Trimer"` | Include 3-body corrections | More accurate but slower; `"Dimer"` is faster |
| `dimer_cutoff=10.0` | How far out to include fragment pairs (Å) | 10 Å captures most important interactions; increase for charged systems |
| `trimer_cutoff=5.0` | How far out for 3-body terms (Å) | Keep smaller than dimer_cutoff; 3-body terms decay faster |
| `included_fragments` | Restrict which fragments participate | Use to limit to the binding pocket (faster, often sufficient) |

---

## End-to-End: From PDB to Interaction Energy

Don't have a QDX Topology file? Start from a PDB. Rush's **Prepare Complex** module handles protonation, missing atoms, and charge assignment automatically.

### Step 1: Prepare the system

```python
import json
from pathlib import Path
from rush.client import RunOpts
from rush.prepare_complex import prepare_complex

trc = prepare_complex(
    Path("1hsg.pdb"),
    ligand_names=["MK1", "HOH"],  # Residue names for ligands/waters
    run_opts=RunOpts(name="Tutorial: Prepare Complex"),
    collect=True,
)

# Inspect what Prepare Complex determined about charges
print("Charged residues:")
for i, (name, charge) in enumerate(
    zip(trc.residues.seqs, trc.topology.fragment_formal_charges)
):
    if int(charge) != 0:
        print(f"  {i:>4} {name}: {int(charge):+}")
```

:::{admonition} What Prepare Complex does under the hood
:class: tip
It uses **PDBFixer** to fill missing atoms/residues, **PDB2PQR** for protonation states, and **RDKit** for ligand processing. The output TRC (Topology + Residues + Coordinates) is fully annotated with connectivity and formal charges — everything EXESS needs.
:::

### Step 2: Focus on the binding pocket

Running QM on the entire protein is wasteful. Most residues far from the ligand contribute negligibly to the interaction energy. Use `get_fragments_near_fragment` to select just the pocket:

```python
# Find fragments within 5 Å of the ligand
lig_idx = trc.residues.seqs.index("MK1")
pocket_frags = trc.topology.get_fragments_near_fragment(lig_idx, 5.0) + [lig_idx]
print(f"Pocket contains {len(pocket_frags)} fragments (out of {len(trc.residues.seqs)} total)")
```

### Step 3: Run the interaction energy calculation

```python
from rush import exess

# EXESS needs the Topology (not the full TRC)
with open("1hsg_t.json", "w") as f:
    json.dump(trc.topology.to_json(), f, indent=2)

out = exess.interaction_energy(
    "1hsg_t.json",
    lig_idx,
    frag_keywords=exess.FragKeywords(
        level="Dimer",
        included_fragments=pocket_frags,
    ),
    run_opts=RunOpts(name="Tutorial: Interaction Energy E2E"),
    collect=True,
)

json_bytes = download_object(out[0]["path"])
result = json.loads(json_bytes.decode())
print(f"Interaction energy: {result['qmmbe']['expanded_hf_energy']} Eh")
```

:::{admonition} The second output
:class: note
`exess.interaction_energy` returns two outputs. The first is the JSON with energies. The second contains additional exported data (density matrices, etc.) — only populated if you request exports. See the {doc}`Exports tutorial<03-exess-exports>` for details.
:::

---

## Interpreting the Results

The key value is `result['qmmbe']['expanded_hf_energy']`:

- **Negative** → ligand is stabilized by the protein (favorable interaction)
- **More negative** → stronger interaction
- **Units are Hartrees** (1 Eh ≈ 627.5 kcal/mol)

For comparing two ligands in the same pocket, the *difference* in interaction energies is more meaningful than the absolute values — especially at lower levels of theory.

---

## Try It Yourself

The complete example script and data files are available in the rush-py repository:

- **Example script:** [examples/exess-interaction-energy/05_exess_interaction_energy.py](https://github.com/talo/rush-py/blob/main/examples/exess-interaction-energy/05_exess_interaction_energy.py)
- **Data folder:** [examples/exess-interaction-energy/data/](https://github.com/talo/rush-py/tree/main/examples/exess-interaction-energy/data)

You can download these files directly or clone the repository to run the full example.

---

## See Also

- {doc}`Exports tutorial<03-exess-exports>` — extract electron density, ESP, and other properties
- {doc}`Optimization tutorial<04-exess-optimization>` — relax geometries before computing interaction energies
- {doc}`QM/MM tutorial<06-exess-qmmm>` — run dynamics simulations with mixed QM/MM
- [Full example script](https://github.com/talo/rush-py/tree/feat/examples-from-docs/examples/exess-interaction-energy) — runnable version with both examples
