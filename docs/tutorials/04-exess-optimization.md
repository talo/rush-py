# Tutorial 4: EXESS Geometry Optimization

**What you get:** Watch a molecule relax to its most stable geometry, step by step — with energies, gradients, and a full trajectory you can visualize.

| | |
|---|---|
| **Time** | ~1–5 minutes (cloud compute) |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.10+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Why This Matters

Molecules in the real world don't sit in arbitrary geometries — they settle into low-energy conformations. Before you compute any property (interaction energy, charges, spectra), you usually want the *optimized* geometry, because results from a strained or unrealistic structure are meaningless.

Geometry optimization iteratively adjusts atomic positions to minimize the total energy. At each step, EXESS computes the energy gradient (the "force" on each atom) and moves atoms downhill on the potential energy surface. When the gradient is small enough, the structure has converged.

EXESS gives you three engines for this:
- **QM** — Hartree-Fock or MP2 (most accurate, slowest)
- **ML** — AIMNet neural network potential (fast, good for organic molecules)
- **MM** — OpenMM classical force fields (fastest, least accurate for electronic properties)

---

## Quick Start: Optimize Twisted Ethene with QM

This example uses **ethene (C₂H₄)** with an intentionally twisted starting geometry — the two CH₂ groups are rotated 90° so the hydrogen atoms are perpendicular. During optimization, the molecule relaxes to a **planar** structure because the C=C π bond requires parallel p-orbitals for maximum overlap. It's a dramatic visual change (perpendicular → flat) that illustrates a fundamental concept in chemistry.

```python
from rush import exess
from rush.client import RunOpts, save_object

out = exess.optimization(
    "ethene_twisted_t.json",
    100,  # Maximum optimization steps
    standard_orientation="None",  # Keep original frame of reference
    run_opts=RunOpts(name="Tutorial: QM Optimization"),
    collect=True,
)
```

That's it — EXESS will iteratively relax the twisted ethene geometry using Restricted Hartree-Fock / STO-3G (the defaults). You get back two outputs: the **trajectory** (one topology per step) and **step info** (energy + gradient at each step).

:::{admonition} About the defaults
:class: warning
The default **RestrictedHF/STO-3G** is the fastest possible QM level but gives poor absolute geometries and energies. It's ideal for testing your workflow. For production, use at least `method="RestrictedHF", basis="cc-pVDZ"` or `method="RIHF"` with a larger basis. Bond lengths with STO-3G can be off by ~0.05 Å.
:::

:::{admonition} Why twisted ethene?
:class: tip
Ethene is the simplest molecule with a C=C double bond. Starting from a 90° twist forces the optimizer to recover the planar geometry — a direct consequence of π-bond stabilization. The energy drops significantly as the p-orbitals re-align, making it easy to see convergence in both the energy plot and the 3D structure.
:::

---

## Working with the Output

The optimization returns two objects:
1. **Trajectory** — a list of Topology objects, one per step (the geometry at each iteration)
2. **Step info** — a list of dictionaries with `total_energy` (Hartrees) and `max_gradient_component` (Å)

```python
import json
from rush import Topology

# Download both outputs
out_traj_path, out_info_path = [save_object(obj["path"]) for obj in out]
with open(out_traj_path) as f1, open(out_info_path) as f2:
    out_traj_raw, out_info = [json.load(f) for f in (f1, f2)]

out_traj = [Topology.from_json(t) for t in out_traj_raw]

# How quickly did it converge?
print(f"Converged in {len(out_traj)} steps")

# Energy at start vs. end
print(f"Initial energy: {out_info[0]['total_energy']:.8f} Eh")
print(f"Final energy:   {out_info[-1]['total_energy']:.8f} Eh")
print(f"Energy change:  {out_info[-1]['total_energy'] - out_info[0]['total_energy']:.8f} Eh")

# How "flat" is the potential at the end?
print(f"Final max gradient: {out_info[-1]['max_gradient_component']:.2e} Å")

# Compare atom positions
print(f"First atom — start: {out_traj[0].geometry[:3]}")
print(f"First atom — end:   {out_traj[-1].geometry[:3]}")
```

**What to look for:**
- Energy should **decrease** monotonically (if it doesn't, something is wrong)
- Max gradient should get very small (< 1e-4 Å for a tight optimization)
- Fewer steps = the initial geometry was already close to optimal
- For twisted ethene, the final structure should be **planar** — both CH₂ groups in the same plane

:::{tip}
Set `standard_orientation="None"` to prevent EXESS from translating or rotating the molecule. This makes it easy to overlay initial and final structures for visual comparison.
:::

---

## ML Optimization with AIMNet

For faster optimization (especially useful for conformer searches or large organic molecules), use the AIMNet machine-learning potential instead of QM:

```python
out = exess.optimization(
    "ethene_twisted_t.json",
    100,
    basis="STO-2G",  # Minimal basis — no effect on ML, just reduces memory overhead
    optimization_keywords=exess.OptimizationKeywords(
        coordinate_system="Cartesian",
        algorithm="LBFGS",
        lbfgs_keywords=exess.LBFGSKeywords(),
    ),
    standard_orientation="None",
    qm_fragments=[],   # No QM fragments
    mm_fragments=[],    # No MM fragments → everything is ML
    run_opts=RunOpts(name="Tutorial: ML Optimization"),
    collect=True,
)
```

:::{admonition} How fragment assignment works
:class: note
EXESS assigns each fragment to QM, MM, or ML. Setting `qm_fragments=[]` and `mm_fragments=[]` tells EXESS that *all* fragments should use ML (AIMNet). The `basis="STO-2G"` is a technicality — it's the smallest possible basis set and has no effect on the ML calculation, but EXESS still requires one to be specified.
:::

:::{admonition} ML optimization settings
:class: tip
For ML-only runs, you **must** use `coordinate_system="Cartesian"` and `algorithm="LBFGS"`. Other combinations may fail or give poor results.
:::

:::{caution}
The step-info output (energy, gradient) is only populated for **QM regions**. In a pure ML run, the dictionaries will be empty — you'll only get the trajectory.
:::

---

## Visualization

The [full example script](https://github.com/talo/rush-py/tree/feat/examples-from-docs/examples/exess-optimization) generates an interactive HTML report with:
- Energy convergence plot (energy vs. optimization step)
- Side-by-side 3D views of initial (twisted) and optimized (planar) structures
- Summary statistics (method, basis, steps, energy change)

### Example Output

Explore the energy convergence and before/after structures. Or run the code above to generate this yourself!

<iframe src="../_static/outputs/optimization_results.html" width="100%" height="600px" frameborder="0"></iframe>

---

## Try It Yourself

The complete example script and data files are available in the rush-py repository:

- **Example script:** [examples/exess-optimization/04_exess_optimization.py](https://github.com/talo/rush-py/blob/main/examples/exess-optimization/04_exess_optimization.py)
- **Data folder:** [examples/exess-optimization/data/](https://github.com/talo/rush-py/tree/main/examples/exess-optimization/data)

You can download these files directly or clone the repository to run the full example.

---

## See Also

- {doc}`Interaction energy tutorial<05-exess-interaction-energy>` — compute interaction energies on optimized structures
- {doc}`QM/MM tutorial<06-exess-qmmm>` — run molecular dynamics after optimization
- {doc}`Exports tutorial<03-exess-exports>` — extract electron density and ESP from optimized structures
- [Full example script](https://github.com/talo/rush-py/tree/feat/examples-from-docs/examples/exess-optimization)
