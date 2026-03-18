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

EXESS gives you two engines for this:
- **QM** — Hartree-Fock or MP2 (most accurate, slowest)
- **MM** — OpenMM classical force fields (fastest, least accurate for electronic properties)

---

## Quick Start: Optimize Twisted Ethene with QM

This example uses **ethene (C₂H₄)** with an intentionally twisted starting geometry — the two CH₂ groups are rotated 90° so the hydrogen atoms are perpendicular. During optimization, the molecule relaxes to a **planar** structure because the C=C π bond requires parallel p-orbitals for maximum overlap. It's a dramatic visual change (perpendicular → flat) that illustrates a fundamental concept in chemistry.

```python
from rush.exess_geo_opt import exess_geo_opt
from rush.client import RunOpts, save_object

out = exess_geo_opt(
    "ethene_twisted_t.json",
    100,  # Maximum optimization steps
    standard_orientation="None",  # Keep original frame of reference
    run_opts=RunOpts(name="Tutorial: QM Optimization"),
    collect=True,
)
```

That's it — EXESS will iteratively relax the twisted ethene geometry using the default method and basis set. You get back two outputs: the **trajectory** (one topology per step) and **step info** (energy + gradient at each step).

> ⚠️ **Tutorial Basis Set Warning**
>
> This tutorial [example code](https://github.com/talo/rush-py/tree/main/examples/exess-optimization){target="_blank"} uses **STO-3G**, a minimal basis set chosen purely for speed so examples
> run quickly. **STO-3G is not suitable for research or production calculations.** For
> meaningful results, use at least `cc-pVDZ` or larger. See the electronic structure
> methods reference for available basis sets.

:::{admonition} Why twisted ethene?
:class: tip
Ethene is the simplest molecule with a C=C double bond. Starting from a 90° twist forces the optimizer to recover the planar geometry — a direct consequence of π-bond stabilization. The energy drops significantly as the p-orbitals re-align, making it easy to see convergence in both the energy plot and the 3D structure.
:::

---

## Working with the Output

The optimization returns two objects:
1. **Trajectory** — a list of Topology objects, one per step (the geometry at each iteration)
2. **Step info** — a list of parsed step records with `total_energy` (Hartrees) and `max_gradient_component` (Å)

```python
from rush.exess_geo_opt import fetch_outputs, save_outputs

# Save the raw files if you want them on disk
saved_paths = save_outputs(out)

# Fetch parsed results into memory
result = fetch_outputs(out)
out_traj = result.trajectory
out_info = result.steps

# How quickly did it converge?
print(f"Converged in {len(out_traj)} steps")

# Energy at start vs. end
print(f"Initial energy: {out_info[0].total_energy:.8f} Eh")
print(f"Final energy:   {out_info[-1].total_energy:.8f} Eh")
print(f"Energy change:  {out_info[-1].total_energy - out_info[0].total_energy:.8f} Eh")

# How "flat" is the potential at the end?
print(f"Final max gradient: {out_info[-1].max_gradient_component:.2e} Å")

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

## Visualization

The [full example script](https://github.com/talo/rush-py/tree/main/examples/exess-optimization){target="_blank"} generates an interactive HTML report with:
- Energy convergence plot (energy vs. optimization step)
- Side-by-side 3D views of initial (twisted) and optimized (planar) structures
- Summary statistics (method, basis, steps, energy change)

### Example Output

Explore the energy convergence and before/after structures. Or run the code above to generate this yourself!

<iframe src="../_static/outputs/optimization_results.html" width="100%" height="600px" frameborder="0"></iframe>

---

## Try It Yourself

The complete example is available in the rush-py repository:

👉 **[examples/exess-optimization/](https://github.com/talo/rush-py/tree/main/examples/exess-optimization){target="_blank"}**

You can download the files directly or clone the repository to run the full example.

---

## See Also

- {doc}`Interaction energy tutorial<05-exess-interaction-energy>` — compute interaction energies on optimized structures
- {doc}`QM/MM tutorial<06-exess-qmmm>` — run molecular dynamics after optimization
- {doc}`Exports tutorial<03-exess-exports>` — extract electron density and ESP from optimized structures
- [Full example script](https://github.com/talo/rush-py/tree/main/examples/exess-optimization){target="_blank"}
