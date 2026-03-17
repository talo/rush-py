# Overview | [Use NN-xTB](https://nnxtb.qdx.co)

NN-xTB (Neural Network extended Tight Binding) is a GPU-accelerated semi-empirical quantum chemistry method built by QDX. It reparameterizes the GFN2-xTB Hamiltonian with a neural network to approach DFT-level accuracy while retaining the interpretability and speed of tight-binding methods.

NN-xTB fills the gap between fast but approximate classical force fields and accurate but expensive *ab initio* quantum chemistry. It is designed for workflows where you need quantum-level electronic structure information — energies, forces, vibrational frequencies — but cannot afford the cost of full DFT or wavefunction methods across large numbers of structures.

## When to use NN-xTB

NN-xTB is a good fit when you need:

- **Fast single-point energies** across large datasets (screening, filtering, ranking)
- **Per-atom forces** for downstream analysis or geometry validation
- **Vibrational frequencies** for thermochemistry or IR spectra prediction
- **Rapid turnaround** — calculations that would take minutes with DFT complete in seconds with NN-xTB

Typical use cases include:

- **Conformer generation & ranking** — generate and optimize thousands of conformers, rank by energy, and pass only the low-energy ones to DFT
- **Pre-optimization before DFT** — cheaply relax rough input geometries from crystal structures, docking, or generative models before handing to a more expensive method, saving many DFT cycles
- **Thermochemistry at scale** — obtain Hessians and frequencies for large datasets where full DFT would be prohibitive
- **Solvation & pKa estimation** — rapid pKa and solubility screening using implicit solvation models
- **Large and complex systems** — supramolecular complexes and other systems where DFT is impractical
- **Reaction pathway exploration** — finding reaction intermediates and mapping energy surfaces

**Rule of thumb:** Use NN-xTB for fast, approximate answers across many structures. Use EXESS when you need high-accuracy results on specific structures.

## How it works

NN-xTB extends the GFN2-xTB tight-binding method by replacing key empirical parameters with neural-network-predicted values. This retains the interpretable electronic structure of xTB (orbital energies, charge decomposition) while improving accuracy toward DFT-level results.

## Supported features

- **Energy**: Total electronic energy in meV
- **Forces**: Per-atom force vectors in meV/A (enabled by default)
- **Frequencies**: Vibrational frequencies in cm^-1 (optional, more expensive)
- **Charge**: Arbitrary molecular charge (integer)
- **Multiplicity**: Arbitrary spin multiplicity (singlet, doublet, triplet, etc.)

## Limitations

- NN-xTB is a semi-empirical method — for high-level *ab initio* accuracy, use EXESS instead

## Output format

NN-xTB returns a JSON object with three fields:

| Field | Type | Description |
|---|---|---|
| `energy_mev` | `float` | Total energy in millielectronvolts |
| `forces_mev_per_angstrom` | `list[tuple[float, float, float]]` or `null` | Per-atom force vectors (x, y, z) |
| `frequencies_inv_cm` | `list[float]` or `null` | Vibrational frequencies in cm^-1 |

Use `fetch_outputs()` to parse this output into a `NnxtbResult` dataclass:

```python
from rush.nnxtb import NnxtbResult, fetch_outputs

# After collecting a run result
results: NnxtbResult = fetch_outputs(res)

print(f"Energy: {results.energy_mev} meV")
print(f"Forces: {results.forces_mev_per_angstrom}")
print(f"Frequencies: {results.frequencies_inv_cm}")
```

`NnxtbResult` has three fields:
- `energy_mev`
- `forces_mev_per_angstrom`
- `frequencies_inv_cm`
