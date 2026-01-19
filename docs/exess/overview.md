# Overview

EXESS (EXtreme-scale Electronic Structure System, formerly HERMES) is a GPU-accelerated electronic structure program built for large-scale ab initio calculations. It is designed for workflows where GPU throughput, fragmentation, and multi-GPU scaling matter more than a broad feature matrix.

Key strengths at a glance:

- GPU-first HF/RI/MP2/KSDFT workflows tuned for accelerator throughput.
- Fragmentation (MBE) to make large systems tractable and scalable.
- Region-based QM/MM/ML partitioning via `keywords.regions`, including MM force fields (OpenMM) and ML potentials (AIMNet) in QMMM and optimization workflows.
- JSON-first inputs that support batched topologies and automation.
- Native fit for multi-GPU and multi-node runs via Rush and MPI.

This combination makes EXESS a strong fit when you need quantum mechanical energies or gradients on large molecular systems and want a workflow that scales across GPUs and nodes. Unlike traditional CPU-first packages (Gaussian, ORCA, Q-Chem) or plane-wave codes (CP2K), EXESS is optimized around GPU kernels and a fragment-first strategy, and it emphasizes a JSON input format that works well for automation and batched runs.

## What EXESS solves well

EXESS targets:

- Large molecular systems where full-system QM is too expensive.
- Fragment-based quantum chemistry via a Many-Body Expansion (MBE).
- GPU-accelerated energies, gradients, geometry optimization, and AIMD.
- Mixed-fidelity QM/MM/ML runs where fragments are assigned to regions (QMMM and optimization workflows).
- High-throughput runs where a single input describes multiple topologies.

If your workflow needs plane waves or pseudopotentials, EXESS is not the right tool (see limitations below). Thermostatted/barostatted dynamics are supported in QMMM workflows (NVT/NPT via `qmmm.temperature_kelvin` and `qmmm.pressure_atm`), but fully ab initio AIMD is microcanonical only.

## Core ideas and workflow

EXESS is built around four practical ideas:

1. **GPU-first execution.** Methods and kernels are designed to run efficiently on NVIDIA (CUDA) and AMD (HIP) GPUs.
2. **Fragmentation as a primary scaling strategy.** Many-Body Expansion enables accurate calculations on systems that are otherwise too large for full-system QM.
3. **Region-aware modeling.** The `regions` keyword assigns fragments to QM, MM, or ML partitions for QMMM and optimization workflows, with MM driven by OpenMM force fields and ML driven by AIMNet.
4. **Automation-friendly inputs.** The JSON schema supports batched topologies, which is useful for screening or dataset generation.

This philosophy shapes how you should approach method choice, fragmentation, and performance tuning.

## Methods and approximations (what is implemented)

### Hartree-Fock

EXESS supports RHF and UHF energies. Gradients are available for RHF but not UHF. Non-RI HF is limited to basis sets up to D functions, and spherical Gaussians are only supported for non-RI RHF.

### RI-HF and RI-MP2

RI is the main path to scalable HF and MP2 in EXESS:

- **RI-HF** is enabled by setting `scf.fock_build_type = "RI"` with `method = "RestrictedHF"`. RI-HF is RHF-only.
- **RI-MP2** is the supported MP2 implementation (`method = "RestrictedRIMP2"`). MP2 runs are RI by design.

RI reduces computational cost but increases memory use because integrals are stored. It also requires an auxiliary basis set. In return, RI-HF/RI-MP2 support higher angular momentum basis sets (up to F in the primary basis and G in the auxiliary basis).

### KSDFT

Restricted KSDFT is supported via LibXC functionals. LDA, GGA, meta-GGA, and hybrid functionals are available through LibXC, but upstream docs note that meta-GGA and double-hybrid functionals are experimental. Range-separated functionals are not supported. KSDFT gradients are not available in EXESS at present.

Upstream docs summarize method support as follows:

| Calculation | RHF | UHF | RI-HF | RI-MP2 | RestrictedKSDFT |
| --- | --- | --- | --- | --- | --- |
| Energy | yes | yes | yes | yes | yes |
| Gradient | yes | no | yes | yes | no |
| Fragmentation | yes | no | yes | yes | yes |

If gradients are enabled, fragmented gradients are enabled as well.

### Fragmentation theory (MBE)

EXESS implements a Many-Body Expansion (MBE) for non-covalent and covalent systems, with truncation up to fourth order (tetramers). The number of n-mers grows combinatorially with fragment count:

`C(n, k) = n! / (k! * (n - k)!)`

Covalent fragmentation uses hydrogen capping for broken single bonds. If you cut covalent bonds, provide `connectivity` so EXESS can cap and restore bonds correctly.

Upstream docs recommend the MBE reviews at https://doi.org/10.1063/1.5126216 and https://doi.org/10.1021/cr200093j for background and accuracy considerations.

### Gradients, dynamics, and optimization

- Geometry optimization is supported for RHF, RI-HF, and RI-MP2.
- Born-Oppenheimer AIMD uses a Verlet integrator and is microcanonical only (no thermostats or barostats).
- QMMM dynamics support NVT/NPT via the `qmmm` block (temperature and optional pressure), with MM handled through OpenMM and ML via AIMNet.
- Dynamics can be combined with fragmentation for large systems.
- Periodic boundary conditions and water-only classical solvent support are available in AIMD workflows.

## Capabilities and limitations

EXESS is strong when you need GPU performance and fragmentation at scale. It is not a general-purpose electronic structure code with every method or model.

**Strengths:**

- GPU-accelerated RHF, RI-HF, RI-MP2, and KSDFT energies.
- Large-system QM via MBE fragmentation.
- QM/MM/ML partitioning with `regions` for QMMM and optimization workflows.
- Multi-GPU scaling for fragmented runs.
- AIMD and geometry optimization with gradient-enabled methods.

**Limitations (from upstream docs):**

- Non-RI HF supports up to D functions.
- RI-HF/RI-MP2 support up to F in the primary basis and G in the auxiliary basis.
- Spherical Gaussians only for non-RI RHF.
- Hydrogen capping is the only supported covalent bond-breaking scheme.
- UHF is supported only without RI; UHF gradients are not available.
- Plane-wave basis sets are not supported.
- Fully ab initio AIMD is microcanonical; thermostatted/barostatted dynamics require QMMM.

Hardware guidance and known issues are listed in the [reference page](reference).

## Supported methods at a glance

### Basis sets (summary)

| Family | Examples | Notes |
| --- | --- | --- |
| Pople | 6-31G, 6-31G*, 6-31++G** | Common split-valence sets. |
| Dunning | cc-pVDZ, cc-pVTZ, aug-cc-pVDZ | RIFIT auxiliary variants available. |
| def2 | def2-SVP, def2-TZVP, def2-TZVPP | RIFIT auxiliary variants available. |
| STO-nG | STO-2G, STO-3G, STO-6G | Minimal basis sets. |
| pcseg | pcseg-0, pcseg-1 | Segmented basis sets. |

For the full list, see the [reference page](reference).

### DFT functionals (highlighted in upstream docs)

LibXC naming is required. Commonly used examples include:

- LDA: `LDA_X`, `LDA_C_PZ`, `LDA_XC_TETER93`
- GGA: `GGA_XC_PBE`, `GGA_XC_PW91`
- Meta-GGA (experimental): `MGGA_XC_TPSS`, `MGGA_XC_M06_L`
- Hybrids: `HYB_GGA_XC_B3LYP`, `HYB_GGA_XC_PBEH`, `HYB_MGGA_XC_B98`
- Double hybrids (experimental): `B2PLYP`, `revDSD-PBEP86-D4` (D4 is external)

### Correlated methods

- RI-MP2 (`RestrictedRIMP2`) is the supported MP2 implementation.
- The schema also exposes `RestrictedRICCSD`; check deployment support before relying on it.

### Dynamics capabilities

- Born-Oppenheimer AIMD with a Verlet integrator (microcanonical only).
- QMMM dynamics with NVT or NPT control via `qmmm.temperature_kelvin` and `qmmm.pressure_atm`.
- Periodic boundary conditions.
- Water-only classical solvent support (as documented upstream).

## How to choose settings (practical guidance)

### Full-system vs fragmentation

- Use full-system calculations for small to medium systems that fit on a single node.
- Use fragmentation for larger systems or when you need multi-node scaling.
- Accuracy improves with higher-order MBE truncation (dimer < trimer < tetramer), but cost grows combinatorially.

### Building fragments and cutoffs

- Provide `fragments` explicitly as lists of atom indices.
- Use `connectivity` if covalent bonds are cut so EXESS can apply hydrogen capping correctly.
- Use `cutoffs` to limit long-range n-mers and control cost; keep `dimer > trimer > tetramer` when using multiple cutoffs.
- Use `cutoff_type` to choose centroid vs closest-pair distances depending on your system.

### RI and basis set strategy

- RI is required for MP2 (only RI-MP2 is implemented).
- RI-HF/RI-MP2 allow higher angular momentum basis sets than non-RI HF.
- Always supply a matching `aux_basis` when using RI.

### DFT strategy

- Start with `ks_dft.method = "GauXC"` and the default grid.
- Treat meta-GGA and double-hybrid functionals as experimental.
- Range-separated functionals are not supported.

### Gradients, dynamics, and optimization

- Dynamics and optimization require gradients; use RHF, RI-HF, or RI-MP2.
- AIMD is microcanonical only, so choose timestep and total steps accordingly.
- For thermostatted or barostatted dynamics, use QMMM and set `keywords.regions` with `qmmm.temperature_kelvin` (and optional `qmmm.pressure_atm`).
- Geometry optimization defaults to internal coordinates; override only if you understand the tradeoffs.

### GPU scaling and system settings

- Fragmentation is the scaling path for multi-node runs.
- Tune `system.teams_per_node` and `system.gpus_per_team` to control how fragments map to GPUs.
- Use `system.max_gpu_memory_mb` to cap GPU memory if needed (be conservative; requesting more than available will crash).

## Start here: first calculation

Minimal RHF energy input (single topology):

```json
{
  "topologies": [
    { "xyz": "/path/to/water.xyz" }
  ],
  "driver": "Energy",
  "model": {
    "method": "RestrictedHF",
    "basis": "cc-pVDZ",
    "aux_basis": "cc-pVDZ-RIFIT"
  },
  "keywords": {}
}
```

Then run with the EXESS executable or the rush-py client. See the [running guide](running) for CLI and rush-py commands, and the [examples page](examples) for progressively more advanced workflows.
