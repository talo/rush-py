# Electronic Structure Methods

## Supported methods at a glance

### Basis sets (summary)

| Family | Examples | Notes |
| --- | --- | --- |
| Pople | 6-31G, 6-31G*, 6-31++G**, 6-311G** | Common split-valence sets. |
| Dunning | cc-pVDZ, cc-pVTZ, cc-pVQZ, aug-cc-pVTZ | RIFIT auxiliary variants available. |
| def2 | def2-SVP, def2-TZVP, def2-TZVPP, def2-QZVP | RIFIT auxiliary variants available. |
| STO-nG | STO-2G, STO-3G, STO-6G | Minimal basis sets. |
| PCSeg | PCSeg-0, PCSeg-1 | Segmented basis sets. |

For the full list, see the [reference page](reference).

### DFT functionals

EXESS uses LibXC through ExchCXX. Any LibXC functional name accepted by the build can be
requested directly; invalid names cause EXESS to error and print the available functionals.

EXESS also supports custom linear combinations in the ExchCXX syntax, for example:
`0.2*gga_x_pbe+0.8*gga_x_b88+0.2*hf`.

Built-in aliases with EXESS-defined coefficients include:

- `SVWN5` (expanded as `LDA_X+LDA_C_VWN`)
- `B2PLYP`
- `REVDSD-PBEP86-D4`
- `REVDSD-PBEP86-D4(NOFC)`

### Correlated methods

- RI-MP2 (`RestrictedRIMP2`) is the supported MP2 implementation.
- `RestrictedRICCSD` is available in the method list; check deployment support before relying on it.

### Dynamics capabilities

- Born-Oppenheimer AIMD with a Verlet integrator (microcanonical only).
- QMMM dynamics with NVT or NPT control via `qmmm.temperature_kelvin` and `qmmm.pressure_atm`.
- Periodic boundary conditions.
- Water-only classical solvent support.

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

- Start with `ks_dft.method = "BatchDense"` and the default grid.
- Use `GauXC` only when you need it for compatibility or benchmarking.

### Gradients, dynamics, and optimization

- Dynamics and optimization require gradients. Analytical gradients are only available for restricted HF and restricted RI-MP2; use numerical derivatives otherwise.
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

## Hartree-Fock

EXESS supports RHF and UHF energies. Analytical gradients are available for restricted HF but not UHF. Spherical basis functions (`force_cartesian_basis_sets=false`) are only supported for `Energy` calculations.

## RI-HF and RI-MP2

RI is the main path to scalable HF and MP2 in EXESS:

- **RI-HF** is enabled by setting `scf.fock_build_type = "RI"` with `method = "RestrictedHF"`.
- **RI-MP2** is the supported MP2 implementation (`method = "RestrictedRIMP2"`). MP2 runs are RI by design.

RI reduces computational cost but increases memory use because integrals are stored. RI-HF, RI-MP2, and RI-CCSD require an auxiliary basis set, and double-hybrid KSDFT functionals require one as well. In return, RI-HF/RI-MP2 support higher angular momentum basis sets (up to F in the primary basis and G in the auxiliary basis).

## KSDFT

Restricted KSDFT is supported via LibXC functionals (DFT support must be compiled into EXESS). Unrestricted DFT is not implemented. Analytical gradients are not available for KSDFT; use numerical derivatives when needed.

Method support summary (analytical gradients only):

| Calculation | RHF | UHF | RI-HF | RI-MP2 | RestrictedKSDFT |
| --- | --- | --- | --- | --- | --- |
| Energy | yes | yes | yes | yes | yes |
| Analytical gradient | yes | no | yes | yes | no |

## Fragmentation theory (MBE)

EXESS implements a Many-Body Expansion (MBE) for non-covalent and covalent systems, with truncation up to fourth order (tetramers). The number of n-mers grows combinatorially with fragment count:

`C(n, k) = n! / (k! * (n - k)!)`

Covalent fragmentation uses hydrogen capping for broken single bonds. If you cut covalent bonds, provide `connectivity` so EXESS can cap and restore bonds correctly.

For background and accuracy considerations, see https://doi.org/10.1063/1.5126216 and https://doi.org/10.1021/cr200093j.

(exess-gradients-dynamics-optimization)=
## Gradients, dynamics, and optimization

- Geometry optimization is supported for RHF, RI-HF, and RI-MP2.
- Born-Oppenheimer AIMD uses a Verlet integrator and is microcanonical only (no thermostats or barostats).
- QMMM dynamics support NVT/NPT via the `qmmm` block (temperature and optional pressure), with MM handled through OpenMM and ML via AIMNet.
- Dynamics can be combined with fragmentation for large systems.
- Periodic boundary conditions and water-only classical solvent support are available in AIMD workflows.

(exess-limitations)=
## Limitations

EXESS is strong when you need GPU performance and fragmentation at scale. It is not a general-purpose electronic structure code with every method or model.

**Limitations:**

- Unrestricted KSDFT is not currently supported.
- Unrestricted RI-MP2 is not currently supported (use `RestrictedRIMP2`).
- The UM09 (MMD) Fock build path is not currently defined for UHF when using spherical basis functions.
- Analytical gradients are currently limited to restricted HF and restricted RI-MP2.
- Hydrogen capping is currently the only supported covalent bond-breaking scheme.
- Plane-wave basis sets are not currently supported.
- Relativistic calculations are not currently supported.
- Elements beyond xenon (Z > 54) are not currently supported.
- Excited-state methods are not currently supported. EXESS does not natively implement TDDFT, excited-state coupled-cluster/response methods, transition dipoles between electronic states, or core-level XUV/X-ray spectroscopy.
- Fully ab initio AIMD is currently microcanonical only; thermostatted/barostatted dynamics require QMMM.

:::{only} internal
Notes on provenance:
- UHF KSDFT: `EXESS/src/methods/hf/energy/hf.cpp` ("UHF not implemented for DFT").
- Unrestricted RI-MP2: `EXESS/src/core/calculation_input/calculation_model.cpp` (throws on `UnrestrictedRIMP2`).
- UM09 UHF + spherical: `EXESS/src/methods/hf/energy/hf.cpp` (assert "MMD not defined for UHF").
- Analytical gradients: `EXESS/src/core/calculation_input/calculation_types.hpp` (`supports_analytical_gradients`).
- Spherical basis restriction: `EXESS/src/core/calculation_input/calculation_input.cpp` (driver check for `force_cartesian_basis_sets=false`).
:::

Hardware guidance and known issues are listed in the [reference page](reference).
