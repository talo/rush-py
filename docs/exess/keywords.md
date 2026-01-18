# Keyword reference

Keywords live under the top-level `keywords` object. The main groups are `scf`, `frag`, `guess`, `optimization`, `dynamics`, `boundary`, `ff`, `log`, `export`, `ks_dft`, `rtat`, and `debug`.

## scf

| Keyword | Type | Brief |
| --- | --- | --- |
| `max_iters` | double | Max number of SCF iterations (default: 30). |
| `max_diis_history_length` | int | Max size of DIIS window (default: 8). |
| `batch_size` | int | Number of integral shell pairs per batch (default: 2560). |
| `convergence_metric` | string | Convergence metric: `DIIS`, `Energy`, `Density`. |
| `convergence_threshold` | double | SCF convergence threshold (default: 1e-6). |
| `density_threshold` | double | Integral screening threshold (default: 1e-10). |
| `density_basis_set_projection_fallback_enabled` | bool | Fall back to basis set projection if unconverged (default: true). |
| `fock_build_type` | string | Fock build algorithm: `HGP`, `UM09`, `RI`. |
| `compress_ri_b` | bool | Compress the RI B matrix (default: false). |
| `store_ri_b_on_host` | bool | Store RI B matrix on host (documented but commented in upstream docs). |

Example:

```json
"scf": {
  "max_iters": 40,
  "max_diis_history_length": 12,
  "convergence_threshold": 1e-6,
  "density_threshold": 1e-10,
  "density_basis_set_projection_fallback_enabled": false,
  "fock_build_type": "RI",
  "compress_ri_b": false,
  "convergence_metric": "DIIS"
}
```

Details:

- `max_iters`: Maximum SCF iterations. Defaults to 30.
- `max_diis_history_length`: Size of DIIS extrapolation space. Larger values use more memory.
- `batch_size`: Shell-pair batch bin size. Suggested to scale in multiples of 128; do not go below 128.
- `convergence_metric`: `Energy`, `Density`, or `DIIS`. Default is `DIIS`.
- `convergence_threshold`: Default 1e-6. Suggested values in upstream docs:
  - 1e-6 for non-fragmented RHF + RI-MP2 with `Density`/`DIIS`.
  - 1e-8 for non-fragmented RHF + RI-MP2 with `Energy`.
  - 1e-6 for dimer-level RHF + RI-MP2.
  - 1e-8 for trimer/tetramer-level calculations with `DIIS`.
  - 1e-10 for large tetramer-level calculations with `DIIS`.
- `density_threshold`: Default 1e-10. Lower values speed up SCF with potential accuracy loss.
- `density_basis_set_projection_fallback_enabled`: If unconverged, rerun using STO-3G then project to the target basis.
- `fock_build_type`:
  - `HGP`: Head-Gordon-Pople algorithm, optimized for dense systems.
  - `UM09`: Ufimtsev-Martinez algorithm, optimized for screening-heavy systems.
  - `RI`: Resolution-of-identity approximation (requires auxiliary basis, higher memory use).
- `compress_ri_b`: Compress RI B matrix (experimental).
- `store_ri_b_on_host`: When GPU memory is insufficient, store B on host (documented but commented in upstream docs).

## frag

| Keyword | Type | Brief |
| --- | --- | --- |
| `level` | int | Fragmentation level (documented as 1-4). Examples also use `Dimer`, `Trimer`, `Tetramer`. |
| `reference_fragment` | int | Reference fragment for interaction energy calculations. |
| `included_fragments` | array[int] | Fragment IDs to include. |
| `enable_speed` | bool | Experimental queue optimization for AIMD. |
| `cutoffs` | object | Distance cutoffs per level (Angstroms). |
| `cutoff_type` | string | Distance metric: `Centroid` or `MinimalDistance` (also documented as `ClosestPair`). |

Example:

```json
"frag": {
  "cutoff_type": "Centroid",
  "level": "Tetramer",
  "enable_speed": false,
  "cutoffs": {
    "dimer": 1000,
    "trimer": 20,
    "tetramer": 15
  },
  "included_fragments": [0, 1, 2, 3, 4]
}
```

Notes:
- `cutoffs` are in Angstroms and should follow `dimer > trimer > tetramer`.
- `reference_fragment` enables interaction (lattice) energies by selecting a single fragment of interest.
- `included_fragments` limits which fragments are considered.

## guess

| Keyword | Type | Brief |
| --- | --- | --- |
| `external_initial_density_path` | string | Path to external initial density. |
| `bsp` | bool | Basis set projection bootstrap (default: false). |
| `bsp_basis` | string | Lower-resolution basis set for BSP. |
| `bsp_scf_keywords` | object | SCF keywords for the BSP calculation. |
| `hcore` | bool | Use hcore initial guess. |
| `smd` | bool | Superposition of monomer densities (default: true). |
| `ssfd` | bool | Subfragment density guess (experimental, default: false). |
| `ssfd_target_size` | int | Target atoms per subfragment (default: 30). |
| `ssfd_only_converge_in_bsp_basis` | bool | Only converge subfragments in BSP basis (default: true). |
| `ssfd_scf_keywords` | object | SCF keywords for each subfragment calculation. |

## optimization

| Keyword | Type | Brief |
| --- | --- | --- |
| `max_iters` | size_t | Max optimization iterations. |
| `convergence_criteria` | object | Metric and thresholds for convergence. |
| `optimizer_reset_interval` | optional size_t | Reset coordinate system and hessian every N iterations. |
| `coordinate_system` | enum | `Cartesian`, `NaturalInternal`, `DelocalisedInternal`. |
| `hessian_guess` | enum | Hessian guess type. |
| `algorithm` | enum | Optimization algorithm type. |
| `trust_region_keywords` | optional object | Trust-region settings (TRAH only). |

Example:

```json
"optimization": {
  "max_iters": 200,
  "convergence_criteria": {
    "metric": "Baker",
    "gradient_threshold": 5.66918e-4,
    "delta_energy_threshold": 1e-6,
    "step_component_threshold": 1.2e-3
  }
}
```

Details:
- `convergence_criteria.metric`: `GradientOnly` or `Baker`.
- `gradient_threshold`, `delta_energy_threshold`, and `step_component_threshold` are numerical thresholds.
- `coordinate_system`: `DelocalisedInternal` is default and recommended.
- `hessian_guess`: identity, scaled identity (default), Schlegel, or Lindh.
- `algorithm`: trust region augmented hessian or eigenvector following (default).
- `trust_region_keywords` includes `initial_radius`, `max_radius`, `min_radius`, `increase_factor`, `decrease_factor`, `constrict_factor`, `increase_threshold`, `decrease_threshold`, `rejection_threshold`.

## dynamics

| Keyword | Type | Brief |
| --- | --- | --- |
| `n_timesteps` | int | Number of timesteps. |
| `dt` | double | Timestep size in ps (default: 0.001). |
| `use_async_timesteps` | bool | Asynchronous timesteps (expert). |

Example:

```json
"dynamics": {
  "n_timesteps": 10,
  "use_async_timesteps": false,
  "dt": 0.002
}
```

## boundary

Boundary conditions for periodic simulations:

```json
"boundary": {
  "x": { "kind": "Periodic", "range": { "lower": -2, "upper": 3 } },
  "y": { "kind": "Periodic", "range": { "lower": -2, "upper": 3 } },
  "z": { "kind": "Periodic", "range": { "lower": -2, "upper": 3 } }
}
```

## ff

| Keyword | Type | Brief |
| --- | --- | --- |
| `ff_filename` | string | Force field filename path. |

## log

| Keyword | Type | Brief |
| --- | --- | --- |
| `console` | object | Console log settings. |
| `logfiles` | array | File log settings. |

Example:

```json
"log": {
  "console": { "level": "Verbose" },
  "logfiles": [
    {
      "level": "Verbose",
      "prefix_fmt": "[%Y-%m-%d %H:%M:%S.{us} r{rank} {level}] ",
      "directory": "/tmp/exess"
    }
  ]
}
```

Log levels (descending verbosity): `Debug`, `Verbose`, `LargeInfo`, `Info`, `Performance`, `Warning`.

## rtat

RTAT is a runtime auto-tuner for matrix operations. EXESS can use it for GPU BLAS tuning.

| Keyword | Type | Brief |
| --- | --- | --- |
| `enabled` | bool | Enable runtime autotuning. |
| `synchronous` | bool | Use synchronous operations. |
| `json_file_dump_prefix` | optional string | Prefix for RTAT JSON dumps. |

Example:

```json
"rtat": {
  "enabled": true,
  "synchronous": true,
  "json_file_dump_prefix": "prefix"
}
```

## debug

Example:

```json
"debug": {
  "dry_run": false,
  "print_subfragment_xyz": false,
  "max_fragments": 10000,
  "skip_calcs": false
}
```

Documented debug keywords:
- `dry_run`: Validate fragment queue without computing.
- `print_subfragment_xyz`: Print subfragment XYZ (for SSFD).
- `max_fragments`: Limit number of fragments computed.
- `ignore_fragments`: Ignore fragmentation (developer validation).
- `skip_calcs`: Skip computations in fragmentation routines.

## export

Export controls what is written to HDF5 output files:

| Keyword | Type | Brief |
| --- | --- | --- |
| `export_density` | bool | Export density matrix. |
| `export_relaxed_mp2_density_correction` | bool | Export relaxed MP2 density correction. |
| `export_fock` | bool | Export Fock matrix. |
| `export_overlap` | bool | Export overlap matrix. |
| `export_h_core` | bool | Export H core matrix. |
| `export_expanded_density` | bool | Export expanded density. |
| `export_expanded_gradient` | bool | Export expanded gradient. |
| `export_molecular_orbital_coeffs` | bool | Export MO coefficients. |
| `export_gradient` | bool | Export gradient. |
| `export_mulliken_charges` | bool | Export Mulliken charges. |
| `export_bond_orders` | bool | Export bond orders. |
| `export_h_caps` | bool | Export H caps. |
| `export_density_descriptors` | bool | Export density descriptors. |
| `export_esp_descriptors` | bool | Export ESP descriptors. |
| `export_basis_labels` | bool | Export basis labels. |
| `flatten_symmetric` | bool | Export lower triangle for symmetric matrices (default: true). |
| `concatenate_hdf5_files` | bool | Concatenate multi-team HDF5 outputs (can be expensive). |
| `descriptor_grid` | array | Standard grid, grid params, or raw point list. |

## ks_dft

KSDFT is used when `model.method` is `RestrictedKSDFT`. The upstream docs recommend reading the KSDFT paper (DOI: 10.1021/acs.jctc.5c01229).

| Keyword | Type | Brief |
| --- | --- | --- |
| `functional` | string | LibXC functional name (required). |
| `method` | string | XC evaluation method (default: `GauXC`). |
| `use_C_opt` | bool | Use C-matrix optimization for Dense/BatchDense (default: true). |
| `grid` | object | Numerical grid parameters (defaults to ULTRAFINE). |

### functional

Examples: `GGA_XC_PBE`, `HYB_GGA_XC_B3LYP`, `HYB_MGGA_XC_B98`.

Notes from upstream docs:
- Meta-GGA functionals are experimental and supported only with `GauXC`.
- Range-separated functionals are not supported.
- Only `B2PLYP` and `revDSD-PBEP86-D4` double hybrids are implemented; D4 must be added externally.

### method

Available methods:
- `GauXC` (default)
- `Dense`
- `BatchDense`
- `Direct`
- `SemiDirect`

### use_C_opt

Uses the coefficient matrix instead of the density matrix to reduce cost. Default is `true`.

### grid

If unspecified, EXESS uses an ULTRAFINE grid with ROBUST pruning. Grid parameters include:

- Preset sizes: `FINE`, `ULTRAFINE` (default), `SUPERFINE`, `TREUTLER_GM3`, `TREUTLER_GM5`.
- Custom sizes: `radial_size` and `angular_size`.
- Radial quadrature: `MuraKnowles` (default), `MurrayHandyLaming`, `TreutlerAldrichs`.
- Pruning: `ROBUST` (default), `UNPRUNED`, `TREUTLER`.
- Batching:
  - `octree`: `max_size`, `max_depth`, `max_distance`.
  - `space_filling`: `octree` plus `target_batch_size`.
  - `GauXC` batching: `batch_size`.

Examples:

```json
"ks_dft": {
  "functional": "GGA_XC_PBE"
}
```

```json
"ks_dft": {
  "functional": "HYB_GGA_XC_B3LYP",
  "method": "Dense",
  "grid": {
    "default_grid": "ULTRAFINE",
    "radial_quad": "MuraKnowles",
    "pruning_scheme": "ROBUST"
  }
}
```

```json
"ks_dft": {
  "functional": "HYB_GGA_XC_B3LYP",
  "method": "BatchDense",
  "grid": {
    "octree": { "max_size": 512 }
  }
}
```
