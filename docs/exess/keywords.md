# Keyword reference

Keywords live under the top-level `keywords` object. The groups recognized by the EXESS/libqdx schema are:

`scf`, `ks_dft`, `rtat`, `frag`, `boundary`, `debug`, `export`, `guess`, `log`, `dynamics`, `integrals`, `force_field`, `optimization`, `gradient`, `hessian`, `machine_learning`, `qmmm`, `regions`.

## scf

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `max_iters` | int | 50 | Max number of SCF iterations. |
| `max_diis_history_length` | int | 8 | Max size of DIIS window. |
| `batch_size` | int | 2560 | Shell-pair batches per bin. |
| `convergence_metric` | string | `DIIS` | `DIIS`, `Energy`, or `Density`. |
| `convergence_threshold` | float | 1e-6 | SCF convergence threshold. |
| `density_threshold` | float | 1e-10 | Density screening threshold. |
| `gradient_screening_threshold` | float | 1e-10 | Gradient screening threshold. |
| `bf_cutoff_threshold` | float | none | Basis function cutoff threshold (DFT/shell pairs). |
| `density_basis_set_projection_fallback_enabled` | bool | none | STO-3G projection fallback. |
| `use_ri` | bool | false | Deprecated RI toggle. |
| `allow_crap_scf` | bool | false | Expert flag. |
| `store_ri_b_on_host` | bool | false | Store RI B on host. |
| `compress_ri_b` | bool | false | Compress RI B matrix. |
| `homo_lumo_guess_rotation_angle` | float | none | HOMO/LUMO rotation (degrees). |
| `fock_build_type` | string | `HGP` | `HGP`, `UM09`, or `RI`. |
| `exchange_screening_threshold` | float | 1e-5 | Exchange screening threshold. |
| `group_shared_exponents` | bool | false | Group shared basis exponents (UM09 only). |

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

- `max_iters`: Default in libqdx is 50 (older docs mention 30).
- `batch_size`: Use multiples of 128; do not go below 128.
- `convergence_metric`: `Energy`, `Density`, or `DIIS`. Default is `DIIS` in libqdx.
- `convergence_threshold`: Suggested values from upstream docs:
  - 1e-6 for non-fragmented RHF + RI-MP2 with `Density`/`DIIS`.
  - 1e-8 for non-fragmented RHF + RI-MP2 with `Energy`.
  - 1e-6 for dimer-level RHF + RI-MP2.
  - 1e-8 for trimer/tetramer-level calculations with `DIIS`.
  - 1e-10 for large tetramer-level calculations with `DIIS`.
- `density_threshold`: Lower values speed up SCF with potential accuracy loss.
- `gradient_screening_threshold`: Additional screening for gradient-related integrals.
- `density_basis_set_projection_fallback_enabled`: If unconverged, rerun using STO-3G then project to the target basis (C++ comments suggest true for fragmented runs, false otherwise).
- `fock_build_type`:
  - `HGP`: Head-Gordon-Pople algorithm, optimized for dense systems.
  - `UM09`: Ufimtsev-Martinez algorithm, optimized for screening-heavy systems.
  - `RI`: Resolution-of-identity approximation (requires auxiliary basis, higher memory use).
- `use_ri`: Deprecated in EXESS; use `fock_build_type = "RI"` instead.
- `homo_lumo_guess_rotation_angle`: Rotation in degrees (0-180) for unrestricted symmetry breaking.
- `exchange_screening_threshold` and `group_shared_exponents` are expert controls for large systems and shared-exponent basis sets.

## frag

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `level` | string | required | `Monomer` .. `Octamer`. |
| `cutoffs` | object | none | Distance cutoffs in Angstroms. |
| `cutoff_type` | string | `ClosestPair` | `Centroid` or `ClosestPair`. |
| `distance_metric` | string | `Max` | `Max`, `Average`, `Min`, `Ryan`. |
| `reference_fragment` | int | none | Reference fragment for interaction energies. |
| `included_fragments` | array[int] | none | Subset of fragments to include. |
| `enable_speed` | bool | false | Experimental queue optimization. |

Notes:

- `cutoffs` can include `dimer`, `trimer`, `tetramer`, `pentamer`, `hexamer`, `heptamer`, `octamer`.
- Distances are in Angstroms and should follow `dimer > trimer > tetramer` when using higher orders.
- `distance_metric` affects higher-order fragment distances and is noted in the C++ schema as undocumented.

Example:

```json
"frag": {
  "cutoff_type": "Centroid",
  "distance_metric": "Max",
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

## guess

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `external_initial_density_path` | string | none | HDF5 density guess path. |
| `bsp` | bool | false | Basis set projection bootstrap. |
| `bsp_basis` | string | empty | Lower-resolution basis set for BSP. |
| `bsp_scf_keywords` | object | none | SCF keywords for BSP. |
| `hcore` | bool | false | Use hcore initial guess. |
| `smd` | bool | none | Superposition of monomer densities. |
| `ssfd` | bool | false | Subfragment density guess (experimental). |
| `ssfd_target_size` | int | 30 | Target atoms per subfragment. |
| `ssfd_only_converge_in_bsp_basis` | bool | true | Only converge subfragments in BSP basis. |
| `ssfd_scf_keywords` | object | none | SCF keywords for subfragment runs. |

`external_initial_density_path` must reference an HDF5 file with a `density` dataset at root for RHF, or `alpha/density` and `beta/density` for UHF. The EXESS schema warns that density guesses from other codes may be incompatible due to basis ordering and normalization. External guesses are not supported for fragmented calculations.

## optimization

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `max_iters` | int | required | Max optimization iterations. |
| `convergence_criteria` | object | defaults | Metric + thresholds. |
| `optimizer_reset_interval` | int | none | Reset coordinate system and hessian every N iterations. |
| `coordinate_system` | string | `DelocalisedInternal` | `Cartesian`, `NaturalInternal`, `DelocalisedInternal`. |
| `constraints` | array[array[int]] | none | Constraints on bonds/angles/dihedrals. |
| `hessian_guess` | string | depends | `Identity`, `ScaledIdentity`, `Schlegel`, `Lindh`. |
| `algorithm` | string | `EigenvectorFollowing` | `EigenvectorFollowing`, `TrustRegionAugmentedHessian`, `LBFGS`. |
| `lbfgs_keywords` | object | none | LBFGS parameters. |
| `trust_region` | object | defaults | Trust-region parameters. |
| `frozen_distance_slippage_tolerance_angstroms` | float | 1e-8 | Slippage tolerance (distance). |
| `frozen_angle_slippage_tolerance_degrees` | float | 1e-8 | Slippage tolerance (angle). |
| `debug_xyz` | bool | false | Debug XYZ output. |
| `output_trc` | string | none | Output TRC path. |
| `fixed_atoms` | array[int] | none | Fixed atoms. |
| `free_atoms` | array[int] | none | Free atoms. |
| `fixed_fragments` | array[int] | none | Fixed fragments. |
| `free_fragments` | array[int] | none | Free fragments. |
| `fix_heavy` | bool | false | Fix heavy atoms. |

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
- Default thresholds come from Baker criteria (see https://doi.org/10.1063/1.1515483).
- `trust_region` defaults are based on Helmich-Paris 2021.
- `constraints` are lists of atom indices specifying constrained bonds, angles, or dihedrals.

## dynamics

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `n_timesteps` | int | required | Number of timesteps. |
| `dt` | float | required | Timestep size in ps. |
| `reuse_orbitals` | bool | false | Reuse orbitals between steps. |
| `use_async_timesteps` | bool | true | Asynchronous timesteps (expert). |

Example:

```json
"dynamics": {
  "n_timesteps": 10,
  "use_async_timesteps": false,
  "dt": 0.002
}
```

## boundary

Boundary conditions for periodic or truncated simulations:

```json
"boundary": {
  "x": { "kind": "Periodic", "range": { "lower": -2, "upper": 3 } },
  "y": { "kind": "Periodic", "range": { "lower": -2, "upper": 3 } },
  "z": { "kind": "Periodic", "range": { "lower": -2, "upper": 3 } }
}
```

`kind` can be `Periodic`, `Rigid`, or `Delete`.

## force_field

| Keyword | Type | Brief |
| --- | --- | --- |
| `ff_filename` | string | Force field filename path. |

## log

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `console` | object | see below | Console log settings. |
| `logfiles` | array | empty | File log settings. |

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

Log levels: `Debug`, `Verbose`, `LargeInfo`, `Info`, `Performance`, `Warning`, `Error`.

Defaults from the C++ schema:

- `console.level`: `LargeInfo` (or `Debug` in debug builds).
- `console.prefix_fmt`: empty string.
- `logfiles.level`: `Verbose`.
- `logfiles.prefix_fmt`: `[%Y-%m-%d %H:%M:%S.{us} r{rank} {level}] `.

## rtat

RTAT is a runtime auto-tuner for matrix operations.

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `enabled` | bool | true | Enable runtime autotuning. |
| `synchronous` | bool | false | Use synchronous operations. |
| `json_file_dump_prefix` | string | none | Prefix for RTAT JSON dumps. |

## export

Export controls what is written to HDF5 output files:

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `export_density` | bool | false | Export density. |
| `export_relaxed_mp2_density_correction` | bool | false | Export relaxed MP2 density correction. |
| `export_fock` | bool | false | Export Fock matrix. |
| `export_overlap` | bool | false | Export overlap matrix. |
| `export_h_core` | bool | false | Export H core matrix. |
| `export_expanded_density` | bool | false | Export expanded density. |
| `export_expanded_gradient` | bool | false | Export expanded gradient. |
| `export_molecular_orbital_coeffs` | bool | false | Export MO coefficients. |
| `export_gradient` | bool | false | Export gradients. |
| `export_external_charge_gradient` | bool | false | Export external charge gradients. |
| `export_mulliken_charges` | bool | false | Export Mulliken charges. |
| `export_chelpg_charges` | bool | false | Export CHELPG charges. |
| `export_bond_orders` | bool | false | Export bond orders. |
| `export_h_caps` | bool | false | Export H caps. |
| `export_density_descriptors` | bool | false | Export density descriptors. |
| `export_esp_descriptors` | bool | false | Export ESP descriptors. |
| `export_expanded_esp_descriptors` | bool | false | Export expanded ESP descriptors. |
| `export_basis_labels` | bool | false | Export basis labels. |
| `export_hessian` | bool | false | Export hessian. |
| `export_mass_weighted_hessian` | bool | false | Export mass-weighted hessian. |
| `export_hessian_frequencies` | bool | false | Export hessian frequencies. |
| `flatten_symmetric` | bool | true | Flatten symmetric matrices. |
| `light_json` | bool | false | Light JSON output. |
| `concatenate_hdf5_files` | bool | false | Concatenate HDF5 outputs. |
| `training_db` | bool | false | Export training DB metadata. |
| `descriptor_grid` | object | none | Grid for descriptor exports. |

`descriptor_grid` can be one of the following structures (libqdx):

- `standard`: `FINE`, `ULTRAFINE`, `SUPERFINE`, `TREUTLER_GM3`, `TREUTLER_GM5`.
- `params`: `points_per_shell`, `order` (`One` or `Two`), `scale`.
- `regular`: `min`, `max`, `spacing` arrays (Cartesian grid).
- `custom`: flat list of points `[x1, y1, z1, x2, y2, z2, ...]`.

The rush-py API maps these to `StandardDescriptorGrid`, `DescriptorGrid`, `RegularDescriptorGrid`, and `CustomDescriptorGrid`.

Note: the rush-py tutorial warns that `export_expanded_esp_descriptors` can trigger an internal OOM error.

## ks_dft

KSDFT is used when `model.method` is `RestrictedKSDFT`. The upstream docs recommend reading the KSDFT paper (DOI: 10.1021/acs.jctc.5c01229).

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `functional` | string | required | LibXC functional name. |
| `method` | string | `GauXC` | XC evaluation method. |
| `use_C_opt` | bool | true | Use C-matrix optimization (Dense/BatchDense). |
| `grid` | object | `{}` | Numerical grid settings. |
| `sp_threshold` | float | none | Single-precision threshold. |
| `dp_threshold` | float | none | Double-precision threshold. |
| `batches_per_batch` | int | 20 | Batch batching for GauXC. |

Grid parameters supported in the libqdx Rust schema:

- `radial_quad`: `MuraKnowles`, `MurrayHandyLaming`, `TreutlerAldrichs`.
- `pruning_scheme`: `ROBUST`, `UNPRUNED`, `TREUTLER`.
- `batch_size`: GauXC batch size.
- `radial_size`, `angular_size`: Custom grid sizes.
- `default_grid`: `FINE`, `ULTRAFINE`, `SUPERFINE`, `TREUTLER_GM3`, `TREUTLER_GM5`.

The C++ schema treats `grid` as raw JSON and may accept additional structures (octree, space-filling curves) described in the upstream docs.

Functional notes from upstream docs:

- Meta-GGA functionals are experimental and supported only with `GauXC`.
- Range-separated functionals are not supported.
- Only `B2PLYP` and `revDSD-PBEP86-D4` double hybrids are implemented; D4 must be added externally.

## integrals

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `scheduler` | string | `Callback` | `Callback` or `RoundRobin`. |
| `n_streams` | int | 4 (CUDA) / 1 (HIP) | GPU stream count. |

## gradient

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `finite_difference_step_size` | float | 5e-3 | Step size for numerical gradients. |
| `method` | string | `Analytical` | `Analytical` or `Numerical`. |

## hessian

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `finite_difference_step_size` | float | 5e-3 | Step size for numerical Hessians. |
| `method` | string | `Numerical` | `Analytical` or `Numerical`. |

## machine_learning

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `ml_type` | string | `AIMNet` | ML model type. |

## qmmm

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `n_timesteps` | int | required | Number of QMMM timesteps. |
| `dt_ps` | float | required | Timestep size in ps. |
| `temperature_kelvin` | float | required | Temperature in Kelvin. |
| `pressure_atm` | float | none | If set, runs NPT; otherwise NVT. |
| `minimisation` | object | none | Classical minimisation settings. |
| `trajectory` | object | none | Trajectory output settings. |
| `energy_csv` | string | none | Path for energy CSV. |
| `restraints` | object | none | Restraints for atoms/fragments. |

`minimisation` fields:

- `err_tol_kj_per_mol_nm` (default 10)
- `max_iterations` (default 0)

`trajectory` fields:

- `format`: `JSON` or `XYZ` (default `JSON`)
- `interval` (default 1)
- `start` (default 0)
- `end` (default max u32)
- `include_waters` (default false)

`restraints` fields:

- `k` (default 2000.0)
- `fixed_atoms` / `free_atoms`
- `fixed_fragments` / `free_fragments`
- `fix_heavy`

## regions

`regions` defines which fragments are treated as QM, MM, or ML (Q4ML/QMMM workflows):

- `qm_fragments`: array[int]
- `mm_fragments`: array[int]
- `ml_fragments`: array[int]

## debug

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `dry_run` | bool | false | Validate fragment queue without computing. |
| `print_subfragment_xyz` | bool | false | Print subfragment XYZ for SSFD. |
| `max_fragments` | int | -1 | Limit number of fragments computed. |
| `ignore_fragments` | bool | false | Ignore fragmentation (developer validation). |
| `skip_calcs` | bool | false | Skip computations in fragmentation routines. |
