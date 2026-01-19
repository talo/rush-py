# Keyword reference

Keywords live under the top-level `keywords` object. The groups recognized by the EXESS/libqdx schema are:

`scf`, `ks_dft`, `rtat`, `frag`, `boundary`, `debug`, `export`, `guess`, `log`, `dynamics`, `integrals`, `force_field`, `optimization`, `gradient`, `hessian`, `machine_learning`, `qmmm`, `regions`.

Defaults in the tables below reflect the EXESS command-line behavior (JSON parser defaults plus EXESS internal defaults). Rush-py defaults are listed at the end of this page.

The upstream manual describes `keywords` as the main set of controls for the calculation. In practice, you will spend most of your time in `scf`, `frag`, `ks_dft`, and the driver-specific groups (`optimization`, `dynamics`, `qmmm`).

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
| `bf_cutoff_threshold` | float | `density_threshold` | Basis function cutoff threshold (DFT/shell pairs). |
| `density_basis_set_projection_fallback_enabled` | bool | auto (fragmented) | STO-3G projection fallback. |
| `use_ri` | bool | false | Deprecated RI toggle. |
| `allow_crap_scf` | bool | false | Expert flag. |
| `store_ri_b_on_host` | bool | false | Store RI B on host. |
| `compress_ri_b` | bool | false | Compress RI B matrix. |
| `homo_lumo_guess_rotation_angle` | float | auto (0 or 45) | HOMO/LUMO rotation (degrees). |
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

- `batch_size`: Use multiples of 128; do not go below 128. Upstream docs cite 10.1021/acs.jctc.0c00768, 10.1021/acs.jctc.1c00720, and 10.1080/00268976.2022.2112987 for details on the shell-pair batch bin container.
- `convergence_metric`: `Energy`, `Density`, or `DIIS`.
- `convergence_threshold`: Suggested values from upstream docs:
  - 1e-6 for non-fragmented RHF + RI-MP2 with `Density`/`DIIS`.
  - 1e-8 for non-fragmented RHF + RI-MP2 with `Energy`.
  - 1e-6 for dimer-level RHF + RI-MP2.
  - 1e-8 for trimer/tetramer-level calculations with `DIIS`.
  - 1e-10 for large tetramer-level calculations with `DIIS`.
- `density_threshold`: Lower values speed up SCF with potential accuracy loss. Upstream guidance suggests exploring 1e-8 to 1e-12 and validating accuracy; too-large values can lead to NaNs. Increasing to 1e-11 or 1e-12 will slow SCF but can improve accuracy for higher-order fragmentation (e.g., tetramers) and produce crisper MP2 orbitals. Validate results against the default before adopting more aggressive thresholds.
- `gradient_screening_threshold`: Additional screening for gradient-related integrals.
- `bf_cutoff_threshold`: If omitted, EXESS uses `density_threshold`.
- `density_basis_set_projection_fallback_enabled`: If omitted, EXESS enables fallback for fragmented calculations and disables it for full-system calculations. When triggered, EXESS reruns SCF in STO-3G and projects the density into the target basis.
- `fock_build_type`:
  - `HGP`: Head-Gordon-Pople algorithm, optimized for dense systems.
  - `UM09`: Ufimtsev-Martinez algorithm, optimized for screening-heavy systems.
  - `RI`: Resolution-of-identity approximation (requires auxiliary basis, higher memory use).
- `use_ri`: Deprecated in EXESS (scheduled for removal in 5.0.0); use `fock_build_type = "RI"` instead. If set, EXESS forces the Fock build type to `RI`.
- `homo_lumo_guess_rotation_angle`: Rotation in degrees (0-180) for unrestricted symmetry breaking. If omitted, EXESS uses 45 degrees for unrestricted singlets and 0 otherwise.
- `fock_build_type` guidance from upstream docs:
  - `HGP` is tuned for dense systems where screening is less important (e.g., compact biomolecules).
  - `UM09` is tuned for screening-heavy systems (e.g., long chains) and can scale better on large systems.
  - `RI` stores integrals, can be faster on small systems, but memory usage rises substantially; it requires an auxiliary basis.
- `store_ri_b_on_host`: Use this if GPU memory is insufficient for RI; this is slower but can still outperform non-RI for some systems.
- `compress_ri_b`: Experimental compression for RI-HF; upstream docs warn it may misbehave.
- `group_shared_exponents`: Expert control used with UM09 and shared-exponent basis sets (e.g., cc-pVDZ).
- `exchange_screening_threshold` and `allow_crap_scf` are expert controls; adjust only with validation.
- `fock_build_type` includes improved screening for large systems (>3000 basis functions); see https://arxiv.org/abs/2407.21445 for details.

## frag

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `level` | string | required | `Monomer` .. `Octamer`. |
| `cutoffs` | object | unset | Distance cutoffs in Angstroms. |
| `cutoff_type` | string | `ClosestPair` | `Centroid` or `ClosestPair`. |
| `distance_metric` | string | `Max` | `Max`, `Average`, `Min`, `Ryan`. |
| `reference_fragment` | int | unset | Reference fragment for interaction energies. |
| `included_fragments` | array[int] | unset | Subset of fragments to include. |
| `enable_speed` | bool | false | Experimental queue optimization. |

Notes:

- `cutoffs` can include `dimer`, `trimer`, `tetramer`, `pentamer`, `hexamer`, `heptamer`, `octamer`.
- Distances are in Angstroms and should follow `dimer > trimer > tetramer` when using higher orders.
- If `cutoffs` is omitted, the calculation proceeds without distance filtering (all n-mers up to `level`); be cautious with fragment counts to avoid excessive compute.
- Truncation counts scale combinatorially: dimers `n(n-1)/2`, trimers `n(n-1)(n-2)/6`, tetramers `n(n-1)(n-2)(n-3)/24`.
- `reference_fragment` enables lattice/interaction energies by summing n-mer corrections that include the reference fragment. Negative values indicate binding; positive values indicate repulsion under the usual convention.
- `included_fragments` restricts the fragment set and treats them as an independent system.
- `cutoff_type`:
  - `Centroid` compares fragment centroids.
  - `ClosestPair` uses the minimal inter-fragment atom distance (more accurate and generally preferred).
- `distance_metric` controls how higher-order distances are computed from pair distances (`Max`, `Min`, `Average`, or `Ryan`).
- `enable_speed` is an experimental queue optimization intended for AIMD workflows (upstream docs label this "broom broom").

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

Enum aliases accepted by libqdx (case variants shown as defined in the parser):

- `level`: `MONOMER`, `Monomer`, `monomer`; `DIMMER`/`Dimer`/`dimer`; `TRIMER`/`Trimer`/`trimer`; `TETRAMER`/`Tetramer`/`tetramer`; `PENTAMER`/`Pentamer`/`pentamer`; `HEXAMER`/`Hexamer`/`hexamer`; `HEPTAMER`/`Heptamer`/`heptamer`; `OCTAMER`/`Octamer`/`octamer`.
- `cutoff_type`: `CENTROID`, `Centroid`, `centroid`; `CLOSEST_PAIR`, `ClosestPair`, `closest_pair`.
- `distance_metric`: `MAX`, `Max`, `max`; `AVERAGE`, `Average`, `average`; `MIN`, `Min`, `min`.

## guess

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `external_initial_density_path` | string | unset | HDF5 density guess path. |
| `bsp` | bool | false | Basis set projection bootstrap. |
| `bsp_basis` | string | empty | Lower-resolution basis set for BSP. |
| `bsp_scf_keywords` | object | unset | SCF keywords for BSP. |
| `hcore` | bool | false | Use hcore initial guess. |
| `smd` | bool | auto (fragmented non-RI) | Superposition of monomer densities. |
| `ssfd` | bool | false | Subfragment density guess (experimental). |
| `ssfd_target_size` | int | 30 | Target atoms per subfragment. |
| `ssfd_only_converge_in_bsp_basis` | bool | true | Only converge subfragments in BSP basis. |
| `ssfd_scf_keywords` | object | unset | SCF keywords for subfragment runs. |

`external_initial_density_path` must reference an HDF5 file with a `density` dataset at root for RHF, or `alpha/density` and `beta/density` for UHF. Guesses are expected to be stored as flattened lower-triangular density matrices. External guesses are not supported for fragmented calculations, and EXESS warns that guesses from other codes may be incompatible due to basis ordering and normalization.

If `smd` is omitted, EXESS enables it for fragmented calculations that are not using RI, and disables it otherwise.

If `bsp_scf_keywords` or `ssfd_scf_keywords` are omitted, EXESS reuses the base SCF keywords.

Additional notes from upstream docs:

- `bsp` (basis set projection) computes a lower-resolution SCF and projects to the target basis; it is off by default and requires `bsp_basis`.
- `ssfd` is an experimental subfragment guess; `ssfd_target_size` controls subfragment size (default 30).
- `ssfd_only_converge_in_bsp_basis` keeps subfragments unconverged in the primary basis and only projects from the bootstrap basis.

## optimization

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `max_iters` | int | required | Max optimization iterations. |
| `convergence_criteria` | object | defaults | Metric + thresholds. |
| `optimizer_reset_interval` | int | unset | Reset coordinate system and hessian every N iterations. |
| `coordinate_system` | string | `DelocalisedInternal` | `Cartesian`, `NaturalInternal`, `DelocalisedInternal`. |
| `constraints` | array[array[int]] | `[]` | Constraints on bonds/angles/dihedrals. |
| `hessian_guess` | string | depends | `Identity`, `ScaledIdentity`, `Schlegel`, `Lindh`. |
| `algorithm` | string | `EigenvectorFollowing` | `EigenvectorFollowing`, `TrustRegionAugmentedHessian`, `LBFGS`. |
| `lbfgs_keywords` | object | unset | LBFGS parameters. |
| `trust_region` | object | defaults | Trust-region parameters. |
| `frozen_distance_slippage_tolerance_angstroms` | float | 1e-8 | Slippage tolerance (distance). |
| `frozen_angle_slippage_tolerance_degrees` | float | 1e-8 | Slippage tolerance (angle). |
| `debug_xyz` | bool | false | Debug XYZ output. |
| `output_trc` | string | unset | Output TRC path. |
| `fixed_atoms` | array[int] | unset | Fixed atoms. |
| `free_atoms` | array[int] | unset | Free atoms. |
| `fixed_fragments` | array[int] | unset | Fixed fragments. |
| `free_fragments` | array[int] | unset | Free fragments. |
| `fix_heavy` | bool | false | Fix heavy atoms. |

Defaults for `convergence_criteria`:

- `metric`: `Baker`
- `gradient_threshold`: `3e-4`
- `delta_energy_threshold`: `1e-6`
- `step_component_threshold`: `3e-4`

Defaults for `trust_region` (only used with `TrustRegionAugmentedHessian`):

- `initial_radius`: 0.4
- `max_radius`: 1e5
- `min_radius`: 1e-5
- `increase_factor`: 1.2
- `decrease_factor`: 0.7
- `constrict_factor`: 0.1
- `increase_threshold`: 0.75
- `decrease_threshold`: 0.25
- `rejection_threshold`: 0.0

`hessian_guess` defaults to `Identity` for Cartesian coordinates, otherwise `ScaledIdentity`.

`lbfgs_keywords` defaults (only used with `LBFGS`):

- `linesearch`: `BacktrackingStrongWolfe`
- `n_corrections`: 6
- `epsilon`: 1e-5
- `max_linesearch`: 40
- `gtol`: 0.9

Guidance from upstream docs and libqdx comments:

- `convergence_criteria.metric`:
  - `Baker`: max gradient component must be within threshold and either delta energy or step component must be within their thresholds.
  - `GradientOnly`: only the gradient threshold is enforced.
- `convergence_criteria` units: `gradient_threshold` (Eh/a0), `delta_energy_threshold` (Eh), `step_component_threshold` (a0).
- `coordinate_system`: `DelocalisedInternal` is the default and strongly recommended; `Cartesian` and `NaturalInternal` are available.
- `hessian_guess`: identity, scaled identity (default and recommended), Schlegel, and Lindh. Upstream docs caution that the non-default models are not recommended for general use.
- `algorithm`: `EigenvectorFollowing` is recommended; `TrustRegionAugmentedHessian` is available but not recommended for most users.
- `optimizer_reset_interval` is an expert feature: every N iterations EXESS will regenerate the coordinate system and reset the Hessian; if omitted, it never resets.
- `constraints` support constrained bond lengths, angles, and dihedrals (lists of atom indices).
- `frozen_distance_slippage_tolerance_angstroms` and `frozen_angle_slippage_tolerance_degrees` control expected slippage in frozen delocalized coordinates.

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

Notes:

- Upstream docs list 1 fs (0.001 ps) as a typical default for `dt`; the schema requires that you set `dt` explicitly.
- `use_async_timesteps` is an expert keyword; use with care.

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

Boundary conditions are specified per axis; `range.lower`/`range.upper` define the box extent for `Periodic` boundaries.

## force_field

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `ff_filename` | string | required | Force field filename path. |

`force_field` is used for classical MM components (e.g., solvent in AIMD/QMMM workflows).

## log

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `console` | object | defaults | Console log settings. |
| `logfiles` | array | `[]` | File log settings. |

Defaults:

- `console.level`: `LargeInfo` (or `Debug` in debug builds).
- `console.prefix_fmt`: empty string.
- `logfiles.level`: `Verbose`.
- `logfiles.prefix_fmt`: `[%Y-%m-%d %H:%M:%S.{us} r{rank} {level}] `.
- `logfiles.directory`: unset.

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

Upstream docs describe this order as descending verbosity.

## rtat

RTAT is a runtime auto-tuner for matrix operations.

Upstream docs note that RTAT is the open-source `rtatblas` library (https://github.com/csnowdon2/rtatblas). When enabled, EXESS uses it to auto-tune GPU BLAS configurations for matrix operations.

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `enabled` | bool | true | Enable runtime autotuning. |
| `synchronous` | bool | false | Use synchronous operations. |
| `json_file_dump_prefix` | string | unset | Prefix for RTAT JSON dumps. |

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
| `descriptor_grid` | object | unset | Grid for descriptor exports. |

`descriptor_grid` can be one of the following structures (libqdx):

- `standard`: `FINE`, `ULTRAFINE`, `SUPERFINE`, `TREUTLER_GM3`, `TREUTLER_GM5`.
- `params`: `points_per_shell`, `order` (`One` or `Two`), `scale`.
- `regular`: `min`, `max`, `spacing` arrays (Cartesian grid).
- `custom`: flat list of points `[x1, y1, z1, x2, y2, z2, ...]`.

Notes:

- `export_gradient` and `export_expanded_gradient` require a gradient-capable driver (Gradient, Dynamics, QMMM, Optimization).
- `export_hessian`, `export_mass_weighted_hessian`, and `export_hessian_frequencies` require a Hessian calculation.
- `export_expanded_esp_descriptors` is documented as causing memory errors; avoid enabling it for production runs.
- rush-py source comments flag a few exports as undocumented or unclear (e.g., `export_molecular_orbital_coeffs`, `export_relaxed_mp2_density_correction`, `export_mass_weighted_hessian`, `export_hessian_frequencies`, and `export_basis_labels`). `export_bond_orders` is described as a pass-through of input connectivity.

## ks_dft

KSDFT is used when `model.method` is `RestrictedKSDFT`. Upstream docs recommend reading the KSDFT paper (see the citations page) before tuning advanced settings.

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `functional` | string | required | LibXC functional name. |
| `method` | string | `GauXC` | XC evaluation method. |
| `use_C_opt` | bool | true | Use C-matrix optimization (Dense/BatchDense). |
| `grid` | object | default grid (ULTRAFINE) | Numerical grid settings. |
| `sp_threshold` | float | SCF density_threshold | Single-precision threshold. |
| `dp_threshold` | float | SCF density_threshold | Double-precision threshold. |
| `batches_per_batch` | int | 20 | Batch batching for GauXC. |

Grid parameters supported in the libqdx Rust schema:

- `radial_quad`: `MuraKnowles`, `MurrayHandyLaming`, `TreutlerAldrichs`.
- `pruning_scheme`: `ROBUST`, `UNPRUNED`, `TREUTLER`.
- `batch_size`: GauXC batch size.
- `radial_size`, `angular_size`: Custom grid sizes.
- `default_grid`: `FINE`, `ULTRAFINE`, `SUPERFINE`, `TREUTLER_GM3`, `TREUTLER_GM5`.

Defaults (EXESS):

- `grid.default_grid`: `ULTRAFINE`.
- `grid.radial_quad`: `MuraKnowles`.
- `grid.pruning_scheme`: `ROBUST`.
- `grid.batch_size`: 512 (GauXC).
- `dp_threshold`: SCF `density_threshold` if omitted.
- `sp_threshold`: `dp_threshold` if set, otherwise SCF `density_threshold`.

Method options (from upstream docs):

- `GauXC` (default): GPU-accelerated XC evaluation with the broadest support.
- `Dense`: Dense matrix evaluation, roughly O(N^3); suitable for small to medium systems.
- `BatchDense`: Batched dense evaluation; O(N^2) with `use_C_opt=true`, O(N) with `use_C_opt=false`.
- `Direct`: Direct evaluation without storing intermediates.
- `SemiDirect`: Hybrid of direct and batch-dense methods.

`use_C_opt` enables C-matrix based XC evaluation, reducing matrix dimensions from `n_basis` to `n_occ` for Dense/BatchDense methods. Upstream docs note it is only valid for `Dense` and `BatchDense`.

Grid configuration details (from upstream docs):

- Default grid presets via `default_grid`: `FINE`, `ULTRAFINE` (default), `SUPERFINE`, `TREUTLER_GM3`, `TREUTLER_GM5`.
- Custom grid sizes via `radial_size` and `angular_size`.
- `radial_quad`: `MuraKnowles` (default), `MurrayHandyLaming`, `TreutlerAldrichs`.
- `pruning_scheme`: `ROBUST` (default), `UNPRUNED`, `TREUTLER`.
- Batching:
  - Closest-atom batching (default when no batch settings are provided).
  - `octree`: `max_size` (default 512), `max_depth` (default unlimited), `max_distance` (default unlimited).
  - `space_filling`: `octree` parameters plus `target_batch_size` (default 1024).
  - `batch_size`: GauXC batch size (default 512).

Examples:

Minimal KSDFT:

```json
"ks_dft": {
  "functional": "GGA_XC_PBE"
}
```

Custom grid defaults:

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

Octree batching:

```json
"ks_dft": {
  "functional": "HYB_GGA_XC_B3LYP",
  "method": "BatchDense",
  "grid": {
    "octree": {
      "max_size": 512
    }
  }
}
```

Functional notes from upstream docs:

- Meta-GGA functionals are experimental and supported only with `GauXC`.
- Range-separated functionals are not supported.
- Only `B2PLYP` and `revDSD-PBEP86-D4` double hybrids are implemented; D4 must be added externally.
- For a full list of functionals, see the LibXC documentation: https://www.tddft.org/programs/libxc/functionals/

Grid guidance from upstream docs:

- Default grid settings (ULTRAFINE with ROBUST pruning) provide a good accuracy/cost balance for most users.
- SUPERFINE grids can improve accuracy but significantly increase compute time.
- Octree batching with BatchDense is useful for large systems where linear scaling is critical.

## integrals

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `scheduler` | string | `Callback` | `Callback` or `RoundRobin`. |
| `n_streams` | int | 4 (CUDA) / 1 (HIP) | GPU stream count. |

If `integrals` is omitted entirely, EXESS uses `Callback` with 4 streams.

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
| `pressure_atm` | float | unset | If set, runs NPT; otherwise NVT. |
| `minimisation` | object | unset | Classical minimisation settings. |
| `trajectory` | object | unset | Trajectory output settings. |
| `energy_csv` | string | unset | Path for energy CSV. |
| `restraints` | object | unset | Restraints for atoms/fragments. |

If `minimisation` is provided, defaults are:

- `err_tol_kj_per_mol_nm`: 10
- `max_iterations`: 0

If `trajectory` is provided, defaults are:

- `format`: `JSON`
- `interval`: 1
- `start`: 0
- `end`: max u32
- `include_waters`: false

If `restraints` is provided, defaults are:

- `k`: 2000.0
- `fix_heavy`: false

Notes:

- `pressure_atm`: if set, EXESS runs NPT; if unset, NVT is used.
- `trajectory.format` can be `JSON` or `XYZ` (default `JSON`).
- `trajectory.include_waters` can be set to omit waters for smaller trajectories.
- `restraints` are mutually exclusive across fixed/free atom/fragment lists; set `free_atoms = []` to fix all atoms.
- `restraints.k` scales the restraint force; larger values mean stronger restraints.
- In the rush-py interface, fragment lists obey these rules: if two of `qm_fragments`, `mm_fragments`, `ml_fragments` are provided, the remainder is inferred; if all three are provided, each fragment must be in exactly one list; providing only one list is invalid.

## regions

`regions` defines which fragments are treated as QM, MM, or ML (Q4ML/QMMM workflows):

- `qm_fragments`: array[int]
- `mm_fragments`: array[int]
- `ml_fragments`: array[int]

If omitted, the EXESS JSON parser sets `mm_fragments` and `ml_fragments` to empty lists and leaves `qm_fragments` unset.

## debug

| Keyword | Type | Default | Brief |
| --- | --- | --- | --- |
| `dry_run` | bool | false | Validate fragment queue without computing. |
| `print_subfragment_xyz` | bool | false | Print subfragment XYZ for SSFD. |
| `max_fragments` | int | -1 | Limit number of fragments computed. |
| `ignore_fragments` | bool | false | Ignore fragmentation (developer validation). |
| `skip_calcs` | bool | false | Skip computations in fragmentation routines. |

Notes:

- `dry_run` runs queue construction only (no computation) to validate fragment counts and detect input issues.
- `print_subfragment_xyz` prints subfragment geometries for SSFD debugging.
- `max_fragments` can be used to limit the number of fragments evaluated for non-covalent systems; the default `-1` means "use all fragments."
- `ignore_fragments` forces a full-system calculation for validation.
- `skip_calcs` skips calculations during fragmentation to debug queue construction performance.
## Rush-py defaults

Rush-py sets some defaults in Python before submitting a run. If a `*_keywords` argument is omitted, rush-py may pass `None` (no overrides) or construct a default object.

Default keyword behavior for the common entry points:

- `exess.exess` / `exess.energy` / `exess.interaction_energy`:
  - `scf_keywords`: unset (EXESS defaults apply).
  - `frag_keywords`: `FragKeywords()` (level `Dimer`, `dimer_cutoff=100.0`, `trimer_cutoff=None`, `tetramer_cutoff=None`, `cutoff_type=None`, `distance_metric=None`).
  - `export_keywords`: `ExportKeywords()` (all fields unset; no exports requested).
- `exess.chelpg`:
  - `scf_keywords`: `SCFKeywords(max_diis_history_length=12, convergence_threshold=1e-8)`.
  - `frag_keywords`: `FragKeywords(level="Monomer")`.
  - `export`: CHELPG charges and bond orders enabled.
- `exess.qmmm`:
  - `scf_keywords`: unset (EXESS defaults apply).
  - `frag_keywords`: `FragKeywords()` (same defaults as above).
  - `trajectory`: `Trajectory()` (all fields unset; EXESS defaults apply).
- `exess.optimization`:
  - `optimization_keywords`: `OptimizationKeywords()` (all fields unset; EXESS defaults apply), with required `max_iters` passed separately.

Rush-py entrypoint defaults (non-keyword parameters):

- `exess.exess` / `exess.energy` / `exess.interaction_energy`:
  - `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`.
- `exess.chelpg`:
  - `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`.
  - Overrides `standard_orientation="None"` and `force_cartesian_basis_sets=false`.
- `exess.qmmm`:
  - `method="RestrictedHF"`, `basis="STO-3G"`, `aux_basis=None`.
  - `dt_ps=0.002`, `temperature_kelvin=290.0`, `pressure_atm=None`.
- `exess.optimization`:
  - `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`.

`FragKeywords` defaults by level in rush-py:

- `Monomer`: `dimer_cutoff=100.0`, `trimer_cutoff=None`, `tetramer_cutoff=None`, `cutoff_type=None`, `distance_metric=None`.
- `Dimer`: `dimer_cutoff=100.0`, `trimer_cutoff=None`, `tetramer_cutoff=None`.
- `Trimer`: `dimer_cutoff=100.0`, `trimer_cutoff=25.0`, `tetramer_cutoff=None`.
- `Tetramer`: `dimer_cutoff=100.0`, `trimer_cutoff=25.0`, `tetramer_cutoff=10.0`.
