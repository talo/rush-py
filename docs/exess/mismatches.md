# Layer mismatches and alignment notes

This section documents mismatches across the EXESS executable schema (libqdx.hpp), the libqdx Rust types, and the rush-py interface. It includes mismatches found in the original docs, as well as differences discovered by inspecting code.

**Schema vs docs**
- `model.method` in the EXESS schema supports `RestrictedKSDFT`, `UnrestrictedRIMP2`, and `RestrictedRICCSD`, while older docs list only `RestrictedHF`, `UnrestrictedHF`, and `RestrictedRIMP2`.
- RI-HF is documented as `RestrictedRIHF`, but the schema does not include this method; RI-HF is controlled via `scf.fock_build_type = "RI"`.
- `driver` supports `Hessian` and `QMMM` in the schema; older docs list only Energy/Gradient/Optimization/Dynamics.
- `frag.level` is a string enum (`Monomer`..`Octamer`) in the schema; older docs list integer levels 1-4.
- `frag.cutoff_type` is `Centroid` or `ClosestPair` in the schema; older docs also mention `MinimalDistance`.
- `frag.distance_metric` exists in the schema (`Max`, `Average`, `Min`, `Ryan`) but is undocumented upstream.
- `dynamics` is a top-level keyword group in the schema; some upstream examples place it under `frag`.
- `force_field` is the schema key for forcefield settings; older docs use `ff`.
- `optimization` uses `trust_region` in the schema, while earlier docs use `trust_region_keywords`.
- SCF default values differ: the schema defaults `max_iters` to 50, whereas older docs state 30.
- `dynamics.use_async_timesteps` defaults to true in the schema; older docs indicate false.

**libqdx C++ vs libqdx Rust**
- `frag.distance_metric` defaults to `Max` in libqdx.hpp but defaults to `Average` in libqdx Rust.
- `density_basis_set_projection_fallback_enabled` defaults are explicit in libqdx Rust but optional in libqdx.hpp.
- `FragmentDistanceMetric::Ryan` exists only in libqdx.hpp.
- `Topology` includes `waters` (water range) and `stereochemistry` in libqdx.hpp; these fields are not present in the Rust `Topology` struct.
- `GuessSCF` uses `bf_cutoff_threshold` in libqdx.hpp but `bt_cutoff_threshold` in libqdx Rust.
- `ssfd_only_converge_in_bsp_basis` has a trailing space in the libqdx.hpp JSON key (`"ssfd_only_converge_in_bsp_basis "`).
- `ks_dft.grid` is free-form JSON in libqdx.hpp, but the Rust schema uses a fixed `XCGridParameters` struct (no octree/space-filling fields).

**rush-py vs schema**
- rush-py exposes a limited `MethodT` list and does not include `RestrictedRICCSD`.
- rush-py `BasisT` / `AuxBasisT` lists are narrower than the EXESS basis set lists.
- rush-py `Model` only exposes `standard_orientation` and `force_cartesian_basis_sets`; `method`/`basis` are function parameters.
- rush-py does not expose `external_charges`, `rtat`, `integrals`, or `ks_dft` keyword groups.
- rush-py `StandardDescriptorGrid` accepts `SG1`/`SG2`, while the schema uses `FINE`/`ULTRAFINE`/`SUPERFINE`/`TREUTLER_GM*`.
- rush-py `OptimizationKeywords` does not pass `constraints` to REX (marked TODO in code).
- rush-py `Topology` does not expose `stereochemistry`, `fragment_multiplicities`, or `waters`.
- rush-py `BondOrder` only covers a subset of EXESS bond-order values and remaps legacy values 254/255.
- rush-py uses Topology paths rather than full EXESS input JSON, so `schema_version`, `title`, and some top-level input fields are not directly set by users.
