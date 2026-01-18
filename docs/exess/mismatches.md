# Layer mismatches and alignment notes

This section documents mismatches across the three layers (EXESS executable, the tengu/libqdx layer, and the rush-py interface) based on the available documentation. It does not yet include findings from code or schema inspections.

## Documented mismatches

- **Method list differences**: `model.method` is documented as `RestrictedHF`, `UnrestrictedHF`, and `RestrictedRIMP2` in the manual, but the examples and capabilities tables also reference `RestrictedRIHF` and `RestrictedKSDFT`.
- **Fragmentation level type**: The manual lists `frag.level` as an integer with options `[1, 2, 3, 4]`, while examples use string values like `Dimer`, `Trimer`, and `Tetramer`.
- **Fragment cutoff type names**: The manual lists `cutoff_type` options as `Centroid` and `MinimalDistance`, while other examples mention `ClosestPair`.
- **Dynamics keyword location**: The keyword reference documents `dynamics` as a top-level `keywords` group, but the AIMD example nests it under `keywords.frag`.
- **GPU/team settings**: The `system` section documents `teams_per_node` and `gpus_per_team`, while some examples and environment variables refer to `ngpus_per_node` and `MBE_NGPUS`.
- **Basis set lists**: The supported basis sets list differs between EXESS/docs/basis_sets.md and the docs_exess manual (def2 sets and formatting of RIFIT basis names). The reference page lists both.
- **Descriptor grid naming**: The export keyword `descriptor_grid` is documented as `StandardGrid`, `GridParams`, or raw points. The rush-py tutorial references `StandardDescriptorGrid`, `DescriptorGrid`, `RegularDescriptorGrid`, and `CustomDescriptorGrid` classes.
- **Output location and access**: EXESS writes JSON and HDF5 to paths controlled by environment variables, while rush-py returns object store paths and uses `save_*_outputs` helpers to download files. This is an expected layer difference but affects user workflows.

## Gaps to verify later

- **tengu/libqdx layer docs**: The tengu-exess README is minimal and libqdx does not include user-facing EXESS documentation. Any schema differences between the Rust types and the EXESS input/output formats should be validated against code or schema definitions in a future pass.
- **Keyword availability**: Some keywords are documented with comments or placeholders (for example `store_ri_b_on_host`). These should be validated against the actual supported inputs.
- **Known issue in rush-py tutorial**: The rush-py exports tutorial notes that `expanded_esp_descriptors` can crash with an OOM error. This is a documented limitation but should be verified upstream.
