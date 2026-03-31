# Outputs

EXESS produces JSON outputs for all calculations, and optional export outputs when requested. In the Rush Python client, module calls return a `Run` handle; use `run.fetch()` for in-memory objects or `run.save()` to download them locally.

## Rush output objects

Rush returns outputs as objects with a path, size, and format:

```json
{
  "path": "UUID_OBJECT_STORE_PATH",
  "size": 0,
  "format": "json"
}
```

:::{admonition} Why is the size 0?
:class: note
In the rush-py tutorial, the `size` field is currently unused (often set to 0) but is intended to carry the object's size in bytes in future revisions.

For EXESS energy calculations, the first output is the JSON result. If no exports are present, `exports` is `None`. If any exports are requested, a second output contains the exported data. By default it is an HDF5 file, but in rush-py it is automatically converted to JSON. To keep it in its original format, pass `convert_hdf5_to_json=False` to `exess.calculate` or `exess.energy`.

In rush-py, `run.fetch()` converts the main EXESS JSON output into Python dataclasses in memory. `run.save()` downloads the raw output objects to the local workspace and returns an `exess.ResultPaths` object with `calc` and optional `exports` path fields.

The helper uses the object store path as the filename and applies a `.hdf5` or `.json` extension as appropriate.

Tools such as `h5py`, `h5glance`, `h5ls`, and `h5dump` are useful for exploring the HDF5 exports directly.

## JSON output structure

High-level structure:

```
Output
|-- schema_version
|-- calculation_time
|-- qmmbe
|-- trajectory
`-- trajectory_qmmbes

QMMBE
|-- schema_version
|-- num_iters
|-- method
|-- distance_metric
|-- distance_method
|-- nmers
|-- reference_fragment
|-- expanded_hf_energy
|-- classical_water_energy
|-- expanded_mp2_ss_correction
|-- expanded_mp2_os_correction
|-- expanded_ccsd_correction
|-- expanded_density
|-- expanded_hf_gradients
`-- expanded_mp2_gradients

Nmer
|-- schema_version
|-- fragments
|-- density
|-- fock
|-- overlap
|-- h_core
|-- coeffs_initial
|-- coeffs_final
|-- molecular_orbital_energies
|-- hf_gradients
|-- mp2_gradients
|-- hf_energy
|-- mp2_ss_correction
|-- mp2_os_correction
|-- delta_hf_energy
|-- delta_mp2_ss_correction
|-- delta_mp2_os_correction
|-- s_squared_eigenvalue
|-- mulliken_charges
|-- chelpg_charges
|-- fragment_distance
|-- bond_orders
|-- h_caps
|-- num_iters
`-- num_basis_fns

Trajectory
`-- frames

Frame
|-- parent
|-- step
|-- simulation_time
|-- time_delta
|-- kinetic_energy
|-- estimated_temperature
|-- positions
|-- velocities
|-- forces
`-- changeset

Changeset
|-- sub
`-- add
```

For batched inputs (`topologies` length > 1), EXESS returns one `Output` object per topology.

### Output

| Name | Type | Brief |
| --- | --- | --- |
| `schema_version` | string | Output schema version. |
| `calculation_time` | double | Total calculation time (seconds). |
| `qmmbe` | optional QMMBE | QM output for a single topology. Not present for `Dynamics`. |
| `trajectory` | optional Trajectory | Present for `Dynamics` calculations. |
| `trajectory_qmmbes` | optional array of QMMBE | QM outputs for each frame in a trajectory. |

### QMMBE

| Name | Type | Brief |
| --- | --- | --- |
| `schema_version` | string | QMMBE schema version. |
| `method` | string | Level of theory. |
| `distance_metric` | optional string | Distance metric (if fragmented). |
| `distance_method` | optional string | Distance method (centroid vs closest-pair). |
| `nmers` | array of array of Nmer | `nmers[0]` monomers, `nmers[1]` dimers, etc. |
| `reference_fragment` | optional int | Reference fragment for interaction energy. |
| `expanded_hf_energy` | optional double | Total HF or interaction energy. |
| `classical_water_energy` | optional double | Energy contribution from classical waters (if present). |
| `expanded_mp2_ss_correction` | optional double | MP2 same-spin correction (total or interaction). |
| `expanded_mp2_os_correction` | optional double | MP2 opposite-spin correction (total or interaction). |
| `expanded_ccsd_correction` | optional double | CCSD correction (total or interaction). |
| `expanded_density` | optional Tensor64 | Full-system density matrix (square or packed). |
| `expanded_hf_gradients` | optional array of double | Full-system HF gradients, ordered by topology atom order. |
| `expanded_mp2_gradients` | optional array of double | Full-system MP2 gradients, ordered by topology atom order. |
| `num_iters` | int | Total SCF iterations. |

If `reference_fragment` is present, the expanded energies should be interpreted as interaction energies for that fragment. In that mode, `expanded_density` and gradients are omitted. If `expanded_hf_gradients` is present, `expanded_mp2_gradients` is absent (and vice versa).

### Nmer

| Name | Type | Brief |
| --- | --- | --- |
| `schema_version` | string | N-mer schema version. |
| `fragments` | array[int] | Fragment IDs in this n-mer. |
| `density` | optional Tensor64 | Density matrix for the n-mer (square or packed). |
| `fock` | optional Tensor64 | Fock matrix (square or packed). |
| `overlap` | optional Tensor64 | Overlap matrix (square or packed). |
| `h_core` | optional Tensor64 | Core Hamiltonian (square or packed). |
| `coeffs_initial` | optional Tensor64 | Initial coefficient matrix. |
| `coeffs_final` | optional Tensor64 | Final coefficient matrix. |
| `molecular_orbital_energies` | optional array[double] | MO energies; one entry per basis function. |
| `hf_gradients` | optional array[double] | HF gradients, ordered by fragment then atom. |
| `mp2_gradients` | optional array[double] | MP2 gradients, ordered by fragment then atom. |
| `hf_energy` | optional double | HF energy for the n-mer. |
| `mp2_ss_correction` | optional double | MP2 same-spin correction. |
| `mp2_os_correction` | optional double | MP2 opposite-spin correction. |
| `ccsd_correction` | optional double | CCSD correction. |
| `delta_hf_energy` | optional double | HF delta energy (e.g., dimer interaction). |
| `delta_mp2_ss_correction` | optional double | MP2 same-spin delta correction. |
| `delta_mp2_os_correction` | optional double | MP2 opposite-spin delta correction. |
| `s_squared_eigenvalue` | optional double | Spin contamination metric for unrestricted calculations. |
| `mulliken_charges` | optional array[double] | Mulliken charges. |
| `chelpg_charges` | optional array[double] | CHELPG charges. |
| `fragment_distance` | optional double | Dimer distance (only for dimers). |
| `bond_orders` | optional array[array[double]] | Bond order adjacency matrix. |
| `h_caps` | optional array[int] | Indices of hydrogen caps in this fragment (local to the fragment). |
| `num_iters` | int | SCF iterations to converge. |
| `num_basis_fns` | int | Number of basis functions. |

N-mer gradients are ordered by fragment, then by atom order within each fragment. If `hf_gradients` is present then `mp2_gradients` is absent (and vice versa). Hydrogen cap indices are local to the fragment, not to the full-system atom ordering, and are present whenever matrices such as density or fock are exported.

### Trajectory

| Name | Type | Brief |
| --- | --- | --- |
| `frames` | array[Frame] | Frame tree for the trajectory. |

Frames are stored as a tree of divergent paths; frames with smaller `step` values appear before larger steps.

### Frame

| Name | Type | Brief |
| --- | --- | --- |
| `parent` | optional int | Parent frame index. |
| `step` | int | Logical step in simulation. |
| `simulation_time` | double | Physical simulation time. |
| `time_delta` | double | Time since previous frame. |
| `kinetic_energy` | double | Total kinetic energy. |
| `estimated_temperature` | double | Estimated instantaneous temperature. |
| `positions` | optional array[double] | Atom positions after changeset. |
| `velocities` | optional array[double] | Atom velocities after changeset. |
| `forces` | optional array[double] | Atom forces after changeset. |
| `changeset` | optional Changeset | Changes applied to prior frame. |

Positions/velocities/forces are stored after applying each frame's changeset.

### Changeset

| Name | Type | Brief |
| --- | --- | --- |
| `sub` | optional array[int] | Atom indices to remove from previous topology. |
| `add` | optional Topology | Topology to merge into previous frame. |

Changesets are applied in order: remove (`sub`) atoms first, then merge (`add`) the new topology.

## HDF5 output structure

HDF5 exports are organized by topology and fragment level:

```
|-- topology_0
|   |-- dimers
|   |   |-- 0:1
|   |   |   |-- density
|   |   |   |-- relaxed_mp2_density_correction
|   |   |   |-- fock
|   |   |   |-- final_coefficients
|   |   |   |-- gradient
|   |   |   |-- initial_coefficients
|   |   |   |-- mo_energies
|   |   |   |-- overlap
|   |   |   |-- core_hamiltonian
|   |   |   |-- mulliken_charges
|   |   |   |-- bond_orders
|   |   |   |-- hydrogen_caps_vector
|   |   |   |-- descriptor_grid
|   |   |   |-- density_descriptors
|   |   |   `-- esp_descriptors
|   |   `-- ...
|   |-- trimers
|   |-- ...
|   `-- fragmentation
|       |-- mbe_expanded_density
|       `-- mbe_expanded_gradient
|-- topology_1
`-- ...
```

### File naming

For multi-team runs without concatenation, EXESS produces multiple HDF5 files:

```
<calculation_name>.hdf5
<calculation_name>_export_0.hdf5
<calculation_name>_export_1.hdf5
...
```

If `concatenate_hdf5_files` is enabled, the `_export_N` files are merged into `<calculation_name>.hdf5`.

### Exported matrices

`mbe_expanded` matrices are full-system matrices formed by combining fragment matrices with MBE coefficients.

| Name | Dimensions | Brief |
| --- | --- | --- |
| `density` | `{nbas, nbas}` or `{nbas * (nbas + 1) / 2}` | Converged fragment density (square or packed). |
| `relaxed_mp2_density_correction` | `{nbas, nbas}` or packed | MP2 relaxed density correction. |
| `fock` | `{nbas, nbas}` or packed | Converged fragment Fock matrix. |
| `final_coefficients` | `{nbas, nbas}` | Final coefficient matrix. |
| `gradient` | `{natoms, 3}` | Fragment gradients. |
| `initial_coefficients` | `{nbas, nbas}` | Initial coefficient matrix. |
| `mo_energies` | `{nbas}` | MO energies. |
| `overlap` | `{nbas, nbas}` or packed | Overlap matrix. |
| `core_hamiltonian` | `{nbas, nbas}` or packed | Core Hamiltonian. |
| `mulliken_charges` | `{natoms}` | Mulliken charges. |
| `chelpg_charges` | `{natoms}` | CHELPG charges. |
| `bond_orders` | `{natoms, natoms}` or packed | Bond order adjacency matrix. |
| `hydrogen_caps_vector` | `{natoms}` | 0/1 vector for hydrogen caps. |
| `mbe_expanded_density` | `{nbas, nbas}` or packed | Full-system density. |
| `mbe_expanded_gradient` | `{natoms, 3}` | Full-system gradients. |
| `descriptor_grid` | `{ngrid_points, 3}` | Grid points (x, y, z). |
| `descriptor_grid_weights` | `{ngrid_points}` | Grid weights (Lebedev grids only). |
| `density_descriptors` | `{ngrid_points, 2}` | Density values at grid points. |
| `esp_descriptors` | `{ngrid_points, 2}` | ESP values at grid points. |

For descriptor grids exported as JSON, the grid coordinates are transposed: the first array contains the x coordinates, the second the y coordinates, and the third the z coordinates. Grid weights are also exported (often as 1.0 values).

## Descriptor grids (rush-py exports)

Descriptor exports (`export_density_descriptors`, `export_esp_descriptors`) require a `descriptor_grid`. Upstream rush-py guidance recommends avoiding fragmentation for descriptor grids, since fragment-level descriptors complicate interpretation.

Example (regular grid with three points):

```python
from rush import RunOpts, RunSpec, exess

run = exess.energy(
    "input_topology.json",
    frag_keywords=None,
    export_keywords=exess.ExportKeywords(
        export_density_descriptors=True,
        export_esp_descriptors=True,
        descriptor_grid=exess.RegularDescriptorGrid(
            min=[0.0, 0.0, 0.0],
            max=[2.0, 2.0, 2.0],
            spacing=[2.0, 2.0, 2.0],
        ),
    ),
    run_spec=RunSpec(storage=1000, gpus=1),
    run_opts=RunOpts(
        name="Rush-Py Tutorial: EXESS Exports 2",
        tags=["rush-py", "test", "1kuw", "electron density", "ESP"],
    ),
)
```

As rush-py converts EXESS exports to JSON by default, `run.save()` saves the exported data as JSON. Example JSON structure:

```json
{
  "density_descriptors": [
    0.019335589097924273,
    0.03332198726559439,
    0.0001262822899485241
  ],
  "esp_descriptors": [
    -14.73877477683919,
    -11.399947425592524,
    -6.359468034195182
  ],
  "descriptor_grid": [
    [0.0, 1.0, 2.0],
    [0.0, 1.0, 2.0],
    [0.0, 1.0, 2.0]
  ],
  "descriptor_grid_weights": [
    1.0,
    1.0,
    1.0
  ]
}
```

Notes from the rush-py tutorial:

- `export_expanded_esp_descriptors` currently triggers an OOM error; avoid enabling it.
- When fragmentation is enabled, descriptor values are stored per n-mer in HDF5. When converted to JSON, each descriptor type becomes its own top-level key.
- `CustomDescriptorGrid` takes a flat list of coordinates `[x1, y1, z1, x2, y2, z2, ...]`.
- `StandardDescriptorGrid` accepts `SG1` or `SG2`. These appear to be very fine grids with many points (behavior still being documented).
- `DescriptorGrid` uses `points_per_shell`, `order` (`"One"` or `"Two"`), and `scale` (construction details still being documented).
- The EXESS validation tests include other grid construction styles; support and documentation are still pending.

## Post-processing

The EXESS GitHub repository includes a Julia-based Ovito exporter for iterative trajectories:

```bash
cd tools/export
julia ovito_exporter.jl -s [PATH_TO_HDF5_OUTPUT]
```

The output is written to `tools/export/output.xyz` by default, with an optional `-d` destination flag.
