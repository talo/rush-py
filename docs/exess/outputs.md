# Outputs

EXESS produces JSON outputs for all calculations, and optional HDF5 outputs when exports are requested. In the Rush Python client, outputs are returned as object store paths; use `save_*_outputs` helpers to download them locally.

## Rush output objects

Rush returns outputs as objects with a path, size, and format:

```json
{
  "path": "UUID_OBJECT_STORE_PATH",
  "size": 0,
  "format": "json"
}
```

For EXESS energy calculations, the first output is the JSON result. If any exports are requested, a second output is an HDF5 file containing the exported data.

## JSON output structure

High-level structure:

```
IndividualCalculation
|-- schema_version
|-- calculation_time
|-- qmmbes
|-- trajectory
`-- trajectory_qmmbes

QMMBE
|-- schema_version
|-- num_iters
|-- method
|-- distance_metric
|-- nmers
|-- reference_fragment
|-- expanded_hf_energy
|-- expanded_mp2_ss_correction
|-- expanded_mp2_os_correction
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
|-- mulliken_charges
|-- chelpg_charges
|-- fragment_distance
|-- bond_orders
|-- h_caps
|-- num_iters
`-- num_basis_fns

Trajectory
|-- parent
|-- step
|-- simulation_time
|-- time_delta
|-- kinetic_energy
|-- positions
|-- velocities
|-- forces
`-- changeset

Changeset
|-- sub
`-- add
```

### IndividualCalculation

| Name | Type | Brief |
| --- | --- | --- |
| `calculation_time` | double | Total calculation time (seconds). |
| `qmmbes` | optional array of QMMBE | List of QM outputs, one per topology. Not present for `Dynamics`. |
| `trajectory` | optional Trajectory | Present for `Dynamics` calculations. |
| `trajectory_qmmbes` | optional array of QMMBE | QM outputs for each frame in a trajectory. |

### QMMBE

| Name | Type | Brief |
| --- | --- | --- |
| `method` | string | Level of theory. |
| `distance_metric` | optional string | Distance metric (if fragmented). |
| `nmers` | array of array of Nmer | `nmers[0]` monomers, `nmers[1]` dimers, etc. |
| `reference_fragment` | optional int | Reference fragment for interaction energy. |
| `expanded_hf_energy` | optional double | Total HF or interaction energy. |
| `expanded_mp2_ss_correction` | optional double | MP2 same-spin correction (total or interaction). |
| `expanded_mp2_os_correction` | optional double | MP2 opposite-spin correction (total or interaction). |
| `expanded_density` | optional Tensor64 | Full-system density matrix (square or packed). |
| `expanded_hf_gradients` | optional array of double | Full-system HF gradients, ordered by topology atom order. |
| `expanded_mp2_gradients` | optional array of double | Full-system MP2 gradients, ordered by topology atom order. |

### Nmer

| Name | Type | Brief |
| --- | --- | --- |
| `fragments` | array[int] | Fragment IDs in this n-mer. |
| `density` | optional Tensor64 | Density matrix for the n-mer (square or packed). |
| `fock` | optional Tensor64 | Fock matrix (square or packed). |
| `overlap` | optional Tensor64 | Overlap matrix (square or packed). |
| `h_core` | optional Tensor64 | Core Hamiltonian (square or packed). |
| `coeffs_initial` | optional Tensor64 | Initial coefficient matrix. |
| `coeffs_final` | optional Tensor64 | Final coefficient matrix. |
| `molecular_orbital_energies` | optional array[double] | MO energies. |
| `hf_gradients` | optional array[double] | HF gradients, ordered by fragment then atom. |
| `mp2_gradients` | optional array[double] | MP2 gradients, ordered by fragment then atom. |
| `hf_energy` | optional double | HF energy for the n-mer. |
| `mp2_ss_correction` | optional double | MP2 same-spin correction. |
| `mp2_os_correction` | optional double | MP2 opposite-spin correction. |
| `delta_hf_energy` | optional double | HF delta energy (e.g., dimer interaction). |
| `delta_mp2_ss_correction` | optional double | MP2 same-spin delta correction. |
| `delta_mp2_os_correction` | optional double | MP2 opposite-spin delta correction. |
| `mulliken_charges` | optional array[double] | Mulliken charges. |
| `chelpg_charges` | optional array[double] | CHELPG charges. |
| `fragment_distance` | optional double | Dimer distance (only for dimers). |
| `bond_orders` | optional array[array[double]] | Bond order adjacency matrix. |
| `h_caps` | optional array[int] | Indices of hydrogen caps in this fragment (local to the fragment). |
| `num_iters` | int | SCF iterations to converge. |
| `num_basis_fns` | int | Number of basis functions. |

### Trajectory

| Name | Type | Brief |
| --- | --- | --- |
| `parent` | optional int | Parent frame index. |
| `step` | int | Logical step in simulation. |
| `simulation_time` | double | Physical simulation time. |
| `time_delta` | double | Time since previous frame. |
| `kinetic_energy` | double | Total kinetic energy. |
| `positions` | optional array[double] | Atom positions after changeset. |
| `velocities` | optional array[double] | Atom velocities after changeset. |
| `forces` | optional array[double] | Atom forces after changeset. |
| `changeset` | optional Changeset | Changes applied to prior frame. |

### Changeset

| Name | Type | Brief |
| --- | --- | --- |
| `sub` | optional array[int] | Atom indices to remove from previous topology. |
| `add` | optional Topology | Topology to merge into previous frame. |

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

## Post-processing

The EXESS repo includes a Julia-based Ovito exporter for iterative trajectories:

```bash
cd tools/export
julia ovito_exporter.jl -s [PATH_TO_HDF5_OUTPUT]
```

The output is written to `tools/export/output.xyz` by default, with an optional `-d` destination flag.
