# Reference

## Supported basis sets

The upstream docs contain two overlapping lists. Both are reproduced here for completeness.

### From EXESS/docs/basis_sets.md

- 3-21G
- 4-31G
- 5-21G
- 6-21G
- 6-31G
- 6-311G
- 6-31G(2df,p)
- 6-31G(3df,3pd)
- 6-31G*
- 6-31G**
- 6-31G**-RIFIT (only as aux basis set)
- 6-31+G
- 6-31+G*
- 6-31+G**
- 6-31++G
- 6-31++G*
- 6-31++G**
- 6-311G**-RIFIT (only as aux basis set)
- PCSeg-0
- PCSeg-1
- STO-2G
- STO-3G
- STO-4G
- STO-5G
- STO-6G
- aug-cc-pVDZ-RIFIT (only as aux basis set)
- aug-cc-pVDZ
- aug-cc-pVTZ-RIFIT (only as aux basis set)
- aug-cc-pVTZ
- cc-pVDZ-RIFIT (only as aux basis set)
- cc-pVDZ
- cc-pVTZ-RIFIT (only as aux basis set)
- cc-pVTZ

### From docs_exess manual

- 3-21G
- 4-31G
- 5-21G
- 6-21G
- 6-311G
- 6-311G**-RIFIT
- 6-31G
- 6-31G(2df,p)
- 6-31G(3df,3pd)
- 6-31G*
- 6-31G**-RIFIT
- 6-31G**
- 6-31+G
- 6-31+G*
- 6-31+G**
- 6-31++G
- 6-31++G*
- 6-31++G**
- STO-2G
- STO-3G
- STO-4G
- STO-5G
- STO-6G
- aug-cc-pVDZ / RIFIT
- aug-cc-pVTZ / RIFIT
- cc-pVDZ / RIFIT
- cc-pVTZ / RIFIT
- def2-SVP / RIFIT
- def2-SVPD / RIFIT
- def2-TZVP / RIFIT
- def2-TZVPD / RIFIT
- def2-TZVPP / RIFIT
- def2-TZVPPD / RIFIT
- pcseg-0
- pcseg-1

Differences between these lists are tracked in the mismatches page.

## Lebedev grids

Mapping of Lebedev grid sizes to maximum spherical harmonic degree:

| Grid size | Max degree |
| --- | --- |
| 6 | 3 |
| 14 | 5 |
| 26 | 7 |
| 38 | 9 |
| 50 | 11 |
| 74 | 13 |
| 86 | 15 |
| 110 | 17 |
| 146 | 19 |
| 170 | 21 |
| 194 | 23 |
| 230 | 25 |
| 266 | 27 |
| 302 | 29 |
| 350 | 31 |
| 434 | 35 |
| 590 | 41 |
| 770 | 47 |
| 974 | 53 |
| 1202 | 59 |
| 1454 | 65 |
| 1730 | 71 |
| 2030 | 77 |
| 2354 | 83 |
| 2702 | 89 |
| 3074 | 95 |
| 3470 | 101 |
| 3890 | 107 |
| 4334 | 113 |
| 4802 | 119 |
| 5294 | 125 |
| 5810 | 131 |

## Environment variables

From EXESS docs:

| Name | Brief |
| --- | --- |
| `OMP_NUM_THREADS` | Number of OpenMP threads. |
| `EXESS_HDF5_OUTPUT_PATH` | Directory for HDF5 outputs. |
| `EXESS_OUTPUT_PATH` | Directory for JSON outputs. |
| `MBE_NGPUS` | GPUs per node for fragmented runs; overrides `ngpus_per_node` in input. |
| `USE_COLORED_LOG_LEVELS` | Colorize log levels. |

From the installation docs:

- `EXESS_PATH`: root path for EXESS.
- `EXESS_RECORDS_PATH`: records directory.
- `EXESS_VALIDATION_PATH`: validation directory (used by Julia validation scripts).

## Hardware considerations

### NVIDIA

- Supports NVIDIA GPUs from Tesla (compute capability 70) onward.
- Consumer GPUs with adequate compute capability work, but <6 GB RAM is limiting.
- Supported up to Hopper (compute capability 90).
- CUDA 11.1+ supported.
- NVHPC toolkit supported.

### AMD

- Requires MAGMA with HIP support.
- ROCm 5.7.0 is documented as most stable; newer versions may vary.
- Tested primarily on MI250x (gfx90a).
- ROCm runtime bug can crash large 4-center kernels for gradients; RI-HF can avoid this.

## Known issues

- NVIDIA: no issues listed.
- AMD: out-of-resources errors can occur; reduce `max_gpu_memory_mb` or use RI in `fock_build_type`.

## Reporting issues

Before reporting issues, consult the known issues above. Report bugs to placeholder@rush.exess.co with details on hardware, software, and an input that reproduces the issue.

## Performance

The upstream performance page currently contains a placeholder line: "vroom vroom we're faster than everyone else".

## Release notes (v4.0.0-beta)

### Added

- CODEOWNERS
- Eigenvector Following (EF) optimizer
- Min and max hessian eigenvalues in optimizer debug
- Internal Coord Inferrer removes ill-formed coords
- AssertAllEQ
- Removal of linearly dependent primitives via Gaussian elimination
- Label PRs
- Gradient energy reduction kernel with coalesced reads and unit test
- 3C gradient integrals use the HGP RR scheme
- Performance printing
- Improved rimp2 single gpu
- Small value filtering in BuvP and BiaP
- Hcore initial guess
- Hcore generated SAD guesses
- Restricted Logging
- Spherical harmonics library
- Normalize function for BLAS provider
- Delocalised internal coords (only for unconstrained optimization)
- Validation support for Geometry Optimization
- Grid library lebedev rule
- QNext fast matrix exponentiation algorithm
- BLAS Provider Iamax routine
- BLAS Provider Axpy supports integer tensors
- Alternate RIHF algorithm without recomputation
- 6-31G**-RIFIT and 6-311G**-RIFIT bases
- Improved RI-HF with sparsity utilization
- Split SP and DP Flop Counter
- QNext numerical preconditioned CG step
- Support for single precision gemm
- Orbital matrix converter (AO -> MO block method)
- Allow compressed B for RI-HF grads
- Export of relaxed MP2 density correction matrices per fragment
- QNext orbital energy sorting
- External initial density guess for RHF and UHF individual calculations
- Multi-node HDF5 export capability
- Atomic grid of lebedev grids
- Support for constraints in delocalised coordinate system
- Baker set validation suite

### Fixed

- Optimization in Cartesian Coords now uses the correct coordinates
- Unsynchronized RTAT
- Initialize RTAT correctly
- Increased default runtime for single node validation on Setonix
- Fix summit compilation with ugly flag
- Changelog JSON file tree now up to date
- AIMD H caps
- Log levels
- Calculation types cleanup
- Double factorial
- Fill2CRepulsion, fill_matrix_H2O_631Gs unit test, broken by removal of axial normalisation
- Improved tree printer design
- Double linking of cartesian coords tests
- Hessian updater should be a namespace
- All unit tests fixed
- Test exess script now covers all scripts
- Geometry optimization synchronisation issues
- Geometry optimization incorrect partial derivates of torsion angle coords
- Bond order export does not need to be set to enable geometry optimization in internal coordinates
- Proper error message if an empty topology is provided
- Perlmutter compilation
- Addressed Setonix env change
- Fix cmake handling of module install dir

### Breaking

- Removed `num_iters` from QMMBE as it is redundant
- `hessian_guess_type` -> `hessian_guess`
- `use_internal_coordinates` bool -> `coordinate_system` enum
- Bond order inference now subject to `flatten_symmetric` export option for HDF5 output

### Changed

- Deleted axial correction factors
- Removed exposed private variables from Tensor class
- Pull request template to have a CI checklist
