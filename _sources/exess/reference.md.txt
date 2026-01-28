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

## Installation (HPC build notes)

The upstream installation guide targets HPC system administrators building EXESS from source. Key dependencies:

- C/C++ compiler with C++17 support
- CUDA or ROCm compiler
- MPI library
- OpenMP support
- HDF5
- MAGMA with HIP support (AMD systems)

Upstream notes: EXESS has a minimal dependency set, but the team cannot guarantee out-of-the-box builds for non-standard compilers that they cannot test.

Example build on Gadi (NCI):

```bash
module load julia/1.9.1
module load cuda/12.0.0
module load openmpi/4.0.1
module load hdf5/1.12.1
module load gcc/12.2.0
module load cmake/3.24.2
module load intel-mkl/2023.2.0
module load python3/3.10.0

mkdir build
cd build
CUDAARCHS="70;80" cmake -DCMAKE_INSTALL_PREFIX=$PATH_TO_INSTALL ../
make -j install
```

Example build on Setonix (Pawsey):

```bash
module load gcc/12.2.0
module load cray-hdf5/1.12.2.7
module load rocm/5.7.3
module load cmake/3.27.7
module load magma/2.8.0-${custom}
module load craype-accel-amd-gfx90a
module load julia
export MPI_ROOT=$MPICH_DIR
export MPICH_GPU_SUPPORT_ENABLED=1

mkdir build
cd build
cmake .. -DGPU_RUNTIME=HIP -DMPI_ROOT=$MPI_ROOT -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_INSTALL_PREFIX=$PATH_TO_INSTALL ../
make -j install
```

After installation, the runtime requires `EXESS_RECORDS_PATH` plus the `run.sh` and `runexess` scripts; other source tree content can be removed.

Custom installer templates live under `modulefiles/` in the upstream repo (for Gadi/Setonix). GNU is the recommended build environment, but Cray, NVHPC, and Intel compilers are known to work; report compilation issues with compiler/MPI/CUDA/ROCm versions to the EXESS team.

## Hardware considerations

### NVIDIA

- Supports NVIDIA GPUs from Tesla (compute capability 70) onward.
- Consumer GPUs with adequate compute capability work, but <6 GB RAM is limiting.
- Supported up to Hopper (compute capability 90).
- CUDA 11.1+ supported.
- NVHPC toolkit supported.
- Performance scales with the GPU's double-precision throughput.
- If you have access to newer NVIDIA hardware, upstream docs ask that you open an issue.

### AMD

- Requires MAGMA with HIP support.
- ROCm 5.7.0 is documented as most stable; newer versions may vary.
- Tested primarily on MI250x (gfx90a).
- Other gfx architectures are not tested.
- ROCm runtime bug can crash large 4-center kernels for gradients; RI-HF can avoid this.

## Known issues

- NVIDIA: no issues listed.
- AMD: out-of-resources errors can occur; reduce `max_gpu_memory_mb` or use RI in `fock_build_type`. Upstream docs include this example error:

```text
:0:rocdevice.cpp
:2688: 1214497773164 us: [pid:853101 tid:0x14e50c57d700]
Callback: Queue 0x14dbb9800000 Aborting with error :
HSA_STATUS_ERROR_OUT_OF_RESOURCES:
```

## Reporting issues

Before reporting issues, consult the known issues above. Report bugs to placeholder@rush.exess.co with details on hardware, software, and an input that reproduces the issue.

## License

For licensing questions, contact placeholder@qdx.co.

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
