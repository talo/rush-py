# Overview

EXESS (EXtreme-scale Electronic Structure System, formerly HERMES) is a high-performance, GPU-accelerated electronic structure program designed for large-scale ab initio calculations. Development is led at the Australian National University and the University of Melbourne.

## Capabilities

### Fragmentation methods

EXESS implements the Many Body Expansion (MBE) for both non-covalent and covalent fragmentation. It supports truncation up to fourth order (tetramers). For covalent systems, EXESS can break single bonds using hydrogen capping to account for broken valences. The number of polymers grows combinatorially with the number of fragments, following:

`C(n, k) = n! / (k! * (n - k)!)`

### Levels of theory

| Calculation | RHF | UHF | RI-HF | RI-MP2 | RestrictedKSDFT |
| --- | --- | --- | --- | --- | --- |
| Energy | yes | yes | yes | yes | yes |
| Gradient | yes | no | yes | yes | no |
| Fragmentation | yes | no | yes | yes | yes |

Notes from the upstream docs:
- If gradients are enabled, fragmented gradients are also enabled.
- RHF supports basis sets up to D functions.
- RI-HF is only available for RHF.
- Spherical Gaussians are only supported for RHF-SCF (non-RI) calculations.
- RI-HF/RI-MP2 support up to F functions in the primary basis and up to G in the auxiliary basis.
- KSDFT supports a wide range of exchange-correlation functionals (LDA, GGA, meta-GGA, hybrid) via LibXC.

### General routines

- Geometry optimization (RHF, RI-HF, RI-MP2).
- Born-Oppenheimer ab initio molecular dynamics (microcanonical only; no thermostats or barostats).
- Periodic boundary conditions and classical solvent support (water only) are available for AIMD.

## Limitations

Documented limitations include:
- Up to D functions for non-RI Hartree-Fock.
- Up to G functions for RI methods (RI-HF, RI-MP2).
- Spherical Gaussians only for non-RI HF-SCF.
- Hydrogen capping is the only supported bond-breaking scheme.
- UHF is supported only without RI.
- Plane wave basis sets are not supported.

Hardware guidance and known issues are listed in the [reference page](reference).
