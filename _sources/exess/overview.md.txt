# Overview

EXESS (the Extreme-scale Electronic Structure System) is a GPU-accelerated quantum chemistry platform developed by QDX and made available through Rush. It is designed to deliver high-accuracy electronic structure calculations at scale, from high-throughput quantum workloads to chemically complex biomolecular environments. EXESS is built for cases where accuracy, throughput, and scalability must coexist.

A defining feature of EXESS is its regions-based workflow model. Regions provide a simple, explicit way to describe "what part of the system is treated with what level of theory," enabling consistent execution across full-QM calculations, fragmentation workflows, QM/MM, QM/ML, and molecular dynamics—without rebuilding the workflow each time.

## Regions: the organizing principle

In EXESS, regions are not just input syntax; they are the central abstraction that makes complex workflows practical and automatable. A regions specification lets you define, in a single place, how a system is partitioned into:

- quantum regions (treated with HF (Hartree-Fock), KS-DFT (Kohn-Sham Density Functional Theory), RI-MP2 (Resolution of Identity Møller-Plesset second-order perturbation theory), CCSD (Coupled Cluster Singles and Doubles) energies, etc.)
- environment regions (treated with classical MM (molecular mechanics) force fields and/or ML (machine learning) potentials)
- fragment definitions and recombination logic for large heterogeneous systems
- dynamic regions for simulations, where the chemically active region can be tracked and simulated over time

This structure is what allows EXESS to move smoothly between throughput-oriented quantum chemistry and large-system simulations, while keeping the workflow explicit, reproducible, and easy to automate.


## High-throughput, multi-GPU quantum chemistry

EXESS provides a high-performance engine for non-fragmented calculations, including HF, KS-DFT, RI-MP2, and GPU-accelerated CCSD energies. These calculations are optimized around GPU-first computational kernels, multi-GPU scaling, and high throughput across large batches of independent jobs.

This supports workflows such as large-scale screening, geometry optimization, and reaction profiling, as well as the systematic generation of high-fidelity quantum mechanical reference data—energies, gradients, and electronic properties—that can be used to train, validate, and benchmark machine-learning models.

EXESS is not a machine-learning model generator. Its role is to generate accurate quantum data at a scale that makes ML training and validation genuinely data-driven rather than data-limited.

## Fragmentation, QM/MM, and fragmentation-based *ab initio* molecular dynamics

For proteins and other large, heterogeneous systems, EXESS uses regions to express scientifically meaningful decompositions of the system and to make correlated QM methods tractable.

Fragmentation workflows partition a large system into chemically meaningful subsystems that are treated quantum mechanically and recombined to recover system-level properties at the target level of theory. These workflows underpin QDX's largest biomolecular demonstrations with EXESS, particularly at MP2-class accuracy on GPUs.

Regions also provide a clean route to QM/MM (quantum mechanics/molecular mechanics) and hybrid embedding: chemically active regions (binding pockets, metal centers, reactive intermediates, covalent warheads) can be treated with high-accuracy QM, while the surrounding environment is treated with MM and/or ML. Because this is expressed through regions, the same setup naturally extends to fragmentation-based *ab initio* molecular dynamics, where long-time-scale simulations become practical by focusing high-level QM where it matters most and using scalable fragmentation/embedding for the remainder.

In drug discovery, this enables mechanistic simulations that capture polarization, charge transfer, bond formation and cleavage, and electronic reorganization in realistic biomolecular environments.

## Integrated quantum–ML workflows on GPUs

EXESS includes native integration with machine-learning potentials within the same regions framework, enabling QM/ML hybrid workflows without breaking the execution model or leaving the GPU-first stack. These ML components are implemented from scratch with GPU performance as a primary design constraint.

The next major addition is *NN-xTB* (Neural Network extended Tight Binding), extending EXESS toward fast semi-empirical quantum models that retain a clear connection to electronic structure theory while benefiting from GPU acceleration and ML-driven parameterization. This complements high-accuracy QM by enabling efficient multi-fidelity workflows inside the same regions-driven pipeline.

## Roadmap: pushing accuracy without losing scalability

EXESS is designed to evolve toward increasingly accurate quantum chemical methods while remaining GPU-native. Today, it supports CCSD energies on GPUs. Future development targets higher-level coupled-cluster methods (including perturbative triples corrections and beyond) and multi-reference approaches—quantum chemical methods designed for systems with strongly correlated electrons—tailored for chemically complex, heterogeneous systems.

All new methods follow the same guiding principle: they are designed natively for GPUs and multi-GPU execution, rather than adapted from CPU-first implementations.

## Positioning

EXESS is built for workflows where quantum chemistry must be a production capability, not a bottleneck. Regions provide the glue that connects high-throughput quantum calculations, fragmentation at biomolecular scale, QM/MM and QM/ML embedding, and fragmentation-based *ab initio* molecular dynamics into a single coherent, automation-ready platform.

Rather than aiming for exhaustive feature coverage, EXESS emphasizes depth, performance, and scientific rigor—so that accurate quantum mechanics can scale both in system size and in workload volume.

## What EXESS solves well

EXESS targets:

- Large molecular systems where full-system QM is too expensive.
- Fragment-based quantum chemistry via a Many-Body Expansion (MBE).
- GPU-accelerated energies, gradients, geometry optimization, and AIMD.
- Mixed-fidelity QM/MM/ML runs where fragments are assigned to regions (QMMM and optimization workflows).
- High-throughput runs where a single input describes multiple topologies.

If your workflow needs plane waves or pseudopotentials, EXESS is not the right tool (see limitations below). Thermostatted/barostatted dynamics are supported in QMMM workflows (NVT/NPT via `qmmm.temperature_kelvin` and `qmmm.pressure_atm`), but fully ab initio AIMD is microcanonical only.

## Core ideas and workflow

EXESS is built around four practical ideas:

1. **GPU-first execution.** Methods and kernels are designed to run efficiently on NVIDIA (CUDA) and AMD (HIP) GPUs.
2. **Fragmentation as a primary scaling strategy.** Many-Body Expansion enables accurate calculations on systems that are otherwise too large for full-system QM.
3. **Region-aware modeling.** The `regions` keyword assigns fragments to QM, MM, or ML partitions for QMMM and optimization workflows, with MM driven by OpenMM force fields and ML driven by AIMNet.
4. **Automation-friendly inputs.** The JSON schema supports batched topologies, which is useful for screening or dataset generation.

This philosophy shapes how you should approach method choice, fragmentation, and performance tuning.