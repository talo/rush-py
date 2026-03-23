# Performance & Benchmarking

Computational chemistry workloads vary enormously depending on your inputs and settings. Two single-point energy calculations can differ in cost by several orders of magnitude. This page explains what drives runtime and how to estimate cost for your own use case with a few short test runs.

## Why we can't give you a single number

Runtime is not a fixed property of a calculation type — it is the product of your system size, the method you choose, the basis set, convergence behaviour, and the hardware target your job lands on. A benchmark on a different system with different settings would give you a figure that might be meaningless or actively misleading for your workload.

The most reliable way to estimate cost at scale is to **run a small, representative set of your own inputs and measure directly**.

## What drives runtime

### System size

Number of atoms (or basis functions) is the dominant cost driver. Scaling is non-linear — doubling the system size more than doubles runtime, depending on the method.

### Level of theory

Semi-empirical methods like NN-xTB are orders of magnitude cheaper than DFT, which is itself cheaper than correlated wavefunction methods. Method choice matters more than any other single variable.

### Basis set

For DFT and ab initio methods, a larger basis set increases the number of basis functions and cost substantially — independently of system size.

### Convergence

SCF and geometry optimisation iterations vary per system. A difficult electronic structure or a poor starting geometry can multiply runtime unpredictably.

### Calculation type

Single-point energies, geometry optimisations, frequency calculations, and dynamics all have different cost profiles. Geometry optimisations and dynamics are inherently iterative, so their total cost depends on how many steps are needed.

### Hardware target

Absolute walltime varies with the GPU generation and cluster your job runs on. Scaling behaviour is consistent, but prefactors differ between targets. See {doc}`05-hardware` for the available targets.

## How to benchmark your own workload

This approach requires only a small amount of compute — typically enough to give you a reliable cost estimate before committing to a large batch.

### 1. Pick representative inputs

Select 3–5 systems that span the range of what you intend to run — ideally including your smallest, largest, and a typical case. Avoid cherry-picking easy inputs; the goal is a realistic sample.

### 2. Run with your actual settings

Use the same method, basis set, convergence criteria, and job configuration you plan to use in production. Benchmarking with simplified settings will underestimate real cost.

### 3. Record walltime

For each test run, note the walltime. You can track this by comparing timestamps before and after collection, or by inspecting the `history.json` file written to your workspace (see {doc}`04-general-info`).

### 4. Plot cost versus system size

Even with 3–5 data points you can fit a scaling curve and extrapolate to larger systems. The shape of the curve is predictable for a given method — your test runs supply the real-world prefactors.

### 5. Extrapolate to your full dataset

Use the curve to estimate total walltime for your intended scale. Build in a buffer — outliers (unusual electronic structure, convergence failures) will pull the average up.

## What to measure

Depending on your calculation type, different metrics are most informative:

| Calculation type | Useful metric | Notes |
|---|---|---|
| Single-point energy | Walltime vs number of atoms (or basis functions) | For semi-empirical methods, atom count is sufficient. For DFT, basis set size matters independently. |
| Geometry optimisation | Walltime per optimisation cycle; total cycles to convergence | Cycle count is system-dependent and harder to predict — sample variance will be higher. |
| Conformer generation | Walltime per conformer vs number of atoms | Cost is also sensitive to the number of conformers requested and pruning settings. |

## Capping resource usage with RunSpec

If you want to guard against unexpectedly long or expensive runs — especially useful when benchmarking unfamiliar systems — you can set an explicit walltime limit using `RunSpec`. A job that reaches the limit will be stopped rather than running indefinitely.

```python
from rush.client import RunSpec

# Cap the run at 60 minutes of walltime
spec = RunSpec(walltime=60)

# Pass it to any computation module
result = exess.energy(trc, run_spec=spec, collect=True)
```

`RunSpec` accepts the following parameters:

| Parameter | Type | Description |
|---|---|---|
| `target` | `str` or `None` | Hardware target: `"Bullet"`, `"Bullet2"`, `"Bullet3"`, `"Gadi"`, or `"Setonix"`. Defaults to a randomly chosen Bullet cluster. |
| `walltime` | `int` or `None` | Maximum wall-clock time in minutes. The job is stopped if this limit is reached. |
| `cpus` | `int` or `None` | Number of CPU cores to request. Default is module-specific. |
| `gpus` | `int` or `None` | Number of GPUs to request. Default is module-specific. |
| `nodes` | `int` or `None` | Number of compute nodes. Most single-molecule calculations run on one node; multi-node is relevant for very large systems on supercomputer targets. |
| `storage` | `int` or `None` | Scratch storage allocation. Defaults to 10. |
| `storage_units` | `str` or `None` | Units for storage: `"KB"`, `"MB"`, or `"GB"`. Defaults to `"MB"`. |

> **Tip:** When running a test batch to measure scaling, set a conservative `walltime` limit on each job. This prevents a single unexpectedly expensive system from consuming your entire test allocation, and makes it easier to identify outliers in your dataset.

## Service units

Service units (SUs) are the billing currency of HPC centres. One SU is broadly equivalent to one CPU core-hour, with GPU nodes carrying a multiplier to reflect hardware cost.

> ⚠️ **SU definitions differ between HPC centres** — a SU on Gadi is not the same as a SU on Setonix. When benchmarking, record walltime separately from SU cost. Walltime scales predictably across hardware; SU cost is centre-specific. For large-scale workflows, walltime is the more portable number for planning purposes.
