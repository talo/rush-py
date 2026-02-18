# Examples

These examples progress from minimal inputs to production-style workflows. They use the EXESS JSON input format, which is documented in the {doc}`EXESS input reference <input>`, and reference keywords defined in the {doc}`keyword reference <keywords>`.

## Start here: first calculation

Minimal RHF energy input (single topology):

```json
{
  "topologies": [
    { "xyz": "/path/to/water.xyz" }
  ],
  "driver": "Energy",
  "model": {
    "method": "RestrictedHF",
    "basis": "cc-pVDZ",
    "aux_basis": "cc-pVDZ-RIFIT"
  },
  "keywords": {}
}
```

Run with the EXESS CLI or rush-py:

```bash
module load exess
runexess input.json -g NGPUS
```

```python
from rush import exess

res = exess.energy("input_topology.json", collect=True)
```

## Full-system basics

### RHF energy (single system)

```json
{
  "topologies": [
    {
      "geometry": [
        -4.3997, 1.0764, 7.7009,
        -3.9597, 0.2664, 7.3109,
        -4.4297, 0.9964, 8.7009
      ],
      "symbols": ["O", "H", "H"]
    }
  ],
  "driver": "Energy",
  "model": {
    "method": "RestrictedHF",
    "basis": "cc-pVDZ",
    "aux_basis": "cc-pVDZ-RIFIT"
  },
  "keywords": {
    "scf": {
      "max_iters": 100,
      "max_diis_history_length": 8,
      "convergence_threshold": 1e-10,
      "density_threshold": 1e-10,
      "convergence_metric": "Energy"
    }
  }
}
```

### RI-HF (via `fock_build_type`)

RI-HF is configured by using `RestrictedHF` with `fock_build_type: "RI"`:

```json
{
  "topologies": [
    {
      "geometry": [
        -4.3997, 1.0764, 7.7009,
        -3.9597, 0.2664, 7.3109,
        -4.4297, 0.9964, 8.7009
      ],
      "symbols": ["O", "H", "H"]
    }
  ],
  "driver": "Energy",
  "model": {
    "method": "RestrictedHF",
    "basis": "cc-pVDZ",
    "aux_basis": "cc-pVDZ-RIFIT"
  },
  "keywords": {
    "scf": {
      "max_iters": 100,
      "fock_build_type": "RI",
      "max_diis_history_length": 8,
      "convergence_threshold": 1e-10,
      "density_threshold": 1e-10,
      "convergence_metric": "Energy"
    }
  }
}
```

### RI-MP2 energy

```json
"model": {
  "method": "RestrictedRIMP2",
  "basis": "cc-pVDZ",
  "aux_basis": "cc-pVDZ-RIFIT"
}
```

### KSDFT example (PBE)

```json
{
  "topologies": [
    {
      "geometry": [
        -4.3997, 1.0764, 7.7009,
        -3.9597, 0.2664, 7.3109,
        -4.4297, 0.9964, 8.7009
      ],
      "symbols": ["O", "H", "H"]
    }
  ],
  "driver": "Energy",
  "model": {
    "method": "RestrictedKSDFT",
    "basis": "cc-pVDZ"
  },
  "keywords": {
    "scf": {
      "max_iters": 50,
      "max_diis_history_length": 8,
      "convergence_threshold": 1e-6,
      "convergence_metric": "Energy"
    },
    "ks_dft": {
      "functional": "GGA_XC_PBE"
    }
  }
}
```

### Gradient calculation

```json
"driver": "Gradient",
"model": {
  "method": "RestrictedHF",
  "basis": "cc-pVDZ",
  "aux_basis": "cc-pVDZ-RIFIT"
}
```

### Batched input (multiple topologies)

```json
{
  "topologies": [
    { "xyz": "/path/to/water_1.xyz" },
    { "xyz": "/path/to/water_2.xyz" }
  ]
}
```

## Fragmentation workflows (MBE)

### Fragmented MBE energy (dimer level)

```json
{
  "topologies": [
    {
      "geometry": [
        0.0, -1.207, 1.207,
        -0.365, -2.012, 1.575,
        0.045, -1.364, 0.264,
        0.232, 1.076, -1.308,
        0.828, 1.353, -2.003,
        -0.599, 1.509, -1.506,
        -2.274, 1.303, 0.971,
        -1.737, 1.624, 1.696,
        -1.829, 0.513, 0.664,
        -2.272, -1.224, -1.322,
        -2.136, -0.414, -1.813,
        -2.265, -0.961, -0.402,
        0.935, 1.982, 1.885,
        1.347, 1.122, 1.814,
        1.289, 2.361, 2.689
      ],
      "symbols": [
        "O", "H", "H",
        "O", "H", "H",
        "O", "H", "H",
        "O", "H", "H",
        "O", "H", "H"
      ],
      "fragments": [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]
      ],
      "connectivity": [],
      "fragment_formal_charges": [0, 0, 0, 0, 0]
    }
  ],
  "model": {
    "method": "RestrictedHF",
    "basis": "STO-3G"
  },
  "keywords": {
    "scf": {
      "max_iters": 40,
      "max_diis_history_length": 12,
      "convergence_threshold": 1e-6,
      "convergence_metric": "Energy"
    },
    "frag": {
      "cutoff_type": "Centroid",
      "level": "Dimer",
      "cutoffs": {
        "dimer": 40,
        "trimer": 30
      }
    }
  },
  "driver": "Energy"
}
```

In this example, `fragments` is an array of arrays holding zero-indexed atom IDs, and `fragment_formal_charges` lists the charge for each fragment:

```json
"fragments": [
  [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]
],
"fragment_formal_charges": [0, 0, 0, 0, 0]
```

The `frag` group controls truncation level, distance cutoffs, and the distance metric (`Centroid` vs `ClosestPair`). See the keyword reference for guidance.

### Interaction (lattice) energy

Add a `reference_fragment` under `frag`:

```json
"frag": {
  "reference_fragment": 0,
  "cutoff_type": "Centroid",
  "level": "Dimer",
  "cutoffs": {
    "dimer": 40,
    "trimer": 30
  }
}
```

This produces an interaction (lattice) energy between the reference fragment and the rest of the system. The usual convention applies: negative values indicate binding, positive values indicate repulsion.

## Dynamics (AIMD)

EXESS can run Born-Oppenheimer AIMD using RHF, RI-HF, or RI-MP2 gradients (fragmented or full-system). Switch the driver to `Dynamics` and add a `dynamics` block:

```json
{
  "topologies": [
    {
      "geometry": [
        0.0, -1.207, 1.207,
        -0.365, -2.012, 1.575,
        0.045, -1.364, 0.264,
        0.232, 1.076, -1.308,
        0.828, 1.353, -2.003,
        -0.599, 1.509, -1.506,
        -2.274, 1.303, 0.971,
        -1.737, 1.624, 1.696,
        -1.829, 0.513, 0.664,
        -2.272, -1.224, -1.322,
        -2.136, -0.414, -1.813,
        -2.265, -0.961, -0.402,
        0.935, 1.982, 1.885,
        1.347, 1.122, 1.814,
        1.289, 2.361, 2.689
      ],
      "symbols": [
        "O", "H", "H",
        "O", "H", "H",
        "O", "H", "H",
        "O", "H", "H",
        "O", "H", "H"
      ],
      "fragments": [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]
      ],
      "connectivity": [],
      "fragment_formal_charges": [0, 0, 0, 0, 0]
    }
  ],
  "model": {
    "method": "RestrictedRIMP2",
    "basis": "cc-pVDZ",
    "aux_basis": "cc-pVDZ-RIFIT"
  },
  "keywords": {
    "scf": {
      "max_iters": 40,
      "max_diis_history_length": 12,
      "convergence_threshold": 1e-6,
      "convergence_metric": "Energy"
    },
    "frag": {
      "cutoff_type": "Centroid",
      "level": "Dimer",
      "cutoffs": {
        "dimer": 40,
        "trimer": 30
      }
    },
    "dynamics": {
      "n_timesteps": 10,
      "dt": 0.002
    }
  },
  "driver": "Dynamics"
}
```

`dt` is in ps (1 fs is 0.001 ps); a typical choice is 1 fs.

## Geometry optimization

Geometry optimization uses RHF, RI-HF, or RI-MP2 gradients. A simple RI-MP2 optimization example:

```json
{
  "topologies": [
    {
      "geometry": [
        -0.75, 0.452, -0.417, -0.696, -0.588, 0.609, 0.82, -0.678, 0.537, 0.892,
        0.417, -0.428, -1.285, 1.273, 0.066, -1.328, 0.08, -1.263, -1.225,
        -1.507, 0.366, -1.029, -0.162, 1.555, 1.158, -1.594, 0.054, 1.31,
        -0.477, 1.488, 1.432, 0.009, -1.29, 1.506, 1.236, -0.056
      ],
      "symbols": ["C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H"]
    }
  ],
  "driver": "Optimization",
  "model": {
    "method": "RestrictedRIMP2",
    "basis": "cc-pVDZ",
    "aux_basis": "cc-pVDZ-RIFIT",
    "standard_orientation": "None"
  },
  "keywords": {
    "scf": {
      "fock_build_type": "RI",
      "max_iters": 100,
      "max_diis_history_length": 8,
      "convergence_threshold": 1e-10,
      "convergence_metric": "DIIS"
    },
    "optimization": {
      "max_iters": 100,
      "coordinate_system": "DelocalisedInternal"
    }
  }
}
```

## Dynamics and optimization

### Ab initio MD (Dynamics)

```json
{
  "topologies": [
    {
      "geometry": [
        0.0, -1.207, 1.207,
        -0.365, -2.012, 1.575,
        0.045, -1.364, 0.264,
        0.232, 1.076, -1.308,
        0.828, 1.353, -2.003,
        -0.599, 1.509, -1.506,
        -2.274, 1.303, 0.971,
        -1.737, 1.624, 1.696,
        -1.829, 0.513, 0.664,
        -2.272, -1.224, -1.322,
        -2.136, -0.414, -1.813,
        -2.265, -0.961, -0.402,
        0.935, 1.982, 1.885,
        1.347, 1.122, 1.814,
        1.289, 2.361, 2.689
      ],
      "symbols": [
        "O", "H", "H",
        "O", "H", "H",
        "O", "H", "H",
        "O", "H", "H",
        "O", "H", "H"
      ],
      "fragments": [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]
      ],
      "connectivity": [],
      "fragment_formal_charges": [0, 0, 0, 0, 0]
    }
  ],
  "model": {
    "method": "RestrictedRIMP2",
    "basis": "cc-pVDZ",
    "aux_basis": "cc-pVDZ-RIFIT"
  },
  "keywords": {
    "scf": {
      "max_iters": 40,
      "max_diis_history_length": 12,
      "convergence_threshold": 1e-6,
      "convergence_metric": "Energy"
    },
    "frag": {
      "cutoff_type": "Centroid",
      "level": "Dimer",
      "cutoffs": {
        "dimer": 40,
        "trimer": 30
      }
    },
    "dynamics": {
      "n_timesteps": 10,
      "dt": 0.002
    }
  },
  "driver": "Dynamics"
}
```

### Geometry optimization

```json
{
  "topologies": [
    {
      "geometry": [
        -0.75, 0.452, -0.417, -0.696, -0.588, 0.609, 0.82, -0.678, 0.537, 0.892,
        0.417, -0.428, -1.285, 1.273, 0.066, -1.328, 0.08, -1.263, -1.225,
        -1.507, 0.366, -1.029, -0.162, 1.555, 1.158, -1.594, 0.054, 1.31,
        -0.477, 1.488, 1.432, 0.009, -1.29, 1.506, 1.236, -0.056
      ],
      "symbols": ["C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H"]
    }
  ],
  "driver": "Optimization",
  "model": {
    "method": "RestrictedRIMP2",
    "basis": "cc-pVDZ",
    "aux_basis": "cc-pVDZ-RIFIT",
    "standard_orientation": "None"
  },
  "keywords": {
    "scf": {
      "fock_build_type": "RI",
      "max_iters": 100,
      "max_diis_history_length": 8,
      "convergence_threshold": 1e-10,
      "convergence_metric": "DIIS"
    },
    "optimization": {
      "max_iters": 100,
      "coordinate_system": "DelocalisedInternal"
    }
  }
}
```

## Rush-py workflows

### Rush-py wrapper example

```python
from rush import exess

res = exess.energy(
    "input_topology.json",
    collect=True,
)
```

For exports, descriptor grids, and output downloading, see the [EXESS exports tutorial](../tutorials/02-exess-exports).
