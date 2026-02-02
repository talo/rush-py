# Keyword reference

Keywords live under the top-level `keywords` object. Recognized groups include:

`scf`, `frag`, `ks_dft`, `export`, `regions`, `optimization`, `qmmm`, `gradient`, `guess`, `integrals`, `rtat`, `hessian`, `dynamics`, `boundary`, `machine_learning`, `force_field`, `log`, `debug`.

Defaults in the parameter listings below reflect the EXESS command-line behavior (JSON parser defaults plus EXESS internal defaults). Rush-py defaults are listed at the end of this page.

In practice, you will spend most of your time in `scf`, `frag`, `ks_dft`, and the driver-specific groups (`optimization`, `dynamics`, `qmmm`).

Icon key:

```{eval-rst}
.. raw:: html

   <div class="exess-icon-key">
     <span class="exess-icon-key__item"><span class="param-note param-note--info" aria-hidden="true"></span> Tip</span>
     <span class="exess-icon-key__item"><span class="param-note param-note--expert" aria-hidden="true"></span> Expert</span>
     <span class="exess-icon-key__item"><span class="param-note param-note--experimental" aria-hidden="true"></span> Experimental</span>
     <span class="exess-icon-key__item"><span class="param-note param-note--broken" aria-hidden="true"></span> Known issues</span>
   </div>
```

## Core Electronic-Structure Keywords

(scf)=
### scf

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "scf": {
               "max_iters": 40,
               "max_diis_history_length": 12,
               "convergence_threshold": 1e-8,
               "density_threshold": 1e-11,
               "fock_build_type": "RI"
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from rush.exess import SCFKeywords, energy

         energy(
             topology_path="molecule_t.json",
             scf_keywords=SCFKeywords(
                 max_iters=40,
                 max_diis_history_length=12,
                 convergence_threshold=1e-8,
                 density_threshold=1e-11,
                 fock_build_type="RI",
             ),
         )
```

```{eval-rst}
.. exess-params::

   .. exess-param:: max_iters
      :type: int
      :default: 50
      :brief: Max number of SCF iterations.

   .. exess-param:: max_diis_history_length
      :type: int
      :default: 8
      :brief: Max size of DIIS window.

   .. exess-param:: batch_size
      :type: int
      :default: 2560
      :brief: Shell-pair batches per bin; use multiples of 128.
      :note: info

      Do not go below 128. For details on the shell-pair batch bin container, see 10.1021/acs.jctc.0c00768, 10.1021/acs.jctc.1c00720, and 10.1080/00268976.2022.2112987.

   .. exess-param:: convergence_metric
      :type: string
      :default: DIIS
      :brief: Convergence metric (``DIIS``, ``Energy``, ``Density``).

   .. exess-param:: convergence_threshold
      :type: float
      :default: 1e-6
      :brief: SCF convergence threshold; tighten for higher-order fragmentation or tighter energy targets.
      :note: info

      Suggested values:

      - ``1e-6`` for non-fragmented RHF + RI-MP2 with ``Density``/``DIIS``.
      - ``1e-8`` for non-fragmented RHF + RI-MP2 with ``Energy``.
      - ``1e-6`` for dimer-level RHF + RI-MP2.
      - ``1e-8`` for trimer/tetramer-level calculations with ``DIIS``.
      - ``1e-10`` for large tetramer-level calculations with ``DIIS``.

   .. exess-param:: density_threshold
      :type: float
      :default: 1e-10
      :brief: Density screening threshold; affects SCF cost and accuracy.
      :note: info

      Lower values speed up SCF with potential accuracy loss. Explore ``1e-8`` to ``1e-12`` and validate accuracy; too-large values can lead to NaNs.

      Increasing to ``1e-11`` or ``1e-12`` will slow SCF but can improve accuracy for higher-order fragmentation (e.g., tetramers) and produce crisper MP2 orbitals. Validate results against the default before adopting more aggressive thresholds.

   .. exess-param:: gradient_screening_threshold
      :type: float
      :default: 1e-10
      :brief: Additional screening for gradient-related integrals.

   .. exess-param:: bf_cutoff_threshold
      :type: float
      :default: density_threshold
      :brief: Basis-function cutoff threshold (defaults to ``density_threshold`` if omitted).

   .. exess-param:: density_basis_set_projection_fallback_enabled
      :type: bool
      :default: auto (fragmented)
      :brief: STO-3G projection fallback toggle.

      If omitted, EXESS enables fallback for fragmented calculations and disables it for full-system calculations.

      When triggered, EXESS reruns SCF in STO-3G and projects the density into the target basis.

   .. exess-param:: allow_crap_scf
      :type: bool
      :default: false
      :brief: Expert flag to allow lower-quality SCF.
      :note: expert

      Expert control; adjust only with validation and verify accuracy before production use.

   .. exess-param:: store_ri_b_on_host
      :type: bool
      :default: false
      :brief: Store RI B matrix on host memory.

      Use this if GPU memory is insufficient for RI; this is slower but can still outperform non-RI for some systems.

   .. exess-param:: compress_ri_b
      :type: bool
      :default: false
      :brief: Compress RI B matrix.
      :note: info

      Compression can reduce GPU memory use enough to run larger systems that would otherwise exceed available RAM.

   .. exess-param:: homo_lumo_guess_rotation_angle
      :type: float
      :default: auto (0 or 45)
      :brief: HOMO/LUMO guess rotation (degrees).

      Rotation in degrees (0-180) for unrestricted symmetry breaking.

      If omitted, EXESS uses 45 degrees for unrestricted singlets and 0 otherwise.

   .. exess-param:: fock_build_type
      :type: string
      :default: HGP
      :brief: Fock build algorithm (``HGP``, ``UM09``, ``RI``).
      :note: info

      Algorithm definitions:

      ``HGP``
        Head-Gordon-Pople algorithm, optimized for dense systems.

      ``UM09``
        Ufimtsev-Martinez algorithm, optimized for screening-heavy systems.

      ``RI``
        Resolution-of-identity approximation (requires auxiliary basis, higher memory use).

      Guidance: ``HGP`` is tuned for dense systems where screening is less important (e.g., compact biomolecules). ``UM09`` is tuned for screening-heavy systems (e.g., long chains) and can scale better on large systems. ``RI`` stores integrals, can be faster on small systems, but memory usage rises substantially.

      ``fock_build_type`` includes improved screening for large systems (>3000 basis functions); see https://arxiv.org/abs/2407.21445 for details.

   .. exess-param:: exchange_screening_threshold
      :type: float
      :default: 1e-5
      :brief: Exchange screening threshold (expert control).
      :note: expert

      Expert control; adjust only with validation.

   .. exess-param:: group_shared_exponents
      :type: bool
      :default: false
      :brief: Group shared basis exponents (UM09 only).
      :note: expert

      Expert control used with UM09 and shared-exponent basis sets (e.g., cc-pVDZ).
```


(frag)=
### frag

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "frag": {
               "cutoff_type": "Centroid",
               "distance_metric": "Average",
               "level": "Tetramer",
               "cutoffs": {
                 "dimer": 1000,
                 "trimer": 20,
                 "tetramer": 15
               },
               "included_fragments": [0, 1, 2, 3, 4]
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from rush.exess import FragKeywords, energy

         energy(
             topology_path="molecule_t.json",
             frag_keywords=FragKeywords(
                 cutoff_type="Centroid",
                 distance_metric="Average",
                 level="Tetramer",
                 dimer_cutoff=1000,
                 trimer_cutoff=20,
                 tetramer_cutoff=15,
                 included_fragments=[0, 1, 2, 3, 4],
             ),
         )
```

```{eval-rst}
.. exess-params::

   .. exess-param:: level
      :type: string
      :default: required
      :brief: Fragment expansion order (``Monomer`` .. ``Octamer``).
      :note: info

      Truncation counts scale combinatorially: dimers :math:`n(n-1)/2`,
      trimers :math:`n(n-1)(n-2)/6`, tetramers
      :math:`n(n-1)(n-2)(n-3)/24`.

   .. exess-param:: cutoffs
      :type: object
      :default: unset
      :brief: Distance cutoffs in Angstroms.
      :note: info

      Keys can include ``dimer``, ``trimer``, ``tetramer``, ``pentamer``,
      ``hexamer``, ``heptamer``, ``octamer``.

      Distances are in Angstroms and should follow ``dimer > trimer >
      tetramer`` when using higher orders.

      If omitted, the calculation proceeds without distance filtering (all
      :math:`n`-mers up to ``level``); be cautious with fragment counts to avoid
      excessive compute.

   .. exess-param:: cutoff_type
      :type: string
      :default: ClosestPair
      :brief: Distance definition (``Centroid`` or ``ClosestPair``).

      ``Centroid`` compares fragment centroids.

      ``ClosestPair`` uses the minimal inter-fragment atom distance (more
      accurate and generally preferred).

   .. exess-param:: distance_metric
      :type: string
      :default: Max
      :brief: Reduce pair distances (``Max``, ``Average``, ``Min``).

      Controls how higher-order distances are computed from pair distances.

   .. exess-param:: reference_fragment
      :type: int
      :default: unset
      :brief: Reference fragment for interaction energies.
      :note: info

      Enables lattice/interaction energies by summing :math:`n`-mer corrections that
      include the reference fragment. Negative values indicate binding; positive
      values indicate repulsion under the usual convention.

   .. exess-param:: included_fragments
      :type: array[int]
      :default: unset
      :brief: Subset of fragments to include.

      Restricts the fragment set and treats them as an independent system.

   .. exess-param:: enable_speed
      :type: bool
      :default: false
      :brief: Experimental queue optimization.
      :note: experimental

      Experimental queue optimization intended for AIMD workflows; avoid unless you can validate against a baseline.
```


(ks_dft)=
### ks_dft

KSDFT is used when `model.method` is `RestrictedKSDFT`. The KSDFT methodologies are described in the following paper:

Stocks, R.; Barca, G. M. J. Efficient Algorithms for GPU Accelerated Evaluation of the DFT Exchange-Correlation Functional. J. Chem. Theory Comput. 2025. [https://doi.org/10.1021/acs.jctc.5c01229](https://doi.org/10.1021/acs.jctc.5c01229).

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedKSDFT",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "ks_dft": {
               "functional": "GGA_XC_PBE",
               "grid": {
                 "default_grid": "SUPERFINE",
                 "radial_quad": "TreutlerAldrichs",
                 "pruning_scheme": "TREUTLER"
               }
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from rush.exess import DefaultGridResolution, KSKeywords, XCGridParameters, energy

         energy(
             topology_path="molecule_t.json",
             method="RestrictedKSDFT",
             ks_keywords=KSKeywords(
                 functional="GGA_XC_PBE",
                 grid=XCGridParameters(
                     resolution=DefaultGridResolution("SUPERFINE"),
                     radial_quad="TreutlerAldrichs",
                     pruning_scheme="TREUTLER",
                 ),
             ),
         )
```

```{eval-rst}
.. exess-params::

   .. exess-param:: functional
      :type: string
      :default: required
      :brief: LibXC functional name.
      :note: info

      EXESS accepts LibXC functional names and ExchCXX linear combinations. Input is
      uppercased internally, so names are case-insensitive.

      Built-in aliases include ``SVWN5``, ``B2PLYP``, ``REVDSD-PBEP86-D4``, and
      ``REVDSD-PBEP86-D4(NOFC)``.

      For a full list of functionals, see the LibXC documentation:
      https://libxc.gitlab.io/functionals

   .. exess-param:: method
      :type: string
      :default: GauXC
      :brief: XC evaluation method.
      :note: info

      ``GauXC`` (default)
        GauXC-backed XC evaluation.

      ``Dense``
        Dense matrix evaluation.

      ``BatchDense``
        Batched dense evaluation (recommended for most production runs).

      ``Direct``
        Direct evaluation without storing intermediates.

      ``SemiDirect``
        Hybrid of direct and batch-dense methods.

   .. exess-param:: use_C_opt
      :type: bool
      :default: true
      :brief: Use C-matrix optimization (Dense/BatchDense).

      ``use_C_opt`` enables C-matrix based XC evaluation, reducing matrix dimensions from ``n_basis`` to ``n_occ`` for Dense/BatchDense methods. It is only valid for ``Dense`` and ``BatchDense``.

   .. exess-param:: grid
      :type: object
      :default: default grid (ULTRAFINE)
      :brief: Numerical grid settings.
      :note: info

      .. rubric:: Grid quality parameters

      Defaults shown where applicable.

      ``radial_quad`` (default: ``MuraKnowles``)
        ``MuraKnowles``, ``MurrayHandyLaming``, ``TreutlerAldrichs``.

      ``pruning_scheme`` (default: ``ROBUST``)
        ``ROBUST``, ``UNPRUNED``, ``TREUTLER``.

      ``consider_weight_zero`` (default: auto)
        Defaults to :math:`10^{-5}` times ``sp_threshold`` if set, otherwise ``dp_threshold``,
        otherwise the SCF ``density_threshold``.

      .. rubric:: Grid size options

      Choose one:

      ``default_grid`` (default: ``ULTRAFINE``)
        Preset grid: ``FINE``, ``ULTRAFINE``, ``SUPERFINE``, ``TREUTLER_GM3``, ``TREUTLER_GM5``.

      ``radial_size``, ``angular_size``
        Custom grid sizes (use together). When set, ``default_grid`` is ignored.

      .. rubric:: Batching options

      Choose one:

      Closest-atom batching
        Default when no batch settings are provided (non-``GauXC`` methods).

      ``octree``
        Uses ``max_size`` (default ``512``), ``max_depth`` (default unlimited),
        ``max_distance`` (default unlimited), ``combine_small_children`` (default ``true``).

      ``space_filling``
        Uses the ``octree`` parameters plus ``target_batch_size`` (default ``1024``).
        ``combine_small_children`` defaults to ``false`` for space-filling.

      ``batch_size``
        GauXC batch size (default ``512``).

      If multiple batching keys are provided, EXESS prioritizes ``octree``, then ``batch_size``,
      then ``space_filling``. For ``GauXC``, EXESS forces GauXC batching and ignores other schemes.

      .. rubric:: Grid guidance

      - Default grid settings (ULTRAFINE with ROBUST pruning) provide a good accuracy/cost balance for most users.
      - SUPERFINE grids can improve accuracy but significantly increase compute time.
      - Octree batching with BatchDense is useful for large systems where linear scaling is critical.

   .. exess-param:: sp_threshold
      :type: float
      :default: SCF density_threshold
      :brief: Single-precision threshold.

      Defaults to ``dp_threshold`` when set, otherwise the SCF ``density_threshold``.

   .. exess-param:: dp_threshold
      :type: float
      :default: SCF density_threshold
      :brief: Double-precision threshold.

   .. exess-param:: batches_per_batch
      :type: int
      :default: 20
      :brief: Batches per batch for ``BatchDense``.

      Only used when ``method = "BatchDense"``.
```

Minimal KSDFT:

```json
"ks_dft": {
  "functional": "GGA_XC_PBE"
}
```

Custom grid defaults:

```json
"ks_dft": {
  "functional": "HYB_GGA_XC_B3LYP",
  "method": "Dense",
  "grid": {
    "default_grid": "ULTRAFINE",
    "radial_quad": "MuraKnowles",
    "pruning_scheme": "ROBUST"
  }
}
```

Octree batching:

```json
"ks_dft": {
  "functional": "HYB_GGA_XC_B3LYP",
  "method": "BatchDense",
  "grid": {
    "octree": {
      "max_size": 512
    }
  }
}
```


(export)=
### export

Export controls what is written to HDF5 output files:

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "export": {
               "export_density": true,
               "export_fock": true,
               "descriptor_grid": {
                 "regular": {
                   "min": [-4.0, -4.0, -4.0],
                   "max": [4.0, 4.0, 4.0],
                   "spacing": [0.2, 0.2, 0.2]
                 }
               }
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from rush.exess import ExportKeywords, RegularDescriptorGrid, energy

         energy(
             topology_path="molecule_t.json",
             export_keywords=ExportKeywords(
                 export_density=True,
                 export_fock=True,
                 descriptor_grid=RegularDescriptorGrid(
                     min=[-4.0, -4.0, -4.0],
                     max=[4.0, 4.0, 4.0],
                     spacing=[0.2, 0.2, 0.2],
                 ),
             ),
         )
```

```{eval-rst}
.. exess-params::

   .. exess-param:: export_density
      :type: bool
      :default: false
      :brief: Export density.
      :note: info

   .. exess-param:: export_relaxed_mp2_density_correction
      :type: bool
      :default: false
      :brief: Export relaxed MP2 density correction.

   .. exess-param:: export_fock
      :type: bool
      :default: false
      :brief: Export Fock matrix.

   .. exess-param:: export_overlap
      :type: bool
      :default: false
      :brief: Export overlap matrix.

   .. exess-param:: export_h_core
      :type: bool
      :default: false
      :brief: Export H core matrix.

   .. exess-param:: export_expanded_density
      :type: bool
      :default: false
      :brief: Export expanded density.
      :note: info

      Provides the whole density matrix for the entire fragment system, rather than per-fragment matrices.

   .. exess-param:: export_expanded_gradient
      :type: bool
      :default: false
      :brief: Export expanded gradient.

      Provides the whole gradient matrix for the entire fragment system, rather than per-fragment matrices.

      Requires a gradient-capable driver (Gradient, Dynamics, QMMM, Optimization).

   .. exess-param:: export_molecular_orbital_coeffs
      :type: bool
      :default: false
      :brief: Export MO coefficients.

   .. exess-param:: export_gradient
      :type: bool
      :default: false
      :brief: Export gradients.

      Requires a gradient-capable driver (Gradient, Dynamics, QMMM, Optimization).

   .. exess-param:: export_external_charge_gradient
      :type: bool
      :default: false
      :brief: Export external charge gradients.

   .. exess-param:: export_mulliken_charges
      :type: bool
      :default: false
      :brief: Export Mulliken charges.
      :note: info

   .. exess-param:: export_chelpg_charges
      :type: bool
      :default: false
      :brief: Export CHELPG charges.
      :note: info

   .. exess-param:: export_bond_orders
      :type: bool
      :default: false
      :brief: Export bond orders.

   .. exess-param:: export_h_caps
      :type: bool
      :default: false
      :brief: Export H caps.

   .. exess-param:: export_density_descriptors
      :type: bool
      :default: false
      :brief: Export density descriptors.
      :note: info

   .. exess-param:: export_esp_descriptors
      :type: bool
      :default: false
      :brief: Export ESP descriptors.
      :note: info

   .. exess-param:: export_expanded_esp_descriptors
      :type: bool
      :default: false
      :brief: Export expanded ESP descriptors.

   .. exess-param:: export_basis_labels
      :type: bool
      :default: false
      :brief: Export basis labels.

   .. exess-param:: export_hessian
      :type: bool
      :default: false
      :brief: Export hessian.

      Requires a Hessian calculation.

   .. exess-param:: export_mass_weighted_hessian
      :type: bool
      :default: false
      :brief: Export mass-weighted hessian.

      Requires a Hessian calculation.

   .. exess-param:: export_hessian_frequencies
      :type: bool
      :default: false
      :brief: Export hessian frequencies.

      Requires a Hessian calculation.

   .. exess-param:: flatten_symmetric
      :type: bool
      :default: true
      :brief: Flatten symmetric matrices.

   .. exess-param:: light_json
      :type: bool
      :default: false
      :brief: Light JSON output.

   .. exess-param:: concatenate_hdf5_files
      :type: bool
      :default: false
      :brief: Concatenate HDF5 outputs.

      Post-process exports into a single HDF5 output file. This is primarily relevant for fragmented runs (particularly when configured for multinode). The concatenation may be expensive.

   .. exess-param:: training_db
      :type: bool
      :default: false
      :brief: Export training DB metadata.

   .. exess-param:: descriptor_grid
      :type: object
      :default: unset
      :brief: Grid for descriptor exports.
      :note: info

      ``descriptor_grid`` can be one of the following structures:

      ``standard``
        ``FINE``, ``ULTRAFINE``, ``SUPERFINE``, ``TREUTLER_GM3``, ``TREUTLER_GM5``.

      ``params``
        ``points_per_shell``, ``order`` (``One`` or ``Two``), ``scale``.

      ``regular``
        ``min``, ``max``, ``spacing`` arrays (Cartesian grid).

      ``custom``
        flat list of points ``[x1, y1, z1, x2, y2, z2, ...]``.
```


## Q4ML: Optimization & Simulation

(regions)=
### regions

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "QMMM",
           "model": {
             "method": "RestrictedHF",
             "basis": "STO-3G"
           },
           "keywords": {
             "qmmm": {
               "n_timesteps": 10,
               "dt_ps": 0.002,
               "temperature_kelvin": 290.0
             },
             "regions": {
               "qm_fragments": [0, 1],
               "mm_fragments": [2, 3]
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from rush.exess import qmmm

         qmmm(
             topology_path="molecule_t.json",
             residues_path="system.residues",
             n_timesteps=10,
             dt_ps=0.002,
             temperature_kelvin=290.0,
             qm_fragments=[0, 1],
             mm_fragments=[2, 3],
         )
```

Rules and defaults:

- Provide at least two of the three lists; the remaining region is inferred as the fragments not mentioned elsewhere.
- If all three lists are provided, they must be disjoint and cover all fragments.
- Supplying only one list is invalid.
- If `regions` is omitted in JSON, `mm_fragments` and `ml_fragments` default to empty and `qm_fragments` is inferred as all fragments (pure QM).
- If any non-QM region exists, residues must be provided; with no residues, the entire system must be QM.
- Non-QM regions are only supported for `QMMM` and `Optimization`; other drivers (including `Energy`) require pure QM regions.
- Non-QM regions are not supported for batched topology inputs.

```{eval-rst}
.. exess-params::

   .. exess-param:: qm_fragments
      :type: array[int]
      :default: inferred
      :brief: Fragments treated as QM.

   .. exess-param:: mm_fragments
      :type: array[int]
      :default: inferred
      :brief: Fragments treated as MM.

   .. exess-param:: ml_fragments
      :type: array[int]
      :default: inferred
      :brief: Fragments treated as ML.
```


(optimization)=
### optimization

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Optimization",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "optimization": {
               "max_iters": 200,
               "algorithm": "LBFGS",
               "coordinate_system": "Cartesian",
               "lbfgs_keywords": {}
             },
             "regions": {
               "qm_fragments": [0],
               "ml_fragments": [1, 2, 3]
             },
             "machine_learning": {
               "ml_type": "AIMNet"
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from rush.exess import LBFGSKeywords, OptimizationKeywords, optimization

         optimization(
             topology_path="molecule_t.json",
             max_iters=200,
             optimization_keywords=OptimizationKeywords(
                 algorithm="LBFGS",
                 coordinate_system="Cartesian",
                 lbfgs_keywords=LBFGSKeywords(),
             ),
             qm_fragments=[0],
             ml_fragments=[1, 2, 3],
         )
```

Fragmentation (``frag``) can be used when a QM region exists; EXESS fragments only the QM region and leaves MM/ML regions intact. Residue requirements and region validation details are covered in the [regions](#regions) section.

```{eval-rst}
.. exess-params::

   .. exess-param:: max_iters
      :type: int
      :default: required
      :brief: Max optimization iterations.

   .. exess-param:: convergence_criteria
      :type: object
      :default: see details
      :brief: Convergence metric and thresholds.

      Fields:

      - ``metric`` — ``string`` (default: ``Baker``)
      - ``gradient_threshold`` — ``float`` (default: ``3e-4``; units: Eh/a0)
      - ``delta_energy_threshold`` — ``float`` (default: ``1e-6``; units: Eh)
      - ``step_component_threshold`` — ``float`` (default: ``3e-4``; units: a0)

      ``metric`` options:

      ``Baker``
        max gradient component must be within threshold and either delta energy or
        step component must be within their thresholds.

      ``GradientOnly``
        only the gradient threshold is enforced.

   .. exess-param:: optimizer_reset_interval
      :type: int
      :default: unset
      :brief: The coordinate system will be regenerated and the Hessian reset every :math:`N` iterations.
      :note: expert

      If omitted, EXESS never regenerates the coordinate system or resets the Hessian.

   .. exess-param:: coordinate_system
      :type: string
      :default: DelocalisedInternal
      :brief: Coordinate system (``Cartesian``, ``NaturalInternal``, ``DelocalisedInternal``).
      :note: info

      ``DelocalisedInternal`` is the default and strongly recommended.

      Machine learning optimizations require ``Cartesian``.

   .. exess-param:: constraints
      :type: array[array[int]]
      :default: []
      :brief: Constrain bond lengths, angles, or dihedrals.
      :note: info

      Specify lists of atom indices to constrain.

   .. exess-param:: hessian_guess
      :type: string
      :default: auto
      :brief: Initial Hessian model (``Identity``, ``ScaledIdentity``, ``Schlegel``, ``Lindh``).

      Defaults to ``Identity`` for Cartesian coordinates, otherwise
      ``ScaledIdentity``.

      Upstream docs caution that non-default models are not recommended for
      general use.

   .. exess-param:: algorithm
      :type: string
      :default: EigenvectorFollowing
      :brief: Optimization algorithm (``EigenvectorFollowing``, ``TrustRegionAugmentedHessian``, ``LBFGS``).
      :note: info

      ``EigenvectorFollowing`` is recommended for most users.

      ``TrustRegionAugmentedHessian`` is available but not recommended for most
      workflows.

      For machine learning optimizations, ``LBFGS`` is strongly recommended.

   .. exess-param:: lbfgs_keywords
      :type: object
      :default: unset
      :brief: LBFGS parameters.
      :note: info

      Required when ``algorithm`` is ``LBFGS`` (an empty object ``{}`` is
      acceptable).

      Fields (defaults apply when ``LBFGS`` is used; set ``{}`` to use defaults):

      - ``linesearch`` — ``string`` (default: ``BacktrackingStrongWolfe``)
      - ``n_corrections`` — ``int`` (default: ``6``)
      - ``epsilon`` — ``float`` (default: ``1e-5``)
      - ``max_linesearch`` — ``int`` (default: ``40``)
      - ``gtol`` — ``float`` (default: ``0.9``)

   .. exess-param:: trust_region
      :type: object
      :default: see details
      :brief: Trust-region parameters (for ``TrustRegionAugmentedHessian``).

      Fields (defaults apply when ``TrustRegionAugmentedHessian`` is used; set ``{}`` to use defaults):

      - ``initial_radius`` — ``float`` (default: ``0.4``)
      - ``max_radius`` — ``float`` (default: ``1e5``)
      - ``min_radius`` — ``float`` (default: ``1e-5``)
      - ``increase_factor`` — ``float`` (default: ``1.2``)
      - ``decrease_factor`` — ``float`` (default: ``0.7``)
      - ``constrict_factor`` — ``float`` (default: ``0.1``)
      - ``increase_threshold`` — ``float`` (default: ``0.75``)
      - ``decrease_threshold`` — ``float`` (default: ``0.25``)
      - ``rejection_threshold`` — ``float`` (default: ``0.0``)

      The defaults have been optimized; changing them is not recommended unless you have a validated use case.

   .. exess-param:: frozen_distance_slippage_tolerance_angstroms
      :type: float
      :default: 1e-8
      :brief: Slippage tolerance (distance).

      Controls expected slippage in frozen delocalized coordinates.
      These tolerances account for small drift when delocalized coordinates are held fixed.

   .. exess-param:: frozen_angle_slippage_tolerance_degrees
      :type: float
      :default: 1e-8
      :brief: Slippage tolerance (angle).

      Controls expected slippage in frozen delocalized coordinates.
      These tolerances account for small drift when delocalized coordinates are held fixed.

   .. exess-param:: debug_xyz
      :type: bool
      :default: false
      :brief: Debug XYZ output.

   .. exess-param:: output_trc
      :type: string
      :default: unset
      :brief: Output TRC path.

   .. exess-param:: fixed_atoms
      :type: array[int]
      :default: unset
      :brief: Fixed atoms.

   .. exess-param:: free_atoms
      :type: array[int]
      :default: unset
      :brief: Free atoms.

   .. exess-param:: fixed_fragments
      :type: array[int]
      :default: unset
      :brief: Fixed fragments.

   .. exess-param:: free_fragments
      :type: array[int]
      :default: unset
      :brief: Free fragments.

   .. exess-param:: fix_heavy
      :type: bool
      :default: false
      :brief: Fix heavy atoms.
```


(qmmm)=
### qmmm

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "QMMM",
           "model": {
             "method": "RestrictedHF",
             "basis": "STO-3G"
           },
           "keywords": {
             "qmmm": {
               "n_timesteps": 1000,
               "dt_ps": 0.002,
               "temperature_kelvin": 290.0,
               "trajectory": {
                 "format": "XYZ",
                 "interval": 10
               },
               "restraints": {
                 "k": 1500.0,
                 "fix_heavy": true
               }
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from rush.exess import Restraints, Trajectory, qmmm

         qmmm(
             topology_path="molecule_t.json",
             residues_path="molecule_r.json",
             n_timesteps=1000,
             dt_ps=0.002,
             temperature_kelvin=290.0,
             trajectory=Trajectory(interval=10),
             restraints=Restraints(k=1500.0, fix_heavy=True),
         )
```

Fragmentation (``frag``) can be used when a QM region exists; EXESS fragments only the QM region and leaves MM/ML regions intact. Residue requirements and region validation details are covered in the [regions](#regions) section.

```{eval-rst}
.. exess-params::

   .. exess-param:: n_timesteps
      :type: int
      :default: required
      :brief: Number of QMMM timesteps.

   .. exess-param:: dt_ps
      :type: float
      :default: required
      :brief: Timestep size in ps.

   .. exess-param:: temperature_kelvin
      :type: float
      :default: required
      :brief: Temperature in Kelvin.

   .. exess-param:: pressure_atm
      :type: float
      :default: unset
      :brief: Optional pressure for NPT runs.
      :note: info

      If set, EXESS runs NPT; if unset, NVT is used.

   .. exess-param:: biases
      :type: array[object]
      :default: unset
      :brief: Bias potentials for QMMM dynamics.
      :note: info

      Each entry must provide exactly one of:

      ``harmonic_cv``
        Fields: ``k``, ``offset``, ``cv``.

      ``moving_harmonic``
        Fields: ``cv``, ``k``, ``rate_per_ps``, optional ``initial_offset``, optional
        ``final_offset``, and ``resolution`` (default ``0.01``).

      ``avoid_bonds``
        Fields: ``index``, optional ``indices`` (default ``[]``), optional ``fragments``
        (default ``[]``), optional ``exceptions`` (default ``[]``), ``steepness``
        (default ``10.0``), ``height`` (default ``10000.0``).

      ``cv`` objects must provide one of:

      ``bond`` (fields: ``index1``, ``index2``), ``angle`` (fields: ``index1``, ``index2``,
      ``index3``), ``dihedral`` (fields: ``index1``, ``index2``, ``index3``, ``index4``), or
      ``linear_combination`` (field: ``scaled_cvs``, each entry has ``scale`` and ``cv``).

   .. exess-param:: pbc_ang
      :type: array[float]
      :default: unset
      :brief: Periodic box lengths in angstroms.

      Three-element vector ``[a, b, c]`` in angstroms.

   .. exess-param:: minimisation
      :type: object
      :default: unset
      :brief: Classical minimisation settings.

      Fields:

      - ``err_tol_kj_per_mol_nm`` — ``float`` (default: ``10``)
      - ``max_iterations`` — ``int`` (default: ``0``)

      ``minimisation`` can only be used in a purely classical run (no QM/ML regions).

   .. exess-param:: trajectory
      :type: object
      :default: unset
      :brief: Trajectory output settings.
      :note: info

      Fields:

      - ``format`` — ``string`` (default: ``JSON``)
      - ``interval`` — ``int`` (default: ``1``)
      - ``start`` — ``int`` (default: ``0``)
      - ``end`` — ``int`` (default: max u32)
      - ``include_waters`` — ``bool`` (default: ``false``)
      - ``forces`` — ``string`` (default: unset)

      ``trajectory.format`` can be ``JSON`` or ``XYZ`` (default ``JSON``).

      ``trajectory.include_waters`` can be set to omit waters for smaller trajectories.

      ``trajectory.forces`` can be ``all``, ``standard``, or ``biases``.

   .. exess-param:: energy_csv
      :type: string
      :default: unset
      :brief: Path for energy CSV.

      When set, EXESS uses a Verlet integrator and does not apply the thermostat (the temperature is not used for integration).

   .. exess-param:: cv_values_csv
      :type: string
      :default: unset
      :brief: Path for CV values CSV.

      Requires at least one CV-based bias (``harmonic_cv`` or ``moving_harmonic``).

   .. exess-param:: restraints
      :type: object
      :default: unset
      :brief: Restraints for atoms/fragments.
      :note: info

      Fields:

      - ``k`` — ``float`` (default: ``2000.0``)
      - ``fixed_atoms`` — ``array[int]`` (default: unset)
      - ``free_atoms`` — ``array[int]`` (default: unset)
      - ``fixed_fragments`` — ``array[int]`` (default: unset)
      - ``free_fragments`` — ``array[int]`` (default: unset)
      - ``fix_heavy`` — ``bool`` (default: ``false``)

      Only one of ``fixed_atoms``, ``free_atoms``, ``fixed_fragments``, or ``free_fragments`` may be specified. Set ``free_atoms = []`` to fix all atoms.

      ``restraints.k`` scales the restraint force; larger values mean stronger restraints.

   .. exess-param:: ffs
      :type: array[string]
      :default: unset
      :brief: Additional QMMM force field files.
```


(gradient)=
### gradient

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Gradient",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "gradient": {
               "method": "Numerical",
               "finite_difference_step_size": 0.004
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: finite_difference_step_size
      :type: float
      :default: 5e-3
      :brief: Step size for numerical gradients.

   .. exess-param:: method
      :type: string
      :default: Analytical
      :brief: ``Analytical`` or ``Numerical``.
```


## Advanced & Diagnostic Keywords

### guess

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "guess": {
               "external_initial_density_path": "guess.h5",
               "bsp": true,
               "bsp_basis": "STO-3G"
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: external_initial_density_path
      :type: string
      :default: unset
      :brief: HDF5 density guess path.

      Must reference an HDF5 file with a ``density`` dataset at root for RHF, or
      ``alpha/density`` and ``beta/density`` for UHF.

      Guesses are expected to be stored as flattened lower-triangular density
      matrices. External guesses are not supported for fragmented calculations,
      and EXESS warns that guesses from other codes may be incompatible due to
      basis ordering and normalization.

   .. exess-param:: bsp
      :type: bool
      :default: false
      :brief: Basis set projection bootstrap.

      Computes a lower-resolution SCF and projects to the target basis. Requires
      ``bsp_basis``.

   .. exess-param:: bsp_basis
      :type: string
      :default: unset
      :brief: Lower-resolution basis set for BSP.

   .. exess-param:: bsp_scf_keywords
      :type: object
      :default: unset
      :brief: SCF keywords for BSP.

      If omitted, EXESS reuses the base SCF keywords.

   .. exess-param:: hcore
      :type: bool
      :default: false
      :brief: Use hcore initial guess.

   .. exess-param:: smd
      :type: bool
      :default: auto (fragmented non-RI)
      :brief: Superposition of monomer densities.

      If omitted, EXESS enables it for fragmented calculations that are not
      using RI, and disables it otherwise.

   .. exess-param:: ssfd
      :type: bool
      :default: false
      :brief: Subfragment density guess.
      :note: experimental

      Experimental subfragment guess. ``ssfd_target_size`` controls subfragment
      size (default 30).

   .. exess-param:: ssfd_target_size
      :type: int
      :default: 30
      :brief: Target atoms per subfragment.

   .. exess-param:: ssfd_only_converge_in_bsp_basis
      :type: bool
      :default: true
      :brief: Only converge subfragments in BSP basis.

      Keeps subfragments unconverged in the primary basis and only projects from
      the bootstrap basis.

   .. exess-param:: ssfd_scf_keywords
      :type: object
      :default: unset
      :brief: SCF keywords for subfragment runs.

      If omitted, EXESS reuses the base SCF keywords.
```


### integrals

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "integrals": {
               "scheduler": "RoundRobin",
               "n_streams": 8
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: scheduler
      :type: string
      :default: Callback
      :brief: Integral scheduler (``Callback`` or ``RoundRobin``).

      If ``integrals`` is omitted entirely, EXESS uses ``Callback`` with 4 streams.

   .. exess-param:: n_streams
      :type: int
      :default: 4 (CUDA) / 1 (HIP)
      :brief: GPU stream count.
```


### rtat

RTAT is a runtime auto-tuner for matrix operations.

RTAT is the open-source [rtatblas](https://github.com/csnowdon2/rtatblas) library. When enabled, EXESS uses it to auto-tune GPU BLAS configurations for matrix operations.

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "rtat": {
               "synchronous": true,
               "json_file_dump_prefix": "rtat"
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: enabled
      :type: bool
      :default: true
      :brief: Enable runtime autotuning.

   .. exess-param:: synchronous
      :type: bool
      :default: false
      :brief: Use synchronous operations.

   .. exess-param:: json_file_dump_prefix
      :type: string
      :default: unset
      :brief: Prefix for RTAT JSON dumps.
```


(hessian)=
### hessian

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Hessian",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "hessian": {
               "finite_difference_step_size": 0.004
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: finite_difference_step_size
      :type: float
      :default: 5e-3
      :brief: Step size for numerical Hessians.

   .. exess-param:: method
      :type: string
      :default: Numerical
      :brief: ``Analytical`` or ``Numerical``.
```


(dynamics)=
### dynamics

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Dynamics",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "dynamics": {
               "n_timesteps": 10,
               "use_async_timesteps": false,
               "dt": 0.002
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: n_timesteps
      :type: int
      :default: required
      :brief: Number of timesteps.

   .. exess-param:: dt
      :type: float
      :default: required
      :brief: Timestep size in ps.

      A typical value is 1 fs (``0.001`` ps); you must set ``dt`` explicitly.

   .. exess-param:: reuse_orbitals
      :type: bool
      :default: false
      :brief: Reuse orbitals between timesteps.

   .. exess-param:: use_async_timesteps
      :type: bool
      :default: true
      :brief: Run asynchronous timesteps.
      :note: expert

      Expert option; validate stability before production runs.
```


### boundary

Boundary conditions for periodic or truncated simulations:

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Dynamics",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "boundary": {
               "x": { "kind": "Periodic", "range": { "lower": -2, "upper": 3 } },
               "y": { "kind": "Periodic", "range": { "lower": -2, "upper": 3 } },
               "z": { "kind": "Periodic", "range": { "lower": -2, "upper": 3 } }
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: x
      :type: object
      :default: unset
      :brief: Boundary configuration for the X axis.

      Each axis entry has:

      ``kind``
        ``Periodic``, ``Rigid``, or ``Delete``.

      ``range``
        ``lower``/``upper`` extents for periodic boundaries.

   .. exess-param:: y
      :type: object
      :default: unset
      :brief: Boundary configuration for the Y axis.

      Same structure as ``x``.

   .. exess-param:: z
      :type: object
      :default: unset
      :brief: Boundary configuration for the Z axis.

      Same structure as ``x``.
```


### machine_learning
```{eval-rst}
.. exess-params::

   .. exess-param:: ml_type
      :type: string
      :default: AIMNet
      :brief: ML model type.

      AIMNet is currently the only supported value.
```


(force_field)=
### force_field

`force_field` supplies a classical force field for water/MM components in fragmented AIMD (Dynamics) and other classical MBE steps. QMMM uses `qmmm.ffs` for additional force fields instead.

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Dynamics",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ",
             "aux_basis": "cc-pVDZ-RIFIT"
           },
           "keywords": {
             "scf": {
               "fock_build_type": "RI"
             },
             "dynamics": {
               "n_timesteps": 10,
               "dt": 0.001
             },
             "frag": {
               "level": "Dimer",
               "cutoffs": {
                 "dimer": 7
               }
             },
             "boundary": {
               "x": { "kind": "Periodic", "range": { "lower": -3, "upper": 4 } },
               "y": { "kind": "Periodic", "range": { "lower": -3, "upper": 6 } },
               "z": { "kind": "Periodic", "range": { "lower": -3, "upper": 6 } }
             },
             "force_field": {
               "ff_filename": "forcefield.xml"
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: ff_filename
      :type: string
      :default: required
      :brief: Force field filename path.

      Used for classical water/MM contributions in fragmented AIMD/MBE; QMMM uses `qmmm.ffs` instead.
```


### log

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "log": {
               "console": { "level": "Verbose" },
               "logfiles": [
                 {
                   "level": "Info",
                   "prefix_fmt": "[%H:%M:%S {level}] ",
                   "directory": "/tmp/exess"
                 }
               ]
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: console
      :type: object
      :default: see details
      :brief: Console log settings.

      Fields:

      - ``level`` — ``string`` (default: ``LargeInfo``). Debug builds use ``Debug``.
      - ``prefix_fmt`` — ``string`` (default: ``""``).

      Log levels: ``Debug``, ``Verbose``, ``LargeInfo``, ``Info``, ``Performance``,
      ``Warning``, ``Error`` (descending verbosity).

   .. exess-param:: logfiles
      :type: array[object]
      :default: []
      :brief: File log settings.

      .. role:: raw-html(raw)
         :format: html

      Fields:

      - ``level`` — ``string`` (default: ``Verbose``)
      - ``prefix_fmt`` — ``string`` (default: :raw-html:`<code>[%Y-%m-%d %H:%M:%S.{us} r{rank} {level}]&nbsp;</code>`)
      - ``directory`` — ``string`` (default: unset)
```


### debug

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedHF",
             "basis": "cc-pVDZ"
           },
           "keywords": {
             "debug": {
               "dry_run": true
             }
           },
           "schema_version": "0.2.0"
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: dry_run
      :type: bool
      :default: false
      :brief: Validate fragment queue without computing.

      Runs queue construction only (no computation) to validate fragment counts and detect input issues.

   .. exess-param:: print_subfragment_xyz
      :type: bool
      :default: false
      :brief: Print subfragment XYZ for SSFD.

      Prints subfragment geometries for SSFD debugging.

   .. exess-param:: max_fragments
      :type: int
      :default: -1
      :brief: Limit number of fragments computed.

      The default ``-1`` means "use all fragments."

   .. exess-param:: ignore_fragments
      :type: bool
      :default: false
      :brief: Ignore fragmentation (developer validation).

      Forces a full-system calculation for validation.

   .. exess-param:: skip_calcs
      :type: bool
      :default: false
      :brief: Skip computations in fragmentation routines.

      Skips calculations during fragmentation to debug queue construction performance.
```

## Rush-py defaults

Rush-py sets some defaults in Python before submitting a run. If a `*_keywords` argument is omitted, rush-py may pass `None` (no overrides) or construct a default object.

Default keyword behavior for the common entry points:

- `exess.exess` / `exess.energy` / `exess.interaction_energy`:
  - `scf_keywords`: unset (EXESS defaults apply).
  - `frag_keywords`: `FragKeywords()` (level `Dimer`, `dimer_cutoff=100.0`, `trimer_cutoff=None`, `tetramer_cutoff=None`, `cutoff_type=None`, `distance_metric=None`).
  - `export_keywords`: `ExportKeywords()` (all fields unset; no exports requested).
- `exess.chelpg`:
  - `scf_keywords`: `SCFKeywords(max_diis_history_length=12, convergence_threshold=1e-8)`.
  - `frag_keywords`: `FragKeywords(level="Monomer")`.
  - `export`: CHELPG charges and bond orders enabled.
- `exess.qmmm`:
  - `scf_keywords`: unset (EXESS defaults apply).
  - `frag_keywords`: `FragKeywords()` (same defaults as above).
  - `trajectory`: `Trajectory()` (all fields unset; EXESS defaults apply).
- `exess.optimization`:
  - `optimization_keywords`: `OptimizationKeywords()` (all fields unset; EXESS defaults apply), with required `max_iters` passed separately.

Rush-py entrypoint defaults (non-keyword parameters):

- `exess.exess` / `exess.energy` / `exess.interaction_energy`:
  - `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`.
- `exess.chelpg`:
  - `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`.
  - Overrides `standard_orientation="None"` and `force_cartesian_basis_sets=false`.
- `exess.qmmm`:
  - `method="RestrictedHF"`, `basis="STO-3G"`, `aux_basis=None`.
  - `dt_ps=0.002`, `temperature_kelvin=290.0`, `pressure_atm=None`.
- `exess.optimization`:
  - `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`.

`FragKeywords` defaults by level in rush-py:

`Monomer`
: `dimer_cutoff=100.0`, `trimer_cutoff=None`, `tetramer_cutoff=None`,
  `cutoff_type=None`, `distance_metric=None`.

`Dimer`
: `dimer_cutoff=100.0`, `trimer_cutoff=None`, `tetramer_cutoff=None`.

`Trimer`
: `dimer_cutoff=100.0`, `trimer_cutoff=25.0`, `tetramer_cutoff=None`.

`Tetramer`
: `dimer_cutoff=100.0`, `trimer_cutoff=25.0`, `tetramer_cutoff=10.0`.
