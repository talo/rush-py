# Input format

EXESS inputs are JSON files loosely based on MolSSI QCSchema. In EXESS, the molecular group is called `topology`, and input files use a `topologies` array to allow batched runs (multiple systems evaluated with the same driver/model/keywords).

This page is organized by the top-level input groups (`topologies`, `model`, `system`, `keywords`, `driver`) and follows QC-JSON conventions where possible.

## Schema overview

Top-level EXESS input fields:

```{eval-rst}
.. exess-params::

   .. exess-param:: topologies
      :type: array[Topology]
      :default: required
      :brief: One or more molecular systems.

   .. exess-param:: residues
      :type: array[Residues]
      :default: unset
      :brief: TRC residue definitions aligned to ``topologies``.

   .. exess-param:: external_charges
      :type: object
      :default: unset
      :brief: External point charges (positions + charges).

   .. exess-param:: driver
      :type: string
      :default: required
      :brief: Calculation type (Energy, Gradient, Dynamics, Optimization, Hessian, QMMM).

   .. exess-param:: model
      :type: object
      :default: required
      :brief: Level of theory (method) + basis configuration.

   .. exess-param:: system
      :type: object
      :default: unset
      :brief: Hardware configuration.

   .. exess-param:: keywords
      :type: object
      :default: required
      :brief: Calculation parameters; may be ``{}``.

   .. exess-param:: schema_version
      :type: string
      :default: 0.2.0
      :brief: Input schema version.

   .. exess-param:: title
      :type: string
      :default: unset
      :brief: Printed in output files.

   .. exess-param:: check_schema
      :type: bool
      :default: true (when enabled)
      :brief: Enable schema validation when supported.
```

In the EXESS CLI JSON, `keywords` is required, but can be an empty object because defaults are applied by the parser. `check_schema` is only honored when schema checks are enabled; otherwise it is ignored.

Icon key:

```{eval-rst}
.. raw:: html

   <div class="exess-icon-key">
     <span class="exess-icon-key__item"><span class="param-note param-note--info" aria-hidden="true"></span> Tip</span>
     <span class="exess-icon-key__item"><span class="param-note param-note--expert" aria-hidden="true"></span> Expert</span>
   </div>
```

## Input System Specification

### topologies

``topologies`` is an array of ``topology`` objects. Each ``topology`` provides molecular
data and optional fragmentation/connectivity. See {ref}`topologies` for the full
reference and validation rules.

Example with inline geometry and symbols:

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [
             {
               "symbols": ["O", "H", "H"],
               "geometry": [
                 -1.1570, 0.0630, -0.4817,
                 -1.1570, 0.8180, -1.0766,
                 -1.1570, -0.6920, -1.0766
               ]
             }
           ],
           "driver": "Energy",
           "model": { "method": "RestrictedHF", "basis": "cc-pVDZ" },
           "keywords": {}
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from pathlib import Path

         from rush.exess import energy
         from rush.mol import Element, Topology

         topology = Topology(
             symbols=[Element.O, Element.H, Element.H],
             geometry=[
                 -1.1570, 0.0630, -0.4817,
                 -1.1570, 0.8180, -1.0766,
                 -1.1570, -0.6920, -1.0766,
             ],
         )

         Path("molecule_t.json").write_text(topology.to_json())

         energy(topology_path="molecule_t.json")
```

Example using an XYZ file:

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [
             { "xyz": "molecule.xyz" }
           ],
           "driver": "Energy",
           "model": { "method": "RestrictedHF", "basis": "cc-pVDZ" },
           "keywords": {}
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from pathlib import Path

         from rush.exess import energy
         from rush.mol import Element, Topology

         lines = Path("molecule.xyz").read_text().splitlines()
         symbols = []
         geometry = []
         for line in lines[2:]:
             if not line.strip():
                 continue
             symbol, x, y, z = line.split()[:4]
             symbols.append(Element.from_str(symbol))
             geometry.extend([float(x), float(y), float(z)])

         topology = Topology(symbols=symbols, geometry=geometry)
         Path("molecule_t.json").write_text(topology.to_json())

         energy(topology_path="molecule_t.json")
```

### residues

``residues`` is typically aligned one-to-one with ``topologies`` (same index). It is
required for QMMM and other workflows that rely on residues. See {ref}`residues` for the
full reference.

Example:

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [
             {
               "xyz": "molecule.xyz",
               "fragments": [[0, 1, 2], [3, 4, 5]]
             }
           ],
           "residues": [
             {
               "residues": [[0, 1, 2], [3, 4, 5]],
               "seqs": ["HOH", "HOH"],
               "seq_ns": [1, 2],
               "insertion_codes": ["", ""]
             }
           ],
           "driver": "QMMM",
           "model": { "method": "RestrictedHF", "basis": "STO-3G" },
           "keywords": {
             "regions": {
               "qm_fragments": [0],
               "mm_fragments": [1]
             },
             "qmmm": {
               "n_timesteps": 100
             }
           }
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from pathlib import Path

         from rush.exess import qmmm
         from rush.mol import Element, Fragment, Residue, Residues, Topology

         topology = Topology(
             symbols=[Element.O, Element.H, Element.H, Element.O, Element.H, Element.H],
             geometry=[
                 0.0000, 0.0000, 0.0000,
                 0.7570, 0.5860, 0.0000,
                 -0.7570, 0.5860, 0.0000,
                 2.8000, 0.0000, 0.0000,
                 3.5570, 0.5860, 0.0000,
                 2.0430, 0.5860, 0.0000,
             ],
             fragments=[Fragment([0, 1, 2]), Fragment([3, 4, 5])],
         )

         residues = Residues(
             residues=[Residue([0, 1, 2]), Residue([3, 4, 5])],
             seqs=["HOH", "HOH"],
             seq_ns=[1, 2],
             insertion_codes=["", ""],
         )

         Path("molecule_t.json").write_text(topology.to_json())
         Path("molecule_r.json").write_text(residues.to_json())

         qmmm(
             topology_path="molecule_t.json",
             residues_path="molecule_r.json",
             n_timesteps=100,
             qm_fragments=[0],
             mm_fragments=[1],
         )
```

### external_charges

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "external_charges": {
             "positions": [0.0, 0.0, 0.0, 1.5, 0.0, 0.0],
             "charges": [0.5, -0.5]
           },
           "driver": "Energy",
           "model": { "method": "RestrictedHF", "basis": "cc-pVDZ" },
           "keywords": {}
         }

   .. tab-item:: Python

      Not supported.
```

```{eval-rst}
.. exess-params::

   .. exess-param:: positions
      :type: array[float]
      :default: required
      :brief: Flat XYZ list of charge positions.

      The positions length must be :math:`3 \times \mathrm{len}(charges)`.

   .. exess-param:: charges
      :type: array[float]
      :default: required
      :brief: Charge values aligned to ``positions``.
```

## Configuration

### driver

The `driver` field selects the calculation type:

`Energy`
: Single-point energy and related properties.

`Optimization`
: Geometry optimization. See {ref}`optimization`, plus {ref}`regions` when using Q4ML regions.

`QMMM`
: QM/MM or QM/ML/MM workflows. See {ref}`qmmm`, plus {ref}`regions` as needed.

`Gradient`
: Analytic or finite-difference gradients. See {ref}`gradient` for gradient options.

`Hessian`
: Hessian and vibrational analysis. See {ref}`hessian`, plus SCF/DFT keywords as needed.

`Dynamics`
: Molecular dynamics. See {ref}`dynamics` and {ref}`force_field`.

### model

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": {
             "method": "RestrictedRIMP2",
             "basis": "cc-pVDZ",
             "aux_basis": "cc-pVDZ-RIFIT",
             "standard_orientation": "None",
             "force_cartesian_basis_sets": false
           },
           "keywords": {}
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from rush.exess import energy

         energy(
             topology_path="molecule_t.json",
             method="RestrictedRIMP2",
             basis="cc-pVDZ",
             aux_basis="cc-pVDZ-RIFIT",
             standard_orientation="None",  # String value, not None.
             force_cartesian_basis_sets=False,
         )
```

```{eval-rst}
.. exess-params::

   .. exess-param:: method
      :type: string
      :default: required
      :brief: Level of theory (RestrictedHF, UnrestrictedHF, RestrictedKSDFT, RestrictedRIMP2, UnrestrictedRIMP2, RestrictedRICCSD).

      Level-of-theory descriptions:

      ``RestrictedHF``
        Restricted Hartree-Fock (closed-shell).

      ``UnrestrictedHF``
        Unrestricted Hartree-Fock (open-shell).

      ``RestrictedKSDFT``
        Restricted Kohn-Sham density functional theory.

      ``RestrictedRIMP2``
        Restricted RI Moller-Plesset second-order perturbation theory.

      ``UnrestrictedRIMP2``
        Unrestricted RI Moller-Plesset second-order perturbation theory.

      ``RestrictedRICCSD``
        Restricted RI coupled cluster singles and doubles.

      HF methods support multiple integral build types, while MP2 is only implemented
      via RI (resolution-of-identity).

      When running through rush-py, ``method`` defaults to ``RestrictedHF`` if omitted.

   .. exess-param:: basis
      :type: string
      :default: required
      :brief: Primary basis set. See :ref:`basis sets <basis_sets>`.

      When running through rush-py, ``basis`` defaults to ``cc-pVDZ`` if omitted.

   .. exess-param:: aux_basis
      :type: string
      :default: unset
      :brief: Auxiliary basis for RI methods. See :ref:`basis sets <basis_sets>`.

   .. exess-param:: standard_orientation
      :type: string
      :default: FullSystem
      :brief: Orientation selection (``FullSystem``, ``None``, ``PerFragment``).
      :note: info

      ``PerFragment`` rotates each fragment independently; this can cause inconsistent
      energies and energy differences in fragmented runs.

      ``None`` prevents translation and rotation, which can make it easier to compare
      an optimization trajectory to the input conformation.

   .. exess-param:: force_cartesian_basis_sets
      :type: bool
      :default: true
      :brief: Force Cartesian basis functions.

      For d orbitals this yields components like :math:`x^2, xy, xz, y^2, yz, z^2`.
```

### system

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: json
         :caption: config.json

         {
           "topologies": [{ "xyz": "molecule.xyz" }],
           "driver": "Energy",
           "model": { "method": "RestrictedHF", "basis": "cc-pVDZ" },
           "system": {
             "max_gpu_memory_mb": 24000,
             "teams_per_node": 4,
             "gpus_per_team": 1
           },
           "keywords": {}
         }

   .. tab-item:: Python

      .. code-block:: python
         :caption: run.py

         from rush.exess import System, energy

         energy(
             topology_path="molecule_t.json",
             system=System(
                 max_gpu_memory_mb=24000,
                 teams_per_node=4,
                 gpus_per_team=1,
             ),
         )
```

```{eval-rst}
.. exess-params::

   .. exess-param:: max_gpu_memory_mb
      :type: uint64
      :default: unset
      :brief: Max GPU memory per process in MB.
      :note: info

      EXESS will crash if you request more memory than is available. When unset, a
      dynamic default of ~75% of GPU memory is used.

   .. exess-param:: oversubscribe_gpus
      :type: bool
      :default: false
      :brief: Allow multiple processes per GPU.
      :note: expert

      Expert setting to allow multiple processes per GPU.

   .. exess-param:: teams_per_node
      :type: uint32
      :default: 1
      :brief: Worker teams per node.

      ``teams_per_node`` and ``gpus_per_team`` control fragmentation scaling; one team
      handles one fragment queue.

   .. exess-param:: gpus_per_team
      :type: uint32
      :default: unset
      :brief: GPUs per team (overridable by ``MBE_NGPUS``).
      :note: info

      ``gpus_per_team`` can be overridden by the ``MBE_NGPUS`` environment variable.
```

### keywords

`keywords` contains method- and run-specific settings. See the {doc}`keyword reference <keywords>` for full details. In the EXESS CLI JSON, `keywords` must be present even if empty; rush-py supplies defaults when omitted.

:::{only} internal
## Default resolution order

Defaults are applied in the following order:

1. rush-py defaults: any non-`None` values set in Python (function defaults or dataclass defaults) are explicit values.
2. EXESS/libqdx JSON parser defaults (as defined in `libqdx.hpp`) for omitted fields.
3. EXESS internal defaults for values that remain unset after parsing.
:::

## Rush-py input mapping

The Rush Python client does not submit the full EXESS input JSON directly. Instead, it accepts a topology path (and sometimes residues path) plus keyword objects and constructs the EXESS params internally.

Key differences:

- `Model` in rush-py only includes `standard_orientation` and `force_cartesian_basis_sets`; `method`, `basis`, and `aux_basis` are function parameters.
- `keywords` in rush-py are passed as Python dataclasses (for `SCFKeywords`, `FragKeywords`, `ExportKeywords`, `OptimizationKeywords`, etc.).
- `frag_keywords` defaults to a dimer fragmentation setup; pass `frag_keywords=None` to run a full-system calculation.
- `external_charges` and some keyword groups (e.g., `rtat`, `integrals`) are not yet exposed in the rush-py API.

See the Rush guides for TRC objects and conversions: [Objects and TRC Files](../guides/03-objects-and-trc-files).

## Rush-py defaults

Default values set by the rush-py entry points:

- `exess.exess` / `exess.energy` / `exess.interaction_energy`: `driver="Energy"` (for `exess.exess`), `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`, `standard_orientation` unset (EXESS default `FullSystem`), `force_cartesian_basis_sets` unset (EXESS default `true`).
- `exess.chelpg`: `method="RestrictedHF"`, `basis="cc-pVDZ"`, `standard_orientation="None"`, `force_cartesian_basis_sets=false`.
- `exess.qmmm`: `method="RestrictedHF"`, `basis="STO-3G"`, `aux_basis=None`, `standard_orientation` unset (EXESS default `FullSystem`), `force_cartesian_basis_sets` unset (EXESS default `true`), `dt_ps=0.002`, `temperature_kelvin=290.0`, `pressure_atm=None`, gradient method `Analytical` with default step size.
- `exess.optimization`: `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`, `standard_orientation` unset (EXESS default `FullSystem`), `force_cartesian_basis_sets` unset (EXESS default `true`), `max_iters` required.

Keyword defaults for rush-py are documented in the keyword reference page.

## Input conversion tools

Several helpers can generate EXESS inputs:

- `parley.py` (https://github.com/JorgeG94/parley_exess) converts between XYZ and EXESS JSON. It can also add minimal defaults for `Dynamics` and `Optimization` drivers.
- `tools/input_transformer/create_json_input.jl` in the EXESS GitHub repository is a Julia helper for generating RHF inputs:

```bash
julia -E 'include("create_json_input.jl"); create_input_rhf("molecule.xyz", "BASIS")'
```
