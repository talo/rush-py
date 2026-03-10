QDX Documentation
==================

Welcome to QDX's documentation hub. These docs cover both the **Rush Python
SDK** for programmatic access to the `Rush platform <https://qdx.co/rush>`__,
which is how we make our quantum chemistry technology available along with
vertical applications to drug discovery developed by our R&D team, and the
**EXESS** quantum chemistry engine that powers many of its calculations.

If you prefer a visual interface over scripting, you can use EXESS directly
`here <https://exess.qdx.co/exess-flow>`__ and the rest of Rush
`here <https://rush.cloud/login>`__ with no setup required.

----

Rush Python SDK
---------------

The Rush Python SDK (``rush-py``) lets you script and automate workflows on
`Rush <https://qdx.co/rush>`__. Through it you can submit quantum
chemistry, molecular dynamics, protein folding, and other computational
chemistry jobs, then retrieve and process results — all from Python.

**What's covered:**

- **Guides** — account setup, installation, authentication, working with
  molecular structures, and hardware selection.
- **Tutorials** — step-by-step walkthroughs for common workflows like
  single-point energies, geometry optimisations, CHELPG charges, interaction
  energies, and QM/MM.
- **API Reference** — full documentation of the client, data model, file
  converters, and every computation module (EXESS, NN-xTB, Boltz, Auto3D,
  PBSA, and more).

**Who is this for?** Computational chemists and developers who
want to integrate Rush into scripts, pipelines, or applications. Start with the
:doc:`Getting Started <guides/01-getting-started>` guide.

.. toctree::
   :maxdepth: 1
   :caption: Rush Python SDK

   guides
   tutorials
   modules
   api

----

EXESS
-----

EXESS (the Extreme-scale Electronic Structure System) is QDX's GPU-accelerated
quantum chemistry engine. It supports workflows ranging from high-throughput quantum calculations on small
and medium systems to fragmentation-based, QM/MM (quantum mechanics/molecular
mechanics), and *ab initio* molecular dynamics simulations of large biomolecular
environments.

**What's covered:**

- **Concepts** — the regions-based workflow model, electronic structure methods,
  basis sets, DFT functionals, and correlated methods.
- **Usage** — how to run EXESS (via Rush or the CLI), input configuration, and
  keyword reference.
- **Results** — output formats, export options, and worked examples.
- **Reference** — complete basis set tables, grid definitions, hardware notes,
  and citations.

These docs are useful for checking whether EXESS is suitable for your workflow.
For information on how to actually run EXESS programmatically, see the
:doc:`Rush Python SDK <guides/01-getting-started>` docs.

.. toctree::
   :maxdepth: 1
   :caption: EXESS

   exess/overview
   exess/electronic-structure-methods
   exess/running
   exess/input
   exess/topologies
   exess/keywords
   exess/examples
   exess/outputs
   exess/reference
   exess/citations

----

* :ref:`search`
* :ref:`genindex`
