:tocdepth: 3

Runs
====

The runs API covers the public run lifecycle surface: identifiers, submission
configuration, run handles, metadata, and helper functions for querying,
collecting, and deleting runs.

.. currentmodule:: rush.runs

Identifiers and Type Aliases
----------------------------

.. autodata:: RunID

.. autotype:: RunStatus

.. autotype:: Target

.. autotype:: StorageUnit

Submission Types
----------------

.. autoclass:: RunSpec
   :members:
   :undoc-members:

.. autoclass:: RunOpts
   :members:
   :undoc-members:

Run Handles and Metadata
------------------------

.. autoclass:: Run
   :members:
   :undoc-members:

.. autoclass:: RunInfo
   :members:
   :undoc-members:

.. autoclass:: RunError
   :members:
   :undoc-members:

Run Queries and Collection
--------------------------

.. autofunction:: fetch_runs

.. autofunction:: fetch_run_info

.. autofunction:: collect_run

.. autofunction:: delete_run
