:tocdepth: 3

Hyper Minimize
==============

.. automodule:: rush.hyper._hyper_minimize_sumo

.. currentmodule:: rush.hyper

Run Submission
--------------

.. autofunction:: hyper_minimize_sumo

Input Types
-----------

.. autoclass:: HyperMinimizeConfig
   :members:
   :undoc-members:

.. autoclass:: MinimizeInput
   :members:
   :undoc-members:

Result Types
------------

Minimization returns :class:`rush.hyper.TRCBatchResultRef` with per-item
results that can be fetched as ``TRC`` objects or ``ItemError`` values.