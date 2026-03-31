:tocdepth: 3

EXESS
=====

For detailed documentation on EXESS capabilities, keywords, and examples, see the `EXESS Documentation <exess/index.html>`_.

.. automodule:: rush.exess._energy

.. currentmodule:: rush.exess

Run Submission
--------------

.. autofunction:: calculate

.. autofunction:: energy

.. autofunction:: interaction_energy

Input Types
-----------------------

.. autoclass:: Model
   :members:
   :undoc-members:

.. autoclass:: System
   :members:
   :undoc-members:

.. autoclass:: SCFKeywords
   :members:
   :undoc-members:

.. autoclass:: FragKeywords
   :members:
   :undoc-members:

.. autoclass:: KSDFTKeywords
   :members:
   :undoc-members:

.. autoclass:: ExportKeywords
   :members:
   :undoc-members:

.. autotype:: MethodT

.. autotype:: BasisT

.. autotype:: AuxBasisT

.. autotype:: StandardOrientationT

.. autotype:: TensorLike

.. autotype:: ConvergenceMetricT

.. autotype:: FockBuildTypeT

.. autotype:: FragmentLevelT

.. autotype:: CutoffTypeT

.. autotype:: DistanceMetricT

Descriptor Grids
----------------

.. autoclass:: DescriptorGrid
   :members:
   :undoc-members:

.. autoclass:: StandardDescriptorGrid
   :members:
   :undoc-members:

.. autoclass:: RegularDescriptorGrid
   :members:
   :undoc-members:

.. autoclass:: CustomDescriptorGrid
   :members:
   :undoc-members:

.. autoclass:: XCGridParameters
   :members:
   :undoc-members:

.. autoclass:: DefaultGridResolution
   :members:
   :undoc-members:

.. autoclass:: CustomGridResolution
   :members:
   :undoc-members:

.. autoclass:: ClosestAtomBatching
   :members:
   :undoc-members:

.. autoclass:: OctreeBatching
   :members:
   :undoc-members:

.. autoclass:: Octree
   :members:
   :undoc-members:

.. autoclass:: SpaceFillingBatching
   :members:
   :undoc-members:

.. autoclass:: GauXCBatching
   :members:
   :undoc-members:

.. autotype:: RadialQuadT

.. autotype:: PruningSchemeT

.. autotype:: XCGridResolutionT

.. autotype:: XCBatchingSchemeT

.. autotype:: KSDFTMethodT

Result Types
------------

.. autoclass:: Result
   :members:
   :undoc-members:

.. autoclass:: ResultPaths
   :members:
   :undoc-members:

.. autoclass:: ResultRef
   :members:
   :undoc-members:

.. autoclass:: Calculation
   :members:
   :undoc-members:

.. autoclass:: ManyBodyExpansion
   :members:
   :undoc-members:

.. autoclass:: Nmer
   :members:
   :undoc-members:
