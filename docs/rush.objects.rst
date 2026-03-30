:tocdepth: 3

Object Store Utilities 
======================

Rush stores some of the data used in runs in its own object store. This API
exposes Rush object-store references and helper functions for uploading,
fetching, and saving objects. It also includes TRC-specific helpers for working
with TRC data in the object store, which are sometimes returned by Rush
modules.

.. currentmodule:: rush.objects

Identifiers
-----------

.. autodata:: ObjectID

Object References
-----------------

.. autoclass:: RushObject
   :members:
   :undoc-members:

TRC Object Support
------------------

.. autoclass:: TRCRef
   :members:
   :undoc-members:

.. autoclass:: TRCPaths
   :members:
   :undoc-members:

Object Store Functions
----------------------

.. autofunction:: upload_object

.. autofunction:: fetch_object

.. autofunction:: save_object

.. autofunction:: save_json
