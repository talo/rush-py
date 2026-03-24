"""
Structure preparation module for the Rush Python client.

Usage::

    from rush import prepare

    # Protein only
    result = prepare.protein("protein.pdb").fetch()

    # Protein-ligand complex
    result = prepare.protein_ligand("complex.pdb", ligand_names=["LIG"]).fetch()
"""

from ._protein import ResultRef, protein
from ._protein_ligand import protein_ligand

__all__ = [
    "protein",
    "protein_ligand",
    "ResultRef",
]
