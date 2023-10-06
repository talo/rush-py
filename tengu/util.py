from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, List, cast

from rdkit import Chem

from .api import ArgId

RDKitMol = Chem.rdchem.mol
SDMolSupplier: Callable[[str], Iterable[RDKitMol]] = cast(Any, Chem).SDMolSupplier
MolToPDBBlock: Callable[..., RDKitMol] = cast(Any, Chem).MolToPDBBlock


def mols_from_sdfpath(ligand_sdf_path: Path, *args: Any, **kwargs: Any) -> Iterable[RDKitMol]:
    """Return an iterator of ligand(s) from an sdf file.

    :param ligand_sdf_path: Path to an sdf file.
    :type  ligand_sdf_path: Path

    :yield LigRDKitMol: parsed rdkit molecule
    """
    # Trash rdkit only takes a path as a str
    yield from SDMolSupplier(str(ligand_sdf_path), *args, **kwargs)


def mols_to_pdbs(mols: Iterable[RDKitMol]) -> Iterable[StringIO]:
    yield from (StringIO(MolToPDBBlock(mol)) for mol in mols)


def sdf_to_pdbs(sdf: str | ArgId) -> List[StringIO]:
    return mols_to_pdbs(mols_from_sdfpath(str))
