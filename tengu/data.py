"""
Utilities for creating QC JSON files (https://molssi-qc-schema.readthedocs.io/en/latest/index.html)
"""
from dataclasses import dataclass, field
from typing import Literal, Optional

import dataclasses_json

QCMethods = Literal["RHF", "RIMP2"]

QCBasisSet = Literal[
    "STO-2G",
    "STO-3G",
    "STO-4G",
    "STO-5G",
    "STO-6G",
    "3-21G",
    "4-31G",
    "5-21G",
    "6-21G",
    "6-31G",
    "6-31G*",
    "6-31G**",
    "6-31+G",
    "6-31+G*",
    "6-31+G**",
    "6-31++G",
    "6-31++G*",
    "6-31++G**",
    "6-31G(2df,p)",
    "6-31G(3df,3pd)",
    "6-31G(d,p)",
    "6-311G",
    "6-311G*",
    "6-311G**",
    "6-311+G",
    "6-311+G*",
    "6-311+G**",
    "6-311++G",
    "6-311++G*",
    "6-311++G**",
    "6-311G(d,p)",
    "6-311G(2df,2pd)",
    "6-311+G(2d,p)",
    "6-311++G(2d,2p)",
    "6-311++G(3df,3pd)",
    "PCSeg-0",
    "PCSeg-1",
    "PCSeg-2",
    "cc-pVDZ-RIFIT",
    "aug-cc-pVDZ-RIFIT",
    "cc-pVTZ-RIFIT",
    "aug-cc-pVTZ-RIFIT",
    "cc-pVDZ",
    "aug-cc-pVDZ",
    "cc-pVTZ",
    "aug-cc-pVTZ",
]


class DataClassJsonMixin(dataclasses_json.DataClassJsonMixin):
    """Override dataclass mixin so that we don't have `"property": null,`s in our output"""

    dataclass_json_config = dataclasses_json.config(  # type: ignore
        undefined=dataclasses_json.Undefined.EXCLUDE,
        exclude=lambda f: f is None,  # type: ignore
    )["dataclasses_json"]


@dataclass
class QCJSONFragments(DataClassJsonMixin):
    """ "fragments" in qc json"""

    fragid: list[int] = field(default_factory=list)
    nfrag: int = 0
    broken_bonds: list[int] = field(default_factory=list)
    fragment_charges: list[int] = field(default_factory=list)
    frag_size: list[int] = field(default_factory=list)


@dataclass
class QCMol(DataClassJsonMixin):
    """ "molecule" in qc json"""

    geometry: list[float] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    fragments: QCJSONFragments = QCJSONFragments()


@dataclass
class QCModel(DataClassJsonMixin):
    """ "model" in qc json"""

    method: QCMethods = "RHF"
    fragmentation: bool = False
    basis: QCBasisSet = "6-31G*"
    aux_basis: str = "cc-pVDZ-RIFIT"


@dataclass
class QCSCF(DataClassJsonMixin):
    """ "keywords.scf" in qc json"""

    niter: int = 40
    ndiis: int = 12
    scf_conv: float = 1e-6
    dynamic_screening_threshold_exp: int = 10
    debug: bool | None = None
    convergence_metric: str = "diis"


@dataclass
class QCFrag(DataClassJsonMixin):
    """ "keywords.frag" in qc json"""

    reference_fragment: Optional[int] = None
    method: str = "MBE"
    fragmentation_level: int = 2
    ngpus_per_node: int = 4
    subset_of_fragments_to_calculate: Optional[list[int]] = None
    monomer_cutoff: int = 40
    monomer_mp2_cutoff: int = 40
    dimer_cutoff: int = 40
    dimer_mp2_cutoff: int = 40
    trimer_mp2_cutoff: int = 20
    lattice_energy_calc: bool = False
    trimer_cutoff: int = 30


@dataclass
class QCKeywords(DataClassJsonMixin):
    """ "keywords" in qc json"""

    scf: QCSCF = QCSCF()
    frag: QCFrag = QCFrag()


@dataclass
class QDXQCMol(DataClassJsonMixin):
    """ "molecule" in qc json"""

    geometry: list[float] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    fragments: Optional[list[list[int]]] = field(default_factory=list)
    connectivity: Optional[list[tuple[int, int, int]]] = field(default_factory=list)
    fragment_charges: Optional[list[int]] = field(default_factory=list)


@dataclass
class QDXQCInput(DataClassJsonMixin):
    """Final qc json input file"""

    topology: QDXQCMol = QDXQCMol()
    model: QCModel = QCModel()
    keywords: QCKeywords = QCKeywords()
    driver: str = "energy"


@dataclass
class QDXEnergy(DataClassJsonMixin):
    energy_type: Literal["lattice", "SPE"] | None  # Whether the calculation is lattice against a ref. monomer
    hf_total: float  # Final single-point energy (HF)
    mp_ss_total: float  # Final single-point energy correction (MP2 same-spin)
    mp_os_total: float  # Final single-point energy correction (MP2 opposite-spin)

    hf: list[float]  # HF energy per fragment group
    mp_ss: list[float]  # MP2 same-spin corrections per fragment group
    mp_os: list[float]  # MP2 opposite-spin corrections per fragment group
    n_mers: list[list[int]]  # Fragment groups
    n_mer_distances: list[float] | None  # Distances between fragments
