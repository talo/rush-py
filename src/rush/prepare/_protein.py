"""
Protein preparation module for the Rush Python client.

This module supports system preparation workflows such as converting PDB inputs
to TRC, protonating and optimizing hydrogen positions, and augmenting
structures with connectivity and formal charge information before downstream
calculations.

Usage::

    from rush import prepare

    result = prepare.protein("protein.pdb").fetch()
    print(result.topology.symbols)
"""

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Literal

from gql.transport.exceptions import TransportQueryError

from rush import Chains, Residues, Topology

from .._trc import TRCPaths, TRCRef, to_chains_vobj, to_residues_vobj, to_topology_vobj
from .._utils import optional_str
from ..client import (
    RunOpts,
    RunSpec,
    RushObject,
    _get_project_id,
    _submit_rex,
)
from ..convert import _single_trc, from_json, from_pdb
from ..mol import TRC
from ..run import RushRun

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultRef:
    """Lightweight reference to prepare-protein output in the Rush object store.

    May contain multiple TRC triplets if the input PDB has multiple models.
    """

    models: list[TRCRef]

    def __getitem__(self, index: int) -> TRCRef:
        return self.models[index]

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self) -> Iterator[TRCRef]:
        return iter(self.models)

    @classmethod
    def from_raw_output(cls, res: Any) -> "ResultRef":
        """Parse raw ``collect_run`` output into a ``ResultRef``.

        The raw output is a list of groups, where each group is a list of
        3 dicts (topology, residues, chains objects).  Multi-model PDBs
        produce multiple groups.
        """
        if not isinstance(res, list) or len(res) == 0:
            raise ValueError(
                f"prepare_protein should return a non-empty list, "
                f"got {type(res).__name__}"
                f"{f' with {len(res)} items' if hasattr(res, '__len__') else ''}."
            )

        models: list[TRCRef] = []
        for i, group in enumerate(res):
            if not isinstance(group, list) or len(group) != 3:
                raise ValueError(
                    f"prepare_protein output group {i} expected a list of 3 elements, "
                    f"got {type(group).__name__}"
                    f"{f' with {len(group)} items' if isinstance(group, list) else ''}."
                )
            topo, resid, chain = group[0], group[1], group[2]
            if (
                not isinstance(topo, dict)
                or not isinstance(resid, dict)
                or not isinstance(chain, dict)
            ):
                raise ValueError(
                    f"prepare_protein output group {i} elements must be dicts."
                )
            models.append(
                TRCRef(
                    topology=RushObject.from_dict(topo),
                    residues=RushObject.from_dict(resid),
                    chains=RushObject.from_dict(chain),
                )
            )

        return cls(models=models)

    def fetch(self) -> list[TRC]:
        """Download prepare-protein output and parse into TRCs.

        Returns one TRC per model in the input PDB.  Most PDBs contain a
        single model, so ``result[0]`` is the common pattern.
        """
        return [model.fetch() for model in self.models]

    def save(self) -> list[TRCPaths]:
        """Download prepare-protein output and save to the workspace.

        Returns one TRCPaths per model in the input PDB.
        """
        return [model.save() for model in self.models]


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def protein(
    mol: TRC
    | TRCRef
    | tuple[
        Path | str | RushObject | Topology,
        Path | str | RushObject | Residues,
        Path | str | RushObject | Chains,
    ]
    | Path
    | str,
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
    opt: bool | None = None,
    debump: bool | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """
    Submit a prepare-protein job for a PDB or TRC file.

    Returns a :class:`~rush.run.RushRun` handle. Call ``.fetch()`` to get the
    parsed TRC, or ``.save()`` to write the output files to disk.
    """

    # Upload inputs
    match mol:
        case TRC():
            trc_ref = TRCRef.upload(mol)
        case TRCRef():
            trc_ref = mol
        case (t, r, c):
            trc_ref = TRCRef(
                RushObject.from_dict(to_topology_vobj(t)),
                RushObject.from_dict(to_residues_vobj(r)),
                RushObject.from_dict(to_chains_vobj(c)),
            )
        case Path() | str():
            input_path = mol
            if isinstance(input_path, str):
                input_path = Path(input_path)

            with open(input_path) as f:
                if input_path.suffix == ".pdb":
                    trc = from_pdb(f.read())
                else:
                    trc = from_json(json.load(f))
            trc = _single_trc(trc, input_path)
            trc_ref = TRCRef.upload(trc)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  prepare_protein = λ topology residues chains →
    prepare_protein_rex_s
      ($run_spec)
      (prepare_protein_rex::PrepareProteinOptions {
        ph = $ph,
        naming_scheme = $naming_scheme,
        capping_style = $capping_style,
        truncation_threshold = $truncation_threshold,
        opt = $opt,
        debump = $debump,
      })
      [( (obj_j topology), (obj_j residues), (obj_j chains) )]
in
  prepare_protein "$topology_vobj_path" "$residues_vobj_path" "$chains_vobj_path"
""").substitute(
        run_spec=run_spec._to_rex(),
        ph=optional_str(ph),
        naming_scheme=optional_str(
            naming_scheme.title() if naming_scheme is not None else None,
            prefix="prepare_protein_rex::NamingScheme::",
        ),
        capping_style=optional_str(
            capping_style.title() if capping_style is not None else None,
            prefix="prepare_protein_rex::CappingStyle::",
        ),
        truncation_threshold=optional_str(truncation_threshold),
        opt=optional_str(opt),
        debump=optional_str(debump),
        topology_vobj_path=trc_ref.topology.path,
        residues_vobj_path=trc_ref.residues.path,
        chains_vobj_path=trc_ref.chains.path,
    )
    try:
        return RushRun(
            _submit_rex(_get_project_id(), rex, run_opts),
            ResultRef,
        )

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
