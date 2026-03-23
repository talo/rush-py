#!/usr/bin/env python3
"""
Boltz module for the Rush Python client.

Boltz predicts folded structures from protein sequences, optional ligands, and
MSA inputs. The fetched output is parsed into Python-friendly result objects,
while the saved output writes the model and JSON artifacts into the workspace.

Usage::

    from rush import boltz

    ref = boltz.fold([ProteinSequence(...)]).collect()
    results = ref.fetch()
    print(results[0].metrics.confidence_score)
"""

import base64
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import numpy.typing as npt
from gql.transport.exceptions import TransportQueryError

from rush.convert import _single_trc, from_json, from_pdb
from rush.mol import TRC

from ._trc import TRCPaths, TRCRef
from ._utils import dict_to_vec_of_tuples_str, optional_str
from .client import (
    RunOpts,
    RunSpec,
    RushObject,
    _get_project_id,
    _json_content_name,
    _submit_rex,
    fetch_object,
    save_json,
    upload_object,
)
from .run import RushRun

# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


@dataclass
class Modification:
    position: int
    ccd: str


@dataclass
class ProteinSequence:
    id: list[str]
    sequence: str
    msa: dict[str, str] | Path | str
    modifications: list[Modification] | None = None
    cyclic: bool | None = None

    def _to_rex(self):
        if isinstance(self.msa, Path) or isinstance(self.msa, str):
            self.msa = upload_object(self.msa)

        return Template(
            """(boltz2_rex::Sequence::Protein {
          id = $id,
          sequence = "$sequence",
          msa = VirtualObject { path = "$msa", format = ObjectFormat::bin, size = 0 },
          modifications = None,
          cyclic = $cyclic,
        })"""
        ).substitute(
            id=f"[{', '.join([f'"{v}"' for v in self.id])}]",
            sequence=self.sequence,
            msa=self.msa["path"],
            cyclic=optional_str(self.cyclic),
        )


@dataclass
class LigandSequence:
    id: list[str]
    smiles: str

    def _to_rex(self):
        return Template(
            """(boltz2_rex::Sequence::Ligand {
          id = $id,
          smiles = "$smiles",
        })"""
        ).substitute(
            id=f"[{', '.join([f'"{v}"' for v in self.id])}]",
            smiles=self.smiles,
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    """Summary confidence metrics returned by Boltz."""

    confidence_score: float
    ptm: float
    iptm: float
    ligand_iptm: float
    protein_iptm: float
    complex_plddt: float
    complex_iplddt: float
    complex_pde: float
    complex_ipde: float


@dataclass
class Affinities:
    """Optional affinity predictions returned for binding runs."""

    affinity_pred_value: float
    affinity_probability_binary: float
    affinity_pred_value1: float
    affinity_probability_binary1: float
    affinity_pred_value2: float
    affinity_probability_binary2: float


@dataclass
class Result:
    """
    Parsed Boltz fold result.

    Returned by ``ResultRef.fetch()`` — one per diffusion sample.
    """

    model: TRC
    metrics: Metrics
    plddt: npt.NDArray[np.float32]
    pae: npt.NDArray[np.float32]
    affinities: Affinities | None = None


@dataclass(frozen=True)
class ResultPaths:
    """Workspace paths for a saved Boltz result bundle."""

    model: TRCPaths
    metrics: Path
    plddt: Path
    pae: Path
    affinities: Path | None = None


def _decode_float_array(output: dict[str, Any]) -> npt.NDArray[np.float32]:
    raw = base64.b64decode(output["data"])
    shape = tuple(int(dim) for dim in output["shape"])
    return np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(shape)


@dataclass(frozen=True)
class DiffusionSampleRef:
    """Parsed reference to a single Boltz diffusion sample."""

    model: TRCRef
    metrics: dict[str, Any]
    plddt: RushObject
    pae: RushObject
    affinities: dict[str, Any] | None


@dataclass(frozen=True)
class ResultRef:
    """Lightweight reference to Boltz outputs in the Rush object store.

    Each element of *samples* is a parsed :class:`DiffusionSampleRef` for one
    diffusion sample.
    """

    diffusion_samples: list[DiffusionSampleRef]

    def __getitem__(self, index: int) -> DiffusionSampleRef:
        return self.diffusion_samples[index]

    def __len__(self) -> int:
        return len(self.diffusion_samples)

    def __iter__(self) -> Iterator[DiffusionSampleRef]:
        return iter(self.diffusion_samples)

    @classmethod
    def from_raw_output(cls, res: Any) -> "ResultRef":
        """Parse raw ``collect_run`` output into a ``ResultRef``."""
        if not isinstance(res, list) or len(res) == 0:
            raise ValueError(
                f"boltz output received unexpected format: {type(res).__name__}"
            )
        # collect_run returns [[sample0, sample1, ...]] — outer list wraps
        # the single run, inner list contains one tuple per diffusion sample.
        out = res[0]
        diffusion_samples: list[DiffusionSampleRef] = []
        for item in out:
            model_obj, metrics, plddt_obj, pae_obj, affinities = item
            topo, resid, chain = model_obj
            diffusion_samples.append(
                DiffusionSampleRef(
                    model=TRCRef(
                        topology=RushObject.from_dict(topo),
                        residues=RushObject.from_dict(resid),
                        chains=RushObject.from_dict(chain),
                    ),
                    metrics=metrics,
                    plddt=RushObject.from_dict(plddt_obj),
                    pae=RushObject.from_dict(pae_obj),
                    affinities=affinities,
                )
            )
        return cls(diffusion_samples=diffusion_samples)

    def fetch(self) -> Iterator[Result]:
        """Download Boltz outputs and parse into Python objects.

        Yields one :class:`Result` per diffusion sample.  Each sample is
        downloaded lazily on iteration — stop early to skip downloads.
        """
        for sample in self.diffusion_samples:
            plddt_raw = fetch_object(sample.plddt.path)
            if isinstance(plddt_raw, bytes):
                plddt_raw = plddt_raw.decode()
            pae_raw = fetch_object(sample.pae.path)
            if isinstance(pae_raw, bytes):
                pae_raw = pae_raw.decode()

            yield Result(
                model=sample.model.fetch(),
                metrics=Metrics(**sample.metrics),
                plddt=_decode_float_array(json.loads(plddt_raw)),
                pae=_decode_float_array(json.loads(pae_raw)),
                affinities=(
                    Affinities(**sample.affinities)
                    if sample.affinities is not None
                    else None
                ),
            )

    def save(self) -> Iterator[ResultPaths]:
        """Download Boltz outputs and save to the workspace.

        Yields one :class:`ResultPaths` per diffusion sample.  Each sample is
        downloaded lazily on iteration — stop early to skip downloads.
        """
        for sample in self.diffusion_samples:
            yield ResultPaths(
                model=sample.model.save(),
                metrics=save_json(
                    sample.metrics,
                    name=_json_content_name("boltz_metrics", sample.metrics),
                ),
                plddt=sample.plddt.save(),
                pae=sample.pae.save(),
                affinities=(
                    save_json(
                        sample.affinities,
                        name=_json_content_name("boltz_affinities", sample.affinities),
                    )
                    if sample.affinities is not None
                    else None
                ),
            )


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def fold(
    sequences: list[ProteinSequence | LigandSequence],
    recycling_steps: int | None = None,
    sampling_steps: int | None = None,
    diffusion_samples: int | None = None,
    step_scale: float | None = None,
    affinity_binder_chain_id: str | None = None,
    affinity_mw_correction: bool | None = None,
    sampling_steps_affinity: int | None = None,
    diffusion_samples_affinity: bool | None = None,
    max_msa_seqs: int | None = None,
    subsample_msa: bool | None = None,
    num_subsampled_msa: int | None = None,
    use_potentials: bool | None = None,
    seed: int | None = None,
    template_path: Path | str | None = None,
    template_threshold_angstroms: float | None = None,
    template_chain_mapping: dict[str, str] | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """
    Submit a Boltz fold job for the given protein/ligand *sequences*.

    Returns a :class:`~rush.run.RushRun` handle. Call ``.collect()`` to get a
    :class:`ResultRef`, then ``.fetch()`` or ``.save()`` on that ref.
    """

    # If necessary, upload template TRC inputs
    has_template = template_path is not None
    if template_path is not None:
        if isinstance(template_path, str):
            template_path = Path(template_path)
        with open(template_path) as f:
            if template_path.suffix == ".pdb":
                trc = from_pdb(f.read())
            else:
                trc = from_json(json.load(f))
        trc = _single_trc(trc, template_path)
        with (
            NamedTemporaryFile(mode="w") as t_f,
            NamedTemporaryFile(mode="w") as r_f,
            NamedTemporaryFile(mode="w") as c_f,
        ):
            json.dump(trc.topology.to_json(), t_f)
            json.dump(trc.residues.to_json(), r_f)
            json.dump(trc.chains.to_json(), c_f)
            t_f.seek(0)
            r_f.seek(0)
            c_f.seek(0)
            topology_vobj = upload_object(t_f.name)
            residues_vobj = upload_object(r_f.name)
            chains_vobj = upload_object(c_f.name)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  boltz = λ topology residues chains →
    boltz2_rex_s
      ($run_spec)
      (boltz2_rex::Boltz2Config {
        recycling_steps = $maybe_recycling_steps,
        sampling_steps = $maybe_sampling_steps,
        diffusion_samples = $maybe_diffusion_samples,
        step_scale = $maybe_step_scale,
        affinity_binder_chain_id = $maybe_affinity_binder_chain_id,
        affinity_mw_correction = $maybe_affinity_mw_correction,
        sampling_steps_affinity = $maybe_sampling_steps_affinity,
        diffusion_samples_affinity = $maybe_diffusion_samples_affinity,
        max_msa_seqs = $maybe_max_msa_seqs,
        subsample_msa = $maybe_subsample_msa,
        num_subsampled_msa = $maybe_num_subsampled_msa,
        use_potentials = $maybe_use_potentials,
        seed = $maybe_seed,
        template_threshold_angstroms = $maybe_template_threshold_angstroms,
        template_chain_mapping = $maybe_template_chain_mapping,
      })
      $sequences
      $template_trc_expr
in
  boltz "$topology_vobj_path" "$residues_vobj_path" "$chains_vobj_path"
""").substitute(
        run_spec=run_spec._to_rex(),
        maybe_recycling_steps=optional_str(recycling_steps),
        maybe_sampling_steps=optional_str(sampling_steps),
        maybe_diffusion_samples=optional_str(diffusion_samples),
        maybe_step_scale=optional_str(step_scale),
        maybe_affinity_binder_chain_id=optional_str(affinity_binder_chain_id),
        maybe_affinity_mw_correction=optional_str(affinity_mw_correction),
        maybe_sampling_steps_affinity=optional_str(sampling_steps_affinity),
        maybe_diffusion_samples_affinity=optional_str(diffusion_samples_affinity),
        maybe_max_msa_seqs=optional_str(max_msa_seqs),
        maybe_subsample_msa=optional_str(subsample_msa),
        maybe_num_subsampled_msa=optional_str(num_subsampled_msa),
        maybe_use_potentials=optional_str(use_potentials),
        maybe_seed=optional_str(seed),
        maybe_template_threshold_angstroms=optional_str(template_threshold_angstroms),
        maybe_template_chain_mapping=(
            f"(Some {dict_to_vec_of_tuples_str(template_chain_mapping)})"
            if template_chain_mapping is not None
            else "None"
        ),
        sequences=f"[\n        {',\n        '.join([f'{seq._to_rex()}' for seq in sequences])},\n      ]",
        template_trc_expr=(
            "(Some ((obj_j topology), (obj_j residues), (obj_j chains)) )"
            if has_template
            else "None"
        ),
        topology_vobj_path=topology_vobj["path"] if has_template else "",
        residues_vobj_path=residues_vobj["path"] if has_template else "",
        chains_vobj_path=chains_vobj["path"] if has_template else "",
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
