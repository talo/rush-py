#!/usr/bin/env python3
"""
Boltz module helpers for the Rush Python client.

Boltz predicts folded structures from protein sequences, optional ligands, and
MSA inputs. The fetched output is parsed into Python-friendly result objects,
while the saved output writes the model and JSON artifacts into the workspace.
"""

import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
from gql.transport.exceptions import TransportQueryError

from rush.convert import _single_trc, from_json, from_pdb
from rush.mol import TRC

from ._output_types import TRCSavedResult
from .client import (
    RunID,
    RunOpts,
    RunSpec,
    _get_project_id,
    _json_content_name,
    _submit_rex,
    collect_run,
    fetch_object,
    save_json,
    save_object,
    upload_object,
)
from .utils import dict_to_vec_of_tuples_str, optional_str


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


@dataclass
class BoltzMetrics:
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
class BoltzAffinities:
    """Optional affinity predictions returned for binding runs."""

    affinity_pred_value: float
    affinity_probability_binary: float
    affinity_pred_value1: float
    affinity_probability_binary1: float
    affinity_pred_value2: float
    affinity_probability_binary2: float


@dataclass
class BoltzResult:
    """
    Parsed Boltz fold result.

    Use `fetch_outputs(boltz(..., collect=True))` to return these dataclasses
    in memory, or `save_outputs(boltz(..., collect=True))` to save the model
    and JSON outputs into the workspace.
    """

    model: TRC
    metrics: BoltzMetrics
    plddt: npt.NDArray[np.float32]
    pae: npt.NDArray[np.float32]
    affinities: BoltzAffinities | None = None


@dataclass(frozen=True)
class BoltzSavedResult:
    """Workspace paths for a saved Boltz result bundle."""

    model: TRCSavedResult
    metrics: Path
    plddt: Path
    pae: Path
    affinities: Path | None = None


@overload
def boltz(
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
    collect: Literal[False] = False,
) -> RunID: ...
@overload
def boltz(
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
    collect: Literal[True] = True,
) -> list[tuple[Any, ...]]: ...
@overload
def boltz(
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
    collect: bool = False,
) -> list[tuple[Any, ...]] | RunID: ...


def boltz(
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
    collect: bool = False,
) -> list[tuple[Any, ...]] | RunID:
    """
    Run Boltz on protein and ligand inputs.

    The collected result is a list of Boltz output bundles, each containing a
    model TRC, confidence metrics, confidence arrays, and optional affinity
    metrics.
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
            if template_path is not None
            else "None"
        ),
        topology_vobj_path=topology_vobj["path"] if has_template else "",
        residues_vobj_path=residues_vobj["path"] if has_template else "",
        chains_vobj_path=chains_vobj["path"] if has_template else "",
    )
    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if not collect:
            return run_id

        out = collect_run(run_id)
        assert isinstance(out, list)
        for out_i in out:
            assert isinstance(out_i, list)
        out = out[0]
        out = [tuple(out_i) for out_i in out]
        return out

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise


def _decode_float_array(output: dict[str, Any]) -> npt.NDArray[np.float32]:
    raw = base64.b64decode(output["data"])
    shape = tuple(int(dim) for dim in output["shape"])
    return np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(shape)


def _fetch_trc_output(
    model_obj: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> TRC:
    topology_obj, residues_obj, chains_obj = model_obj

    def load_component(output_obj: dict[str, Any]) -> Any:
        output = fetch_object(output_obj["path"])
        if isinstance(output, bytes):
            output = output.decode()
        return json.loads(output)

    return from_json(
        {
            "topology": load_component(topology_obj),
            "residues": load_component(residues_obj),
            "chains": load_component(chains_obj),
        }
    )


def fetch_outputs(res: list[tuple[Any, ...]]) -> list[BoltzResult]:
    """
    Fetch Boltz outputs into parsed Python objects in memory.

    Args:
        res: Collected output from boltz(), one item per predicted model.

    Returns:
        Parsed BoltzResult objects in the same order as the collected outputs.
    """

    def fetch_output(res_i: tuple[Any, ...]) -> BoltzResult:
        model_obj, metrics, plddt_obj, pae_obj, affinities = res_i
        plddt = fetch_object(plddt_obj["path"])
        if isinstance(plddt, bytes):
            plddt = plddt.decode()
        pae = fetch_object(pae_obj["path"])
        if isinstance(pae, bytes):
            pae = pae.decode()

        return BoltzResult(
            model=_fetch_trc_output(model_obj),
            metrics=BoltzMetrics(**metrics),
            plddt=_decode_float_array(json.loads(plddt)),
            pae=_decode_float_array(json.loads(pae)),
            affinities=BoltzAffinities(**affinities)
            if affinities is not None
            else None,
        )

    return [fetch_output(res_i) for res_i in res]


def save_outputs(res: list[tuple[Any, ...]]) -> list[BoltzSavedResult]:
    """
    Save Boltz outputs into the workspace.

    Args:
        res: Collected output from boltz(), one item per predicted model.

    Returns:
        Saved Boltz result bundles in the same order as the collected outputs.
    """

    def save_output(res_i: Any) -> BoltzSavedResult:
        model_obj, metrics, plddt_obj, pae_obj, affinities = res_i
        topology_obj, residues_obj, chains_obj = model_obj

        return BoltzSavedResult(
            model=TRCSavedResult(
                topology=save_object(topology_obj["path"]),
                residues=save_object(residues_obj["path"]),
                chains=save_object(chains_obj["path"]),
            ),
            metrics=save_json(
                metrics,
                name=_json_content_name("boltz_metrics", metrics),
            ),
            plddt=save_object(plddt_obj["path"]),
            pae=save_object(pae_obj["path"]),
            affinities=(
                save_json(
                    affinities,
                    name=_json_content_name("boltz_affinities", affinities),
                )
                if affinities is not None
                else None
            ),
        )

    return [save_output(res_i) for res_i in res]
