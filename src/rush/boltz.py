#!/usr/bin/env python3
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile

from gql.transport.exceptions import TransportQueryError

from rush.convert.json import to_json
from rush.convert.pdb import from_pdb

from .client import (
    PROJECT_ID,
    RunOpts,
    RunSpec,
    _submit_rex,
    collect_run,
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
    collect=False,
):
    # If necessary, upload template TRC inputs
    has_template = template_path is not None
    if template_path is not None:
        if isinstance(template_path, str):
            template_path = Path(template_path)
        with open(template_path) as f:
            if template_path.suffix == ".pdb":
                trc = from_pdb(f.read())
                trc_str = to_json(trc)
                trc_dict = json.loads(trc_str)[0]
            else:
                trc_dict = json.load(f)
        with (
            NamedTemporaryFile(mode="w") as t_f,
            NamedTemporaryFile(mode="w") as r_f,
            NamedTemporaryFile(mode="w") as c_f,
        ):
            json.dump(trc_dict["topology"], t_f)
            json.dump(trc_dict["residues"], r_f)
            json.dump(trc_dict["chains"], c_f)
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
        run_id = _submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
