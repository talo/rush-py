#!/usr/bin/env python3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template

from gql.transport.exceptions import TransportQueryError

from .client import (
    RunError,
    RunOpts,
    RunSpec,
    _get_project_id,
    _json_content_name,
    _submit_rex,
    collect_run,
    save_json,
    upload_object,
)
from .utils import float_to_str


@dataclass
class PBSAResult:
    """
    Parsed PBSA result.

    Use `fetch_outputs(pbsa(..., collect=True))` to return this dataclass in
    memory, or `save_outputs(pbsa(..., collect=True))` to save the same values
    as JSON in the workspace.
    """

    solvation_energy: float
    polar_solvation_energy: float
    nonpolar_solvation_energy: float


def pbsa(
    topology_path: Path | str,
    solute_dielectric: float,
    solvent_dielectric: float,
    solvent_radius: float,
    ion_concentration: float,
    temperature: float,
    spacing: float,
    sasa_gamma: float,
    sasa_beta: float,
    sasa_n_samples: int,
    convergence: float,
    box_size_factor: float,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    """
    Run PBSA on the system in the QDX topology file at `topology_path`.

    Returns the
    total solvation energy, polar solvation energy, and nonpolar solvation energy
    of the system, in Hartrees.
    """

    # Upload inputs
    topology_vobj = upload_object(topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  pbsa = λ topology →
    pbsa_rex_s
      ($run_spec)
      (pbsa_rex::PBSAParameters {
        solute_dielectric = $solute_dielectric,
        solvent_dielectric = $solvent_dielectric,
        solvent_radius = $solvent_radius,
        ion_concentration = $ion_concentration,
        temperature = $temperature,
        spacing = $spacing,
        sasa_gamma = $sasa_gamma,
        sasa_beta = $sasa_beta,
        sasa_n_samples = $sasa_n_samples,
        convergence = $convergence,
        box_size_factor = $box_size_factor,
      })
      (obj_j topology)
in
  pbsa "$topology_vobj_path"
""").substitute(
        run_spec=run_spec._to_rex(),
        solute_dielectric=float_to_str(solute_dielectric),
        solvent_dielectric=float_to_str(solvent_dielectric),
        solvent_radius=float_to_str(solvent_radius),
        ion_concentration=float_to_str(ion_concentration),
        temperature=float_to_str(temperature),
        spacing=float_to_str(spacing),
        sasa_gamma=float_to_str(sasa_gamma),
        sasa_beta=float_to_str(sasa_beta),
        sasa_n_samples=sasa_n_samples,
        convergence=float_to_str(convergence),
        box_size_factor=float_to_str(box_size_factor),
        topology_vobj_path=topology_vobj["path"],
    )
    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def _unwrap_output(
    res: list[float] | tuple[float, ...] | str | RunError,
) -> PBSAResult | str | RunError:
    if isinstance(res, (str, RunError)):
        return res

    if isinstance(res, (list, tuple)) and len(res) == 3:
        return PBSAResult(float(res[0]), float(res[1]), float(res[2]))

    return RunError(
        f"Error: pbsa output helper received unexpected format: {type(res)}"
    )


def fetch_outputs(
    res: list[float] | tuple[float, ...] | str | RunError,
) -> PBSAResult | str | RunError:
    return _unwrap_output(res)


def save_outputs(
    res: list[float] | tuple[float, ...] | str | RunError,
) -> Path | str | RunError:
    output = fetch_outputs(res)
    if isinstance(output, (str, RunError)):
        return output

    output_json = asdict(output)
    return save_json(
        output_json,
        name=_json_content_name("pbsa_output", output_json),
    )
