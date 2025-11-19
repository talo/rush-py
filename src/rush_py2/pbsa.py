#!/usr/bin/env python3
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template

import cyclopts
from gql.transport.exceptions import TransportQueryError

from .client import (
    Client,
    RunOpts,
    RunSpec,
    print_run_trace,
)
from .utils import float_to_str


@dataclass
class PBSAResults:
    solvation_energy: float
    polar_solvation_energy: float
    nonpolar_solvation_energy: float


def pbsa(
    client: Client,
    project_id: str,
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
    topology_vobj = client.upload_object(project_id, topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology →
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
  exess "$topology_vobj_path"
""").substitute(
        run_spec=run_spec.to_rex(),
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
        run_id = client.submit_rex(project_id, rex, run_opts)
        if collect:
            run = client.collect_run(run_id)
            if run["status"] == "done":
                return PBSAResults(*run["result"])
            elif run["status"] == "error":
                print_run_trace(run)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def run_pbsa():
    cyclopts.run(pbsa)
