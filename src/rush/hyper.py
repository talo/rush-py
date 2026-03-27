"""
Hyper module for the Rush Python client.

Provides molecular dynamics and solvation functionality via Hyper.
"""

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, TypeGuard, TypeVar, Literal

from gql.transport.exceptions import TransportQueryError

from rush import TRC
from rush.convert import from_json
from ._utils import bool_to_str, float_to_str

from .client import (
    RunOpts,
    RunSpec,
    RushObject,
    _get_project_id,
    _submit_rex,
    upload_object,
    fetch_object,
)
from .run import RushRun

Error = str
T = TypeVar("T")

def _is_result_type(result: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(result, dict) and len(result) == 1 and ("Ok" in result or "Err" in result)

def _map_outputs(
    res: list[Any],
    *,
    on_success: Any,
) -> list[Any | Error]:
    return [
        Error(res_i) if isinstance(res_i, str) else on_success(res_i)
        for res_i in res
    ]

def _to_rush_object(item: TRC | str | Path | RushObject | dict) -> RushObject:
    if isinstance(item, RushObject):
        return item
    if isinstance(item, (str, Path)):
        return RushObject.from_dict(upload_object(item))
    if isinstance(item, TRC):
        return RushObject.from_dict(upload_object({
            "topology": item.topology.to_dict(),
            "residues": item.residues.to_dict(),
            "chains": item.chains.to_dict(),
        }))
    if isinstance(item, dict):
        if "path" in item and "size" in item and "format" in item:
            return RushObject.from_dict(item)
        return RushObject.from_dict(upload_object(item))
    raise TypeError(f"Cannot convert {type(item)} to RushObject")

def _obj_to_rex(obj: RushObject) -> str:
    fmt = obj.format.lower()
    return f"(VirtualObject {{ path = \"{obj.path}\", size = {obj.size}, format = ObjectFormat::{fmt} }})"

@dataclass(frozen=True)
class _TRCObjectRef:
    obj: RushObject

    def fetch(self) -> TRC:
        content = fetch_object(self.obj.path).decode('utf-8')
        return from_json(json.loads(content))

# ---------------------------------------------------------------------------
# Solvate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolvateResultRef:
    _inputs: list[_TRCObjectRef | Error]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "SolvateResultRef":
        if not isinstance(raw, list):
            raise ValueError(f"hyper.solvate should return a list, got {type(raw).__name__}.")
        
        unwrapped = [next(iter(item.values())) if _is_result_type(item) else item for item in raw]
        
        def parse_trc(res_i: Any) -> _TRCObjectRef:
            return _TRCObjectRef(RushObject.from_dict(res_i))
            
        parsed = _map_outputs(unwrapped, on_success=parse_trc)
        return cls(_inputs=parsed)

    def fetch(self) -> list[TRC | Error]:
        return _map_outputs(self._inputs, on_success=lambda ref: ref.fetch())

def solvate(
    input_trcs: list[TRC | str | Path | RushObject | dict],
    max_inputs: int | None = None,
    padding_nm: float | None = None,
    seed: int | None = None,
    timeout_seconds: int | None = None,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[SolvateResultRef]:
    rex_inputs = []
    for t in input_trcs:
        o = _to_rush_object(t)
        rex_inputs.append(_obj_to_rex(o))
    rex_inputs_str = f"[{', '.join(rex_inputs)}]"

    rex = Template("""let
  input_trcs = $inputs
in
  try_hyper_solvate_sumo
    ($run_spec)
    (Some (hyper_solvate_sumo::HyperConfig {
      max_inputs = $max_inputs,
      padding_nm = $padding_nm,
      seed = $seed,
      timeout_seconds = $timeout_seconds
    }))
    input_trcs
""").substitute(
        inputs=rex_inputs_str,
        max_inputs=f"(Some {max_inputs})" if max_inputs is not None else "None",
        padding_nm=f"(Some {float_to_str(padding_nm)})" if padding_nm is not None else "None",
        seed=f"(Some {seed})" if seed is not None else "None",
        timeout_seconds=f"(Some {timeout_seconds})" if timeout_seconds is not None else "None",
        run_spec=run_spec._to_rex(),
    )
    try:
        return RushRun(
            _submit_rex(_get_project_id(), rex, run_opts),
            SolvateResultRef,
        )
    except TransportQueryError as e:
        if e.errors:
            print("Error:", file=sys.stderr)
            for error in e.errors:
                print(f"  {error['message']}", file=sys.stderr)
        raise

# ---------------------------------------------------------------------------
# Minimize
# ---------------------------------------------------------------------------

@dataclass
class MinimizeInput:
    structure: TRC | str | Path | RushObject | dict
    topology_bin: str | Path | RushObject | dict

@dataclass(frozen=True)
class MinimizeResultRef:
    _inputs: list[_TRCObjectRef | Error]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "MinimizeResultRef":
        if not isinstance(raw, list):
            raise ValueError(f"hyper.minimize should return a list, got {type(raw).__name__}.")
        
        unwrapped = [next(iter(item.values())) if _is_result_type(item) else item for item in raw]
        
        def parse_trc(res_i: Any) -> _TRCObjectRef:
            return _TRCObjectRef(RushObject.from_dict(res_i))
            
        parsed = _map_outputs(unwrapped, on_success=parse_trc)
        return cls(_inputs=parsed)

    def fetch(self) -> list[TRC | Error]:
        return _map_outputs(self._inputs, on_success=lambda ref: ref.fetch())

def minimize(
    jobs: list[MinimizeInput],
    max_inputs: int | None = None,
    steps: int | None = None,
    gtol: float | None = None,
    timeout_seconds: int | None = None,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[MinimizeResultRef]:
    rex_jobs = []
    for job in jobs:
        struct_obj = _to_rush_object(job.structure)
        topo_obj = _to_rush_object(job.topology_bin)
        rex_jobs.append(f"""(hyper_minimize_sumo::MinimizeInput {{
          structure = {_obj_to_rex(struct_obj)},
          topology_bin = {_obj_to_rex(topo_obj)}
        }})""")
    jobs_str = f"[{', '.join(rex_jobs)}]"

    rex = Template("""let
  jobs = $jobs
in
  try_hyper_minimize_sumo
    ($run_spec)
    (Some (hyper_minimize_sumo::HyperMinimizeConfig {
      max_inputs = $max_inputs,
      steps = $steps,
      gtol = $gtol,
      timeout_seconds = $timeout_seconds
    }))
    jobs
""").substitute(
        jobs=jobs_str,
        max_inputs=f"(Some {max_inputs})" if max_inputs is not None else "None",
        steps=f"(Some {steps})" if steps is not None else "None",
        gtol=f"(Some {float_to_str(gtol)})" if gtol is not None else "None",
        timeout_seconds=f"(Some {timeout_seconds})" if timeout_seconds is not None else "None",
        run_spec=run_spec._to_rex(),
    )
    try:
        return RushRun(
            _submit_rex(_get_project_id(), rex, run_opts),
            MinimizeResultRef,
        )
    except TransportQueryError as e:
        if e.errors:
            print("Error:", file=sys.stderr)
            for error in e.errors:
                print(f"  {error['message']}", file=sys.stderr)
        raise

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@dataclass
class RunInput:
    sim_config_json: str | Path | RushObject | dict
    topology_bin: str | Path | RushObject | dict
    coordinates: TRC | str | Path | RushObject | dict

@dataclass(frozen=True)
class _RunOutputRef:
    trajectory: RushObject
    checkpoint: RushObject | None

@dataclass
class RunResult:
    trajectory: bytes
    checkpoint: bytes | None

@dataclass(frozen=True)
class RunResultRef:
    _inputs: list[_RunOutputRef | Error]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "RunResultRef":
        if not isinstance(raw, list):
            raise ValueError(f"hyper.run should return a list, got {type(raw).__name__}.")
        
        unwrapped = [next(iter(item.values())) if _is_result_type(item) else item for item in raw]
        
        def parse_run_out(res_i: Any) -> _RunOutputRef:
            traj = RushObject.from_dict(res_i["trajectory"])
            chk = RushObject.from_dict(res_i["checkpoint"]) if res_i.get("checkpoint") else None
            return _RunOutputRef(trajectory=traj, checkpoint=chk)
            
        parsed = _map_outputs(unwrapped, on_success=parse_run_out)
        return cls(_inputs=parsed)

    def fetch(self) -> list[RunResult | Error]:
        def fetch_output(ref: _RunOutputRef) -> RunResult:
            traj = fetch_object(ref.trajectory.path)
            chk = fetch_object(ref.checkpoint.path) if ref.checkpoint else None
            return RunResult(trajectory=traj, checkpoint=chk)
        return _map_outputs(self._inputs, on_success=fetch_output)

def run(
    jobs: list[RunInput],
    max_inputs: int | None = None,
    nsteps: int | None = None,
    dt_ps: float | None = None,
    temperature_k: float | None = None,
    ensemble: Literal["Nve", "Nvt", "Npt"] | None = None,
    minimize_before_run: bool | None = None,
    solvate_before_run: bool | None = None,
    use_gpu: bool | None = None,
    nthreads: int | None = None,
    timeout_seconds: int | None = None,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[RunResultRef]:
    rex_jobs = []
    for job in jobs:
        sim_obj = _to_rush_object(job.sim_config_json)
        topo_obj = _to_rush_object(job.topology_bin)
        coords_obj = _to_rush_object(job.coordinates)
        rex_jobs.append(f"""(hyper_run_sumo::RunInput {{
          sim_config_json = {_obj_to_rex(sim_obj)},
          topology_bin = {_obj_to_rex(topo_obj)},
          coordinates = {_obj_to_rex(coords_obj)}
        }})""")
    jobs_str = f"[{', '.join(rex_jobs)}]"

    rex = Template("""let
  jobs = $jobs
in
  try_hyper_run_sumo
    ($run_spec)
    (Some (hyper_run_sumo::HyperRunConfig {
      max_inputs = $max_inputs,
      nsteps = $nsteps,
      dt_ps = $dt_ps,
      temperature_k = $temperature_k,
      ensemble = $ensemble,
      minimize_before_run = $minimize_before_run,
      solvate_before_run = $solvate_before_run,
      use_gpu = $use_gpu,
      nthreads = $nthreads,
      timeout_seconds = $timeout_seconds
    }))
    jobs
""").substitute(
        jobs=jobs_str,
        max_inputs=f"(Some {max_inputs})" if max_inputs is not None else "None",
        nsteps=f"(Some {nsteps})" if nsteps is not None else "None",
        dt_ps=f"(Some {float_to_str(dt_ps)})" if dt_ps is not None else "None",
        temperature_k=f"(Some {float_to_str(temperature_k)})" if temperature_k is not None else "None",
        ensemble=f"(Some hyper_run_sumo::Ensemble::{ensemble.capitalize()})" if ensemble is not None else "None",
        minimize_before_run=f"(Some {bool_to_str(minimize_before_run)})" if minimize_before_run is not None else "None",
        solvate_before_run=f"(Some {bool_to_str(solvate_before_run)})" if solvate_before_run is not None else "None",
        use_gpu=f"(Some {bool_to_str(use_gpu)})" if use_gpu is not None else "None",
        nthreads=f"(Some {nthreads})" if nthreads is not None else "None",
        timeout_seconds=f"(Some {timeout_seconds})" if timeout_seconds is not None else "None",
        run_spec=run_spec._to_rex(),
    )
    try:
        return RushRun(
            _submit_rex(_get_project_id(), rex, run_opts),
            RunResultRef,
        )
    except TransportQueryError as e:
        if e.errors:
            print("Error:", file=sys.stderr)
            for error in e.errors:
                print(f"  {error['message']}", file=sys.stderr)
        raise

