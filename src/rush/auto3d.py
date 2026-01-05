import sys
from dataclasses import dataclass
from string import Template

from gql.transport.exceptions import TransportQueryError

from rush.client import (
    PROJECT_ID,
    RunOpts,
    RunSpec,
    collect_run,
    submit_rex,
)
from rush.utils import bool_to_str, float_to_str


@dataclass
class Auto3DOptions:
    k: int = 1
    batchsize_atoms: int = 1024
    capacity: int = 40
    convergence_threshold: float = 0.003
    enumerate_isomer: bool = True
    enumerate_tautomer: bool = False
    max_confs: int | None = None
    opt_steps: int = 5000
    patience: int = 1000
    threshold: float = 0.3

    def to_rex(self, reference_fragment: int | None = None):
        return Template(
            """(auto3d_rex::Auto3dOptions {
        k = Some (int $k),
        batchsize_atoms = Some $batchsize_atoms,
        capacity = Some $capacity,
        convergence_threshold = Some $convergence_threshold,
        enumerate_isomer = Some $enumerate_isomer,
        enumerate_tautomer = Some $enumerate_tautomer,
        job_name = None,
        max_confs = $max_confs,
        memory = None,
        mpi_np = Some 4,
        opt_steps = Some $opt_steps,
        optimizing_engine = Some auto3d_rex::Auto3dOptimizingEngines::AIMNET,
        patience = Some $patience,
        threshold = Some $threshold,
        verbose = Some false,
        window = None,
      })"""
        ).substitute(
            k=self.k,
            batchsize_atoms=self.batchsize_atoms,
            capacity=self.capacity,
            convergence_threshold=float_to_str(self.convergence_threshold),
            enumerate_isomer=bool_to_str(self.enumerate_isomer),
            enumerate_tautomer=bool_to_str(self.enumerate_tautomer),
            max_confs=self.max_confs,
            opt_steps=self.opt_steps,
            patience=self.patience,
            threshold=float_to_str(self.threshold),
        )


def auto3d(
    smis: list[str],
    opts: Auto3DOptions = Auto3DOptions(),
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    rex = Template("""let
  auto3d = λ smis →
    try_auto3d_rex
      default_runspec_gpu
      $opts
      $smis
in
  auto3d $smis
""").substitute(
        smis=f"[{', '.join([f'"{smi}"' for smi in smis])}]",
        opts=opts.to_rex(),
        run_spec=run_spec.to_rex(),
    )
    try:
        run_id = submit_rex(PROJECT_ID, rex, run_opts)
        if not collect:
            return run_id

        result = collect_run(run_id)
        # TODO: proper error types
        if isinstance(result, str):
            return result

        def is_result_type(result):
            return (
                isinstance(result, dict)
                and len(result) == 1
                and ("Ok" in result or "Err" in result)
            )

        # TODO: no special cases for Result unwrapping
        return [
            next(iter(r_i.values())) if is_result_type(r_i) else r_i for r_i in result
        ]

    except TransportQueryError as e:
        if e.errors:
            print("Error:", file=sys.stderr)
            for error in e.errors:
                print(f"  {error['message']}", file=sys.stderr)
