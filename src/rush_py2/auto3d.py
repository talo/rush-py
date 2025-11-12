import sys
from dataclasses import dataclass
from string import Template

from gql.transport.exceptions import TransportQueryError

from rush_py2.client import (
    PROJECT_ID,
    RunOpts,
    RunSpec,
    collect_run,
    print_run_trace,
    save_object,
    submit_rex,
)
from rush_py2.utils import bool_to_str, float_to_str


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

        run = collect_run(run_id)
        if run is None:
            print("No run available", file=sys.stderr)
            return None

        result = run["result"]
        if "Ok" in result:
            if "Ok" in result["Ok"]:
                all_smi_confs = []
                for smi_confs_res in run["result"]["Ok"]["Ok"]:
                    if "Ok" in smi_confs_res:
                        all_smi_confs.append(
                            tuple(
                                save_object(smi_conf_vobj["path"], run_id)
                                for smi_conf_vobj in smi_confs_res["Ok"]
                            )
                        )
                    else:
                        all_smi_confs.append(smi_confs_res["Err"])
                return all_smi_confs
            elif "Err" in run["result"]["Ok"]:
                print(f"Error: {run['result']['Ok']['Err']}", file=sys.stderr)
        elif "Err" in run["result"]:
            print(f"Error: {run['result']['Err']}", file=sys.stderr)
        elif run["status"] == "error":
            print_run_trace(run)

        return None

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
