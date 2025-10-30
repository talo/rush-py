from dataclasses import dataclass
from string import Template

from gql.transport.exceptions import TransportQueryError

from rush_py2.client import PROJECT_ID, RunSpec, print_run_trace, submit_rex
from rush_py2.utils import bool_to_str

@dataclass
class Auto3DOptions:
    k = 1
    batchsize_atoms = 1024
    capacity = 40
    convergence_threshold = 0.003
    enumerate_isomer = True
    enumerate_tautomer = False
    job_name = ""
    max_confs = None
    memory = None
    mpi_np = 4
    opt_steps = 5000
    patience = 1000
    threshold = 0.3

    def to_rex(self, reference_fragment: int | None = None):
        return Template(
            """auto3d_rex::Auto3dOptions {
  k = Some (int $k),
  batchsize_atoms = Some $batchsize_atoms,
  capacity = Some $capacity,
  convergence_threshold = Some $convergence_threshold,
  enumerate_isomer = Some $enumerate_isomer,
  enumerate_tautomer = Some $enumerate_tautomer,
  job_name = Some "$job_name",
  max_confs = $max_confs,
  memory = $memory,
  mpi_np = Some $mpi_np,
  opt_steps = Some $opt_steps,
  optimizing_engine = Some auto3d_rex::Auto3dOptimizingEngines::AIMNET,
  patience = Some $patience,
  threshold = Some $threshold,
  verbose = Some false,
  window = None,
}"""
        ).substitute(
            k=self.k,
            batchsize_atoms=self.batchsize_atoms,
            capacity=self.capacity,
            convergence_threshold=self.convergence_threshold,
            enumerate_isomer=bool_to_str(self.enumerate_isomer),
            enumerate_tautomer=bool_to_str(self.enumerate_tautomer),
            job_name=self.job_name,
            max_confs=self.max_confs,
            memory=self.memory,
            mpi_np=self.mpi_np,
            opt_steps=self.opt_steps,
            patience=self.patience,
            threshold=self.threshold,
        )

def auto3d(
    smis: list[str],
    opts: Auto3DOptions = Auto3DOptions(),
    run_spec: RunSpec = RunSpec(),
):
    rex = Template("""let
  results = 
    try_auto3d_rex
      default_runspec_gpu
      ($opts)
      $smis
in
  map 
    (λ x ->
      let
        topology = elem0 (elem0 x), 
        smi = elem1 x
      in
        (smi, topology)
    )
    (zip (map unwrap (unwrap (unwrap results))) $smis)""").substitute(
        smis= f"[{', '.join([f'\"{smi}\"' for smi in smis])}]",
        opts=opts.to_rex(),
        run_spec=run_spec.to_rex(),
    )
    try:
        run = submit_rex(PROJECT_ID, rex)
        if run is not None:
            if run["status"] == "error":
                print_run_trace(run)
            return run
        else:
            print("No run available")
            return None

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")