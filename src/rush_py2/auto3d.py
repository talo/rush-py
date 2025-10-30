from string import Template

from gql.transport.exceptions import TransportQueryError

from rush_py2.client import PROJECT_ID, RunSpec, print_run_trace, submit_rex


def auto3d(
    smis: list[str],
    run_spec: RunSpec = RunSpec(),
):
    rex = Template("""let
  results = 
    try_auto3d_rex
      default_runspec_gpu
      (auto3d_rex::Auto3dOptions {
        k = Some (int 1),
        batchsize_atoms = Some 1024,
        capacity = Some 40,
        convergence_threshold = Some 0.003,
        enumerate_isomer = Some true,
        enumerate_tautomer = Some false,
        job_name = Some "",
        max_confs = None,
        memory = None,
        mpi_np = Some 4,
        opt_steps = Some 5000,
        optimizing_engine = Some auto3d_rex::Auto3dOptimizingEngines::AIMNET,
        patience = Some 1000,
        threshold = Some 0.3,
        verbose = Some false,
        window = None,
      })
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
        run_spec=run_spec.to_rex(),
    )
    try:
        return submit_rex(PROJECT_ID, rex)

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")