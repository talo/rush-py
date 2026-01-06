# Hardware

Rush provides both cloud computing and supercomputing infrastructure for you to perform runs on. When you submit a job on Rush, it remotely runs the job on either our cloud computing infrastructure (three "Bullet" clusters) or one of two supercomputers, Gadi or Setonix. For teams that prefer to "bring their own compute," Rush supports integration with existing academic or corporate HPC setups. Email hello@qdx.co to discuss this with us.

## Run Specification: Target and Resources

You can pass a set of run specifications to each module function via `run_spec=rush.client.RunSpec(...)`. The target ("Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"), walltime (in minutes), storage (in MB, though storage units are configurable as well), cpus, gpus, and nodes are all configurable via this parameter and class.

If the target is left unspecified, one of the three Bullet clusters will be chosen. 

By default, modules have default resources that will be allocated to them. If supplying custom resources, be sure to provide compatible resources for the selected module: for example. if a module requires a GPU but isn't allocated one, the run will fail. 
