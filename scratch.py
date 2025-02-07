from rush import build_blocking_provider

# |hide
# hidden setup for the notebook
import os
import pathlib

WORK_DIR = pathlib.Path("~/qdx/benchmark_notebook").expanduser()
if WORK_DIR.exists():
    os.system(f"rm -r {WORK_DIR}")
os.makedirs(WORK_DIR, exist_ok=True)

# swap into clean workdir so that our tests are deterministic
os.chdir(WORK_DIR)
PUT_YOUR_PREFERRED_WORKING_DIRECTORY_HERE = WORK_DIR
PUT_YOUR_TOKEN_HERE = os.environ["RUSH_TOKEN"]
RUSH_URL = os.environ["RUSH_URL"]
os.environ["RUSH_RESTORE_BY_DEFAULT"] = "False"

client = build_blocking_provider(
    access_token=PUT_YOUR_TOKEN_HERE,
    url=RUSH_URL,
)

rex_code = """
let
    rxdock_options = {
            n_runs = 1,
            radius = 8.0,
            min_volumn = none,
            small_sphere = none
        },

    rxdock = λ protein_conformer_trc small_molecule_conformer_tr ->
            rxdock_rex_s default_runspec rxdock_options protein_conformer_trc small_molecule_conformer_tr none,
    
    prot_structure = load ("3cd65ef4-c8c3-4864-abaf-7cc8b4e02309") 'Structure',
    trc = [
        topology prot_structure,
        residues prot_structure,
        chains prot_structure
    ],

    smol_structure = load ("90d5040b-557b-41b4-8df1-2d27fbf5f5c4" ) 'Structure',
    tr = [
        topology smol_structure,
        residues smol_structure
    ],

    docked_result = rxdock trc [tr],

    min_affinity =  get "score" ( get 2 ( get 0 (get "Ok" (get 0 docked_result)))),

in 
    min_affinity
"""

submission = client.eval_rex_blocking(rex_code, "scratch_rxdock")
