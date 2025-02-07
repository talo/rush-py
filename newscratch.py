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
(\\runspec -> \\options -> \\conformer_ids ->
  let
    conformers = (map (\\id ->
    let
      conformer = load id 'ProteinConformer',
      structure = load (structure_id conformer) 'Structure'
    in
      [ (topology structure), (residues structure), (chains structure) ]
    ) conformer_ids),
    structure_ids = map (\\conformer_id -> structure_id (load conformer_id 'ProteinConformer')) conformer_ids,

    prepare_protein = (prepare_protein_rex runspec options conformers),

    save_structure = (\\topology -> \\residues -> \\chains -> \\ids_names -> 
    let
      structure_name = get 1 (get 1 ids_names),
      pdb_id = get 1 (get 0 ids_names),
      pdb_tag_array = if len pdb_id > 0 then [ pdb_id ] else [],
      structure_id = get 1 (get 0 (get 0 ids_names))
    in
      save (Structure {
        metadata = (Metadata { name = structure_name, description = some "Created by prepare_protein", tags = [ "prepared", "root_id_" ++ structure_id ] ++ pdb_tag_array ++ [ "root" ] }),
        rcsb_id = none,
        topology = VirtualObject { path = get 'path' topology, size = get 'size' topology, format = 'json' },
        residues = VirtualObject { path = get 'path' residues, size = get 'size' residues, format = 'json' },
        chains = VirtualObject { path = get 'path' chains, size = get 'size' chains, format = 'json' }
      })),

    save_protein_conformer = (\\structure_id -> \\protein_id -> \\chains -> \\ids_names -> 
    let
      conformer_name = get 0 (get 1 ids_names),
      pdb_id = get 1 (get 0 ids_names),
      pdb_tag_array = if len pdb_id > 0 then [ pdb_id ] else [],
      root_structure_id = get 1 (get 0 (get 0 ids_names))
    in
      save (ProteinConformer {
        metadata = (Metadata { name = conformer_name, description = some "conformer from prepare_protein", tags = [ "prepared" ] ++ pdb_tag_array ++ [ "root" ] }),
        pdb_id = some pdb_id,
        protein_id = protein_id,
        structure_id = structure_id,
        residues = (map int (get 0 (get "chains" chains)))
      }))
in
  ( map ( \\x -> let result = get 0 x in
    (save_protein_conformer
      (save_structure (get 0 result) (get 1 result) (get 2 result) (get 1 x))
      (protein_id (load (get 0 (get 0 (get 0 (get 1 x)))) "ProteinConformer"))
      (download (VirtualObject { path = (get 'path' (get 2 result)), size = 0, format = 'json' }))
      (get 1 x)
    )
  )
  (zip
    ( get "Ok" ( get 0 ( await (get 1 prepare_protein) )))
    ( zip 
      ( zip 
        (zip conformer_ids structure_ids)
        [ '8FLL' ]
      )
      ( zip 
        ( map (\\conformer_id -> ( name ( metadata (load conformer_id 'ProteinConformer') ) ) ) conformer_ids )
        ( map (\\conformer_id -> ( name ( metadata ( load ( structure_id (load conformer_id 'ProteinConformer') ) 'Structure' ) ) ) ) conformer_ids )
      )
    )
  )
)
)
  (RunSpec { target = "Bullet", resources = Resources { storage = some 10, storage_units = some "MB", gpus = some 1 } }) 
  ({
    truncation_threshold = some 2,
    capping_style = some "Never",
    naming_scheme = none,
    ph = some 7.4
  }) [ "6a414612-2e4f-4cd9-99ba-85183bff1619" ]
"""


submission = client.eval_rex_blocking(rex_code, "some prepare_protein")
