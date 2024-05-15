import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import datargs
import httpx

import rush


@dataclass
class CrossRef:
    database: str
    id: str
    method: str
    resolution: str
    chains: str


@dataclass
class UniprotData:
    entry_type: str
    primary_accession: str
    sequence: str
    molecular_weight: int
    cross_refs: list[CrossRef]


# Parameters

WORK_DIR = Path.cwd()

COLABFOLD_SEARCH_RUN_CONFIG = {
    "resources": {
        "gpus": 0,
        "cpus": 48,
        "mem": 128 * 1024,
        "storage_mounts": ["gdata/if89"],
        "walltime": 360,
    },
    "target": "GADI",
}

COLABFOLD_FOLD_RUN_CONFIG = {
    "resources": {
        "gpus": 1,
        "storage_mounts": ["gdata/if89"],
        "walltime": 360,
    },
    "target": "GADI",
}

PREPARE_PROTEIN_RUN_CONFIG = {}

P2RANK_RUN_CONFIG = {}


def get_uniprot(target: str):
    params = {
        "query": f"organism_name:Human AND reviewed:true AND gene:{target}",
        "fields": "xref_pdb,sequence",
        "format": "json",
    }
    response = httpx.get("https://rest.uniprot.org/uniprotkb/stream", params=params)
    response_json = response.json()
    uniprot_data = []
    for result in response_json["results"]:
        cross_refs = []
        for cross_ref_data in result["uniProtKBCrossReferences"]:
            method = ""
            resolution = ""
            chains = ""
            for property in cross_ref_data["properties"]:
                if property["key"] == "Method":
                    method += property["value"]
                if property["key"] == "Resolution":
                    resolution += property["value"]
                if property["key"] == "Chains":
                    chains += property["value"]
            cross_refs.append(
                CrossRef(cross_ref_data["database"], cross_ref_data["id"], method, resolution, chains)
            )
        uniprot_data.append(
            UniprotData(
                result["entryType"],
                result["primaryAccession"],
                result["sequence"]["value"],
                result["sequence"]["molWeight"],
                cross_refs,
            )
        )
    assert len(uniprot_data) == 1
    return uniprot_data[0]


def get_pdb(id: str):
    print(f"https://files.rcsb.org/download/{id}.pdb.gz")
    response = httpx.get(f"https://files.rcsb.org/download/{id}.pdb.gz", timeout=30.0)
    return str(gzip.decompress(response.content), "utf-8")


def read_fasta(filepath: Path) -> dict[str, str]:
    """Read a FASTA file. See:
    https://blast.ncbi.nlm.nih.gov/doc/blast-topics/queryinpanddatasel.html#accepted-input-formats
    """
    fasta_sequences = {}
    current_key = ""
    current_seq: list[str] = []
    with open(filepath) as f:
        for line in f:
            if line.startswith(">"):
                if len(current_seq) != 0 and current_key not in fasta_sequences:
                    fasta_sequences[current_key] = "".join(current_seq)
                current_key = line[1:].rstrip()
                current_seq = []
            elif line.startswith(";"):
                # ignore comments, if any are present
                continue
            else:
                segments = line.split()
                # the first space-separated segment can be numeric
                if segments[0].isdigit():
                    segments = segments[0]
                # the rest need to be a valid FASTA nucleic acid or amino acid code
                subseq = "".join(segments)
                if set(subseq).issubset("ABCDEFGHIKLMNPQRSTUVWYZX*-"):
                    current_seq.append(subseq)
                else:
                    pass

        if len(current_seq) != 0 and current_key not in fasta_sequences:
            fasta_sequences[current_key] = "".join(current_seq)

        return fasta_sequences


@dataclass
class Args:
    target: str


def main():
    args = datargs.parse(Args)

    # 1.0: Obtain input (FASTA)
    uniprot_data = get_uniprot(args.target)
    uniprot_id = uniprot_data.primary_accession
    print(f"{uniprot_id=}")
    print(f"{uniprot_data.sequence=}")

    client = rush.build_blocking_provider_with_functions(
        batch_tags=["paratus", "protocol", "druggability", uniprot_data.primary_accession],
        workspace=WORK_DIR,
        restore_by_default=True,
    )

    # 1.1: MSA
    (msa_handle,) = client.mmseqs2(
        {"fasta": [uniprot_data.sequence]},
        **COLABFOLD_SEARCH_RUN_CONFIG,
    )

    # 1.2: Produce 3D structure
    (folded_conformers_handle,) = client.colabfold_fold(
        msa_handle,
        **COLABFOLD_FOLD_RUN_CONFIG,
    )
    (folded_conformer_handle,) = client.pick_conformer(folded_conformers_handle, 0)

    # 2.0: Evaluate pockets
    (p2rank_prediction_handle, pymol_viz_handle) = client.p2rank(
        folded_conformer_handle,
        True,
        **P2RANK_RUN_CONFIG,
    )

    # 2.1: Download everything & show evaluation
    msa_handle.download(
        filename=f"{uniprot_id}_1.1_msa.tar.gz",
        overwrite=True,
    )
    folded_conformer_handle.download(
        filename=f"{uniprot_id}_1.2_conformer.json",
        overwrite=True,
    )
    with open(WORK_DIR / "objects" / f"{uniprot_id}_2.0_pocket_data.json", "w") as f:
        json.dump(p2rank_prediction_handle.get(), f, indent=2)
    pymol_viz_handle.download(
        filename=f"{uniprot_id}_2.0_pocket_viz.tar.gz",
        overwrite=True,
    )

    for cross_ref in uniprot_data.cross_refs:
        # 1.2: Get 3D structure from cross-referenced data
        pdb_path = WORK_DIR / "objects" / f"{uniprot_id}-{cross_ref.id}_1.2_structure.pdb"
        if not pdb_path.is_file():
            pdb_path.write_text(get_pdb(cross_ref.id))

        # 1.3: Clean up structure
        (prepared_structures_handle, _) = client.prepare_protein(
            pdb_path,
            None,
            None,
            tags=[cross_ref.id],
            **PREPARE_PROTEIN_RUN_CONFIG,
        )
        (prepared_structure_handle,) = client.pick_conformer(
            prepared_structures_handle, 0, tags=[cross_ref.id]
        )

        # 2.0: Evaluate pockets
        (p2rank_prediction_handle, pymol_viz_handle) = client.p2rank(
            prepared_structure_handle,
            True,
            tags=[cross_ref.id],
            **P2RANK_RUN_CONFIG,
        )

        # 2.1: Download everything & show evaluation
        prepared_structure_handle.download(
            filename=f"{uniprot_id}-{cross_ref.id}_1.3_prepared.json",
            overwrite=True,
        )
        with open(WORK_DIR / "objects" / f"{uniprot_id}-{cross_ref.id}_2.0_pocket_data.json", "w") as f:
            json.dump(p2rank_prediction_handle.get(), f, indent=2)
        pymol_viz_handle.download(
            filename=f"{uniprot_id}-{cross_ref.id}_2.0_pocket_viz.tar.gz",
            overwrite=True,
        )


if __name__ == "__main__":
    main()
