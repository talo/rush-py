import json
from dataclasses import dataclass
from pathlib import Path

import datargs

import rush

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

P2RANK_RUN_CONFIG = {}


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
    input_sequence: Path


def main():
    args = datargs.parse(Args)

    # 1.0: Obtain input (FASTA)
    fasta_name, fasta_sequence = next(iter(read_fasta(args.input_sequence).items()))
    print(f"{fasta_name=}")
    print(f"{fasta_sequence=}")
    uniprot_id = fasta_name.split("|")[1]
    print(f"{uniprot_id=}")

    client = rush.build_blocking_provider_with_functions(
        batch_tags=["paratus", "protocol", "druggability", uniprot_id],
        workspace=WORK_DIR,
        restore_by_default=True,
    )

    # 1.1: MSA
    (msa_handle,) = client.mmseqs2(
        {"fasta": [fasta_sequence]},
        **COLABFOLD_SEARCH_RUN_CONFIG,
    )

    # 1.2: Produce 3D structure
    (folded_conformers_handle,) = client.colabfold_fold(
        msa_handle,
        **COLABFOLD_FOLD_RUN_CONFIG,
    )

    # 2.0: Evaluate pockets
    (folded_conformer_handle,) = client.pick_conformer(folded_conformers_handle, 0)
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


if __name__ == "__main__":
    main()
