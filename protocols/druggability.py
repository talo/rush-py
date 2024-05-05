import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path

import datargs

import rush

# Parameters

WORK_DIR = Path.cwd()

INPUT_FASTA_FILENAME = "input.fasta"

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
    },
    "target": "GADI",
}


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


async def main():
    args = datargs.parse(Args)

    client = await rush.build_provider_with_functions(
        batch_tags=["paratus", "demo"],
        workspace=WORK_DIR,
    )

    # 1.0: Obtain input (FASTA)
    fasta_sequence = next(iter(read_fasta(args.input_sequence).values()))

    # 1.1: MSA
    (msa_handle,) = await client.mmseqs2(
        {"fasta": [fasta_sequence]},
        restore=True,
        **COLABFOLD_SEARCH_RUN_CONFIG,
    )

    # 1.2: Produce 3D structure
    (folded_conformers_handle,) = await client.colabfold_fold(
        msa_handle,
        **COLABFOLD_FOLD_RUN_CONFIG,
    )
    await folded_conformers_handle.download()

    # 2.0: Evaluate pockets
    p2rank_prediction_handle, pml_viz_handle = await client.p2rank(
        folded_conformers_handle,
        use_alpha_config=True,
        **COLABFOLD_FOLD_RUN_CONFIG,
    )

    # 2.1: Show evaluation
    p2rank_prediction = await p2rank_prediction_handle.get()
    for pocket in p2rank_prediction["pockets"]:
        print(pocket)
    viz_path = await pml_viz_handle.download()
    subprocess.run(["tar", "xzvf", viz_path, "-C", client.workspace], check=True)


@dataclass
class Args:
    input_sequence: Path


if __name__ == "__main__":
    asyncio.run(main())
