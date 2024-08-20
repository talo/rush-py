#!/usr/bin/env python3

import csv
import dataclasses
import json
import math
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from time import sleep

import datargs
import httpx
import prettyprinter
from prettyprinter import pprint

from paratus import ZAG_QDX_AMISS_PATHOGENIC, ZAG_ALIGNED_HUMAN_SEQ

prettyprinter.install_extras(exclude=["ipython", "django"])


def aligned_index(seq_a, i_u):
    """
    inputs
    --------
    seq_a: the aligned sequence (with dashes)
    i_u:   the index into the unaligned sequence

    outputs:
    ----
    aligned indices and the amino acid at the index, respectively
    """
    seq_u = seq_a.replace("-", "")
    walk_a = 0
    walk_u = 0
    while walk_u != i_u or seq_a[walk_a] != seq_u[i_u]:
        if seq_a[walk_a] == "-":
            walk_a += 1
        elif seq_a[walk_a] == seq_u[walk_u]:
            walk_a += 1
            walk_u += 1
    i_a = walk_a
    assert i_u == walk_u
    assert i_u <= i_a
    assert seq_u[i_u] == seq_a[i_a]
    return (i_a, seq_a[i_a])


def unaligned_index(seq_a, i_a):
    return len(seq_a[: i_a + 1].replace("-", ""))


AA_1TO3 = {
    "C": "CYS",
    "D": "ASP",
    "S": "SER",
    "Q": "GLN",
    "K": "LYS",
    "I": "ILE",
    "P": "PRO",
    "T": "THR",
    "F": "PHE",
    "N": "ASN",
    "G": "GLY",
    "H": "HIS",
    "L": "LEU",
    "R": "ARG",
    "W": "TRP",
    "A": "ALA",
    "V": "VAL",
    "E": "GLU",
    "Y": "TYR",
    "M": "MET",
}


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return dataclasses.asdict(o)  # type: ignore
        elif isinstance(o, set):
            return sorted(list(o))
        return super().default(o)


@dataclass
class Mutation:
    old_amino_acid_name: str
    new_amino_acid_name: str
    chain_id: str
    residue_number: int
    amiss_is_snv: bool | None
    amiss_pathogenicity_score: float | None
    amiss_pathogenicity_class: str | None
    ddmut_ddG_kcal_per_mol: float | None
    ddmut_ddG_kcal_per_mol_reverse: float | None

    @classmethod
    def from_amiss_row(cls, row: dict[str, str]):
        return Mutation(
            row["a.a.1"],
            row["a.a.2"],
            "A",
            int(row["position"]),
            row["is_snv"] == "y",
            float(row["pathogenicity score"]),
            row["pathogenicity class"],
            None,
            None,
        )

    @classmethod
    def from_paratus_spreadsheet_row(cls, row: dict[str, str]):
        result = []
        aa1, aa2s = row["AA change"].split("->")
        for aa2 in aa2s.split("/"):
            if aa1 != aa2:
                print(aa1, aa2, int(row["AA position in Ref"]))
                result.append(
                    Mutation(
                        aa1,
                        aa2,
                        "A",
                        int(row["AA position in Ref"]),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                )
        return result


def mutations_from_aligned_variant_json(target, region_of_interest=tuple()) -> dict[str, list[Mutation]]:
    mutations = {}  # key = ResidueIndex (int), val: list[AlphaMissenseMutation]
    # Paratus variants (only have the sequence)
    with open(Path.cwd() / "paratus_data" / "paratus_seqs.json") as f:
        paratus_seqs = json.load(f)

    def find_seq(seqs, species_name):
        for seq in paratus_seqs["seqs"]:
            if species_name in seq["name"]:
                return seq
        return None

    seq = find_seq(paratus_seqs["seqs"], target.split("_")[0])
    if not seq:
        print("Couldn't find species!")
        exit(1)

    raw_name = "".join(c if (c.isalnum() or c in ("-")) else "_" for c in seq["name"])
    raw_seq = seq["seq"].replace("-", "")
    for msa_residue_number, variants in region_of_interest.items():
        residue_number = unaligned_index(seq["seq"], msa_residue_number - 1)
        aa = seq["seq"][msa_residue_number - 1]
        assert aa == raw_seq[residue_number - 1]  # should match
        print(f"{raw_name},{msa_residue_number},{residue_number},{aa}")
        if residue_number not in mutations:
            mutations[residue_number] = []
        for variant in (v for v in variants if v != aa):
            mutations[residue_number].append(
                Mutation(
                    aa,
                    variant,
                    "A",
                    residue_number,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
    return mutations


def mutations_from_amiss(target) -> dict[str, list[Mutation]]:
    mutations = {}
    # UniProt-sourced (full UniProt metadata + AlphaMissense)
    with open(Path.home() / f"Downloads/AlphaMissense/AlphaMissense-Search-{target}.tsv") as fd:
        rd = csv.DictReader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            mut = Mutation.from_amiss_row(row)
            # if mut.residue_number in TARGET_RESIDUES
            # and (
            #     mut.amiss_pathogenicity_score
            #     and (mut.amiss_pathogenicity_score >= 0.75 or mut.amiss_pathogenicity_score <= 0.25)
            # ):
            this_aligned_index, this_aa = aligned_index(ZAG_ALIGNED_HUMAN_SEQ, mut.residue_number - 1)
            if mut.residue_number <= 5:
                pass
            if (
                this_aligned_index + 1 in ZAG_QDX_AMISS_PATHOGENIC
                and mut.new_amino_acid_name in ZAG_QDX_AMISS_PATHOGENIC[this_aligned_index + 1]
            ):
                # print(mut.residue_number)
                # print(mut.old_amino_acid_name, ZAG_QDX_AMISS_PATHOGENIC[this_aligned_index + 1])
                # print(mut.old_amino_acid_name, this_aa)
                assert mut.old_amino_acid_name in ZAG_QDX_AMISS_PATHOGENIC[this_aligned_index + 1]
                assert mut.old_amino_acid_name == this_aa
                if mut.residue_number not in mutations:
                    mutations[mut.residue_number] = []
                mutations[mut.residue_number].append(mut)

    return mutations


def mutations_from_paratus_spreadsheet(name) -> dict[str, list[Mutation]]:
    mutations = {}
    with open(Path.home() / "Downloads" / f"{name}_changed_residues.tsv") as fd:
        rd = csv.DictReader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            muts = Mutation.from_paratus_spreadsheet_row(row)
            for mut in muts:
                if mut.residue_number not in mutations:
                    mutations[mut.residue_number] = []
                mutations[mut.residue_number].append(mut)

    return mutations


@dataclass
class Args:
    name: str
    target: str
    region_of_interest: str | None


def main():
    args = datargs.parse(Args)
    if args.target in (
        "mAetAle1_1-291"
        "mArtInt1_1-290"
        "mArtLit_1-290"
        "mCenSen1_1-293"
        "mCynHor1_1-291"
        "mLopEvo1_1-293"
        "mPlaHel1_1-293"
        "mStuPar1_1-292"
        "mUroCon1_1-293"
        "pub_GRCm39_1-307"
    ):
        mutations = mutations_from_aligned_variant_json(args.target)
    elif args.name in ("GCG", "GLP1R"):
        mutations = mutations_from_paratus_spreadsheet(args.name)
    else:
        mutations = mutations_from_amiss(args.target)

    # Print mutations to submit
    n = sum([len(muts_at_res) for muts_at_res in mutations.values()])
    for ms_for_res in mutations.values():
        for m in ms_for_res:
            print(f"A {m.residue_number:>3} {m.old_amino_acid_name} → {m.new_amino_acid_name}")
    print(f"{n} total mutations to submit.")

    target_path = (
        Path.cwd()
        / "objects"
        / "paratus_druggability_results"
        / "druggability_results"
        / "8_GLP1"
        / "alphafold"
    )

    # Write file for mutation data for DDmut submission
    with open(target_path / f"{args.target}_4.1_mutations_from_paratus.txt", "w") as f:
        for ms_for_res in mutations.values():
            for m in ms_for_res:
                f.write(f"A {m.old_amino_acid_name}{m.residue_number}{m.new_amino_acid_name}\n")

    # Make DDMut submission
    # target_path = Path.cwd() / "objects" / "ZAG_species" / f"{args.target}"
    with (
        open(target_path / f"{args.target}_1.2_alphafold.pdb", "rb") as pdb,
        open(target_path / f"{args.target}_4.1_mutations_from_paratus.txt", "rb") as txt,
    ):
        submission = httpx.post(
            "https://biosig.lab.uq.edu.au/ddmut/api/prediction_list",
            files={"pdb_file": pdb, "mutations_list": txt},
            data={"reverse": "True"},
            timeout=60,
        )
        print(f"sent POST to {submission.url}")
        submission_json = submission.json()
        pprint(submission_json)

    # Get DDmut results
    results_url = "https://biosig.lab.uq.edu.au/ddmut/api/prediction_list"
    results_data = {"job_id": submission_json["job_id"]}
    results = httpx.request("GET", results_url, data=results_data)
    print(f"Sent GET to {results.url}")
    results_json = results.json()
    wait_secs = 1
    while "status" in results_json and results_json["status"] == "RUNNING":
        sleep(wait_secs)
        wait_secs = min(wait_secs * 2, 60)
        results = httpx.request("GET", results_url, data=results_data)
        results_json = results.json()
    pprint(results_json)
    with open(target_path / f"{args.target}_4.2_ddmut_from_paratus.json", "w") as f:
        json.dump(results_json, f)

    def find_ddmut_prediction(ddmut_results, m) -> tuple[float, float]:
        for ddmut_result in ddmut_results.values():
            mutation_str = f"{m.old_amino_acid_name}{m.residue_number}{m.new_amino_acid_name}"
            if ddmut_result["chain"] == "A" and ddmut_result["mutation"] == mutation_str:
                print(ddmut_result["prediction"], ddmut_result["prediction_reverse"])
                return ddmut_result["prediction"], ddmut_result["prediction_reverse"]
        return math.nan, math.nan

    # Find DDMut prediction for each Mutation object
    for ms_for_res in mutations.values():
        for m in ms_for_res:
            m.ddmut_ddG_kcal_per_mol, m.ddmut_ddG_kcal_per_mol_reverse = find_ddmut_prediction(
                results_json, m
            )
            m.old_amino_acid_name = AA_1TO3[m.old_amino_acid_name]
            m.new_amino_acid_name = AA_1TO3[m.new_amino_acid_name]

    # Write final mutation data
    with open(target_path / f"{args.target}_4.3_mutation_results_from_paratus.json", "w") as f:
        json.dump(
            {
                "protein_name": args.name,
                "uniprot_id": args.target,
                "job_id": submission_json["job_id"],
                "mutations": mutations,
            },
            f,
            indent=2,
            cls=EnhancedJSONEncoder,
        )

    with open(target_path / f"{args.name}_{args.target}_mutation_results_from_paratus.csv", "w") as f:
        mutations_list = [m for ms_for_res in mutations.values() for m in ms_for_res]
        writer = csv.writer(f)
        writer.writerow(mutations_list[0].__annotations__.keys())
        for m in mutations_list:
            writer.writerow(m.__dict__.values())


if __name__ == "__main__":
    main()
