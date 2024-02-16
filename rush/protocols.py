from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .graphql_client.enums import MemUnits, ModuleInstanceTarget
from .graphql_client.input_types import ModuleInstanceResourcesInput
from .provider import Provider

frag_keywords = {
    "dimer_cutoff": 25,
    "dimer_mp2_cutoff": 25,
    "fragmentation_level": 2,
    "method": "MBE",
    "monomer_cutoff": 30,
    "monomer_mp2_cutoff": 30,
    "ngpus_per_node": 1,
    "reference_fragment": 1,
    "trimer_cutoff": 10,
    "trimer_mp2_cutoff": 10,
    "fragmented_energy_type": "InteractivityEnergy",
}

scf_keywords = {
    "convergence_metric": "diis",
    "dynamic_screening_threshold_exp": 10,
    "ndiis": 8,
    "niter": 40,
    "scf_conv": 0.000001,
}

default_model = {"method": "RIMP2", "basis": "cc-pVDZ", "aux_basis": "cc-pVDZ-RIFIT", "frag_enabled": True}


async def run_qp(
    provider: Provider,
    qp_gen_inputs_path: str,
    hermes_energy_path: str,
    qp_collate_path: str,
    pdb: Provider.Arg[Path] | Path,
    gro: Provider.Arg[Path] | Path,
    lig: Provider.Arg[Path] | Path,
    lig_type: Provider.Arg[Literal["sdf", "mol2"]] | Literal["sdf", "mol2"],
    lig_res_id: Provider.Arg[str] | str,
    model: Provider.Arg[dict[str, Any]]
    | dict[str, Any] = Provider.Arg(provider=None, id=None, value=default_model),
    keywords: Provider.Arg[dict[str, Any]] = Provider.Arg(
        None, None, {"frag": frag_keywords, "scf": scf_keywords, "debug": {}, "export": {}, "guess": {}}
    ),
    amino_acids_of_interest: Provider.Arg[list[tuple[str, int]]] = Provider.Arg(None, None, None),
    use_new_fragmentation_method: Provider.Arg[bool] | bool | None = None,
    hermes_target: ModuleInstanceTarget | None = None,
    hermes_resources: ModuleInstanceResourcesInput | None = None,
    tags: list[str] | None = None,
    restore: bool | None = False,
) -> tuple[Provider.Arg[Any], Provider.Arg[Any]]:
    """
    Construct the input and output module instance calls for QP run.
    :param qp_gen_inputs_path: The path of the QP gen inputs module.
    :param hermes_energy_path: The path of the Hermes energy module.
    :param qp_collate_path: The path of the QP collate module.
    :param pdb: The PDB file containing both the protein and ligand.
    :param gro: The GRO file containing ligand.
    :param lig: The ligand file.
    :param lig_type: The type of ligand file.
    :param lig_res_id: The residue ID of the ligand.
    :param model: The model to use for the QP Hermes run.
    :param keywords: The keywords to use for the QP Hermes run.
    :param amino_acids_of_interest: The amino acids of interest to restrict the run to.
    :param target: The target to run the module on.
    :param resources: The resources to run the module with.
    :param autopoll: The autopoll interval and timeout.
    :param tag: The tags to apply to all module instances, arguements and outs.

    :return: The hermes energy output argument.
    :return: The qp collate output argument.
    """

    if hermes_resources is not None and hermes_resources.gpus and keywords.value is not None:
        keywords.value["frag"]["ngpus_per_node"] = hermes_resources.gpus

    qp_prep_conf = {
        "ligand_file_type": lig_type if isinstance(lig_type, str) else lig_type.value,
        "ligand_res_id": lig_res_id if isinstance(lig_res_id, str) else lig_res_id.value,
        "amino_acids_of_interest": (
            amino_acids_of_interest
            if isinstance(amino_acids_of_interest, list)
            else amino_acids_of_interest.value
        ),
        "use_new_fragmentation_method": (
            use_new_fragmentation_method if isinstance(use_new_fragmentation_method, bool) else None
        ),
    }

    if use_new_fragmentation_method is not None:
        qp_prep_conf["use_new_fragmentation_method"] = (
            use_new_fragmentation_method.value
            if isinstance(use_new_fragmentation_method, Provider.Arg)
            else use_new_fragmentation_method
        )

    qp_prep_instance = await provider.run(
        qp_gen_inputs_path,
        [pdb, gro, lig, model, keywords, qp_prep_conf],
        tags=tags,
        out_tags=([tags, tags, tags, tags] if tags else None),
        restore=restore,
        resources=ModuleInstanceResourcesInput(storage=20, storage_units=MemUnits.MB),
    )
    print("launched qp_prep_instance", qp_prep_instance.id)
    try:
        hermes_instance = await provider.run(
            hermes_energy_path,
            [
                qp_prep_instance.outs[0].id,
                qp_prep_instance.outs[1].id,
                qp_prep_instance.outs[2].id,
            ],
            hermes_target,
            hermes_resources,
            tags=tags,
            out_tags=([tags, tags] if tags else None),
            restore=restore,
        )

        print("launched hermes_instance", hermes_instance.id)
    except Exception:
        await provider.delete_module_instance(qp_prep_instance.id)
        raise

    try:
        qp_collate_instance = await provider.run(
            qp_collate_path,
            [
                hermes_instance.outs[0].id,
                qp_prep_instance.outs[3].id,
            ],
            tags=tags,
            out_tags=([tags] if tags else None),
            restore=restore,
            resources=ModuleInstanceResourcesInput(storage=20, storage_units=MemUnits.MB),
        )
    except Exception:
        await provider.delete_module_instance(qp_prep_instance.id)
        await provider.delete_module_instance(hermes_instance.id)
        raise

    return (
        Provider.Arg(provider=provider, id=hermes_instance.outs[0].id),
        Provider.Arg(provider=provider, id=qp_collate_instance.outs[0].id),
    )
