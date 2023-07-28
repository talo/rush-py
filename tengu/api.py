import base64
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import dataclasses_json
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

ModuleInstanceId = str


tag = gql(
    """
mutation tag($moduleInstanceId: ModuleInstanceId, $argumentId: ArgumentId, $moduleId: ModuleId, $tags: [String!]!) {
    tag(module_instance: $moduleInstanceId, argument: $argumentId, module: $moduleId, tags: $tags)
}
"""
)

untag = gql(
    """
mutation untag($moduleInstanceId: ModuleInstanceId, $argumentId: ArgumentId, $moduleId: ModuleId, $tags: [String!]!) {
    untag(module_instance: $moduleInstanceId, argument: $argumentId, module: $moduleId, tags: $tags)
}
"""
)

upload = gql(
    """
    mutation upload($typeinfo: JSON, $file: Upload) {
        upload(typeinfo: $typeinfo, file: $file ) { id value }
    }
    """
)


argument = gql(
    """
query ($id: ArgumentId!) {
    argument(id: $id) {
        id
        typeinfo
        value
        created_at
        tags
    }
    }
    """
)

arguments_query = gql(
    """
query ($first: Int, $after: String, $last: Int, $before: String, $typeinfo: JSON, $tags: [String!]) {
    me { account {
        arguments(first: $first, last: $last, after: $after, before: $before, typeinfo: $typeinfo, tags: $tags) {
            nodes {
                id
                typeinfo
                value
                created_at
                tags
            }
        }
    } }
}
"""
)


modules = gql(
    """
query ($first: Int, $after: String, $last: Int, $before: String, $path: String, $tags: [String!]) {
    modules(first: $first, last: $last, after: $after, before: $before, path: $path, tags: $tags) {
        nodes {
            id
            path
            created_at
            deleted_at
            ins
            outs
            tags
        }
    }
}
"""
)

latest_modules = gql(
    """
query ($first: Int, $after: String, $last: Int, $before: String, $names: [String!]) {
    latest_modules(first: $first, last: $last, after: $after, before: $before, names: $names) {
        nodes {
            id
            path
            created_at
            deleted_at
            ins
            outs
        }
    }
}
"""
)

run_mutation = gql(
    """
    mutation run($instance: ModuleInstanceInput) {
       run(instance: $instance) {
         id
         outs {id}
       }
    }
    """
)

delete_module_instance = gql(
    """
    mutation delete_module_instance($moduleInstanceId: ModuleInstanceId) {
        delete_module_instance(module: $moduleInstanceId) { id }
    }
    """
)

# module instance fragment
module_instance_fragment = """
    id
    tags
    created_at
    deleted_at
    account_id
    path
    ins {
      id
      typeinfo
      value
    }
    outs {
      id
      typeinfo
      value
    }
    queued_at
    admitted_at
    dispatched_at
    completed_at
    status
    target
    resources {
      gpus
      nodes
      mem
      storage
      walltime
    }
    progress {
      n
      n_expected
      n_max
      done
    }
"""


module_instance_query = gql(
    """query($id: ModuleInstanceId) {
    module_instance(id: $id) {
    """
    + module_instance_fragment
    + """
    stdout {
      nodes { content id created_at }
    }
    stderr {
      nodes { content id created_at }
    }
  } }"""
)

module_instances_query = gql(
    """query($first: Int, $after: String, $last: Int, $before: String, $path: String, $name: String, $status: ModuleInstanceStatus, $tags: [String!]) {
    me { account {
    module_instances(first: $first, last: $last, after: $after, before: $before, path: $path, status: $status, name: $name, tags: $tags) {
    nodes {
    """
    + module_instance_fragment
    + """
  } } } } }"""
)

object_query = gql(
    """query($id: ArgumentId!) {
        object(id: $id)
    }
    """
)


class DataClassJsonMixin(dataclasses_json.DataClassJsonMixin):
    """Override dataclass mixin so that we don't have `"property": null,`s in our output"""

    dataclass_json_config = dataclasses_json.config(  # type: ignore
        undefined=dataclasses_json.Undefined.EXCLUDE,
        exclude=lambda f: f is None,  # type: ignore
    )["dataclasses_json"]


T1 = TypeVar("T1")


@dataclass
class Arg(Generic[T1], DataClassJsonMixin):
    id: str | None = None
    value: T1 | None = None


frag_keywords = {
    "dimer_cutoff": 25,
    "dimer_mp2_cutoff": 25,
    "fragmentation_level": 2,
    "method": "MBE",
    "monomer_cutoff": 30,
    "monomer_mp2_cutoff": 30,
    "ngpus_per_node": 4,
    "reference_fragment": 293,
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


class Provider:
    """
    A class representing a provider for the Tengu quantum chemistry workflow platform.
    """

    def __init__(self, access_token: str, url: str = "https://tengu.qdx.ai"):
        """
        Initialize the TenguProvider with a specified URL and access token.

        :param url: The URL of the Tengu platform.
        :param access_token: The access token for authentication.
        """
        transport = RequestsHTTPTransport(url=url, headers={"authorization": f"bearer {access_token}"})

        self.client = Client(transport=transport)

    def argument(self, id: str) -> Any:
        """
        Retrieve an argument from the database.

        :param id: The ID of the argument.
        :return: The argument.
        """
        response = self.client.execute(argument, variable_values={"id": id})
        return response.get("argument")

    def arguments(
        self,
        first: int | None = None,
        after: str | None = None,
        last: int | None = None,
        before: str | None = None,
        tags: list[str] | None = None,
    ) -> list[Any]:
        """
        Retrieve a list of arguments.
        """
        response = self.client.execute(
            arguments_query,
            variable_values={
                "first": first,
                "after": after,
                "last": last,
                "before": before,
                "tags": tags,
            },
        )

        arguments = response["me"]["account"]["arguments"]

        if arguments:
            arguments = arguments.get("nodes")
            if arguments:
                return arguments
        return []

    def object(self, id):
        """
        Retrieve an object from the database.

        :param id: The ID of the object.
        :return: The object.
        """
        response = self.client.execute(object_query, variable_values={"id": id})
        return response.get("object")

    def latest_modules(
        self,
        first: int | None = None,
        after: str | None = None,
        last: int | None = None,
        before: str | None = None,
        names: list[str] | None = None,
    ):
        """
        Retrieve a list of modules.

        :param path: The path of the module.
        :param name: The name of the module.
        :param first: The maximum number of modules to retrieve.
        :param after: The cursor to start retrieving modules from.
        :param last: The maximum number of modules to retrieve.
        :param before: The cursor to start retrieving modules from.
        :return: A list of modules.
        """
        response = self.client.execute(
            latest_modules,
            variable_values={
                "first": first,
                "after": after,
                "last": last,
                "before": before,
                "names": names,
            },
        )
        return response.get("latest_modules")

    def modules(
        self,
        path: str | None = None,
        name: str | None = None,
        first: int | None = None,
        after: str | None = None,
        last: int | None = None,
        before: str | None = None,
        tags: list[str] | None = None,
    ):
        """
        Retrieve a list of modules.

        :param path: The path of the module.
        :param name: The name of the module.
        :param first: The maximum number of modules to retrieve.
        :param after: The cursor to start retrieving modules from.
        :param last: The maximum number of modules to retrieve.
        :param before: The cursor to start retrieving modules from.
        :return: A list of modules.
        """
        response = self.client.execute(
            modules,
            variable_values={
                "path": path,
                "name": name,
                "first": first,
                "after": after,
                "last": last,
                "before": before,
                "tags": tags,
            },
        )
        return response.get("modules")

    def run(
        self,
        path: str,
        args: list[Arg],
        target: Literal["GADI", "NIX", "NIX_SSH"] | None = None,
        resources: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        out_tags: list[list[str] | None] | None = None,
    ):
        """
        Run a module with the given inputs and outputs.
        :param path: The path of the module.
        :param args: The arguments to the module.
        :param target: The target to run the module on.
        :param resources: The resources to run the module with.
        :param tags: The tags to apply to the module.
        :param out_tags: The tags to apply to the outputs of the module.
                         If provided, must be the same length as the number of outputs.
        """
        response = self.client.execute(
            run_mutation,
            variable_values={
                "instance": {
                    "path": path,
                    "args": [arg.to_dict() for arg in args],
                    "target": target,
                    "resources": resources,
                    "tags": tags,
                    "out_tags": out_tags,
                }
            },
        )
        module_instance = response.get("run")
        if module_instance:
            return module_instance
        else:
            raise RuntimeError(response)

    def qp_run(
        self,
        qp_gen_inputs_path: str,
        hermes_energy_path: str,
        qp_collate_path: str,
        pdb: Arg[Path],
        gro: Arg[Path],
        lig: Arg[Path],
        lig_type: Arg[Literal["sdf", "mol2"]],
        lig_res_id: Arg[str],
        model: Arg[dict[str, Any]] = Arg(None, default_model),
        keywords: Arg[dict[str, Any]] = Arg(None, {"frag": frag_keywords, "scf": scf_keywords}),
        amino_acids_of_interest: Arg[list[tuple[str, int]]] = Arg(None, None),
        dry_run: Arg[bool] = Arg(None, False),
        export_density: Arg[bool] = Arg(None, False),
        target: Literal["GADI", "NIX", "NIX_SSH"] | None = None,
        resources: dict[str, Any] | None = None,
        autopoll: tuple[int, int] | None = None,
        tags: list[str] | None = None,
    ):
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
        """

        qp_prep_instance = self.run(
            qp_gen_inputs_path,
            [pdb, gro, lig, lig_type, lig_res_id, model, keywords, amino_acids_of_interest],
            tags=tags,
            out_tags=([tags, tags, tags, tags] if tags else None),
        )
        try:
            hermes_instance = self.run(
                hermes_energy_path,
                [
                    Arg(qp_prep_instance["outs"][0]["id"], None),
                    Arg(qp_prep_instance["outs"][1]["id"], None),
                    Arg(qp_prep_instance["outs"][2]["id"], None),
                    dry_run,
                    export_density,
                ],
                target,
                resources,
                tags=tags,
                out_tags=([tags] if tags else None),
            )
        except:
            self.delete_module_instance(qp_prep_instance["id"])
            raise

        try:
            qp_collate_instance = self.run(
                qp_collate_path,
                [
                    Arg(hermes_instance["outs"][0]["id"], None),
                    Arg(qp_prep_instance["outs"][3]["id"], None),
                ],
                tags=tags,
                out_tags=([tags] if tags else None),
            )
        except:
            self.delete_module_instance(qp_prep_instance["id"])
            self.delete_module_instance(hermes_instance["id"])
            raise

        if autopoll:
            prep = self.poll_module_instance(qp_prep_instance["id"], *autopoll)
            if prep["status"] == "FAILED":
                self.delete_module_instance(hermes_instance["id"])
                self.delete_module_instance(qp_collate_instance["id"])
                raise RuntimeError(prep["error"])

            hermes = self.poll_module_instance(hermes_instance["id"], *autopoll)
            if hermes["status"] == "FAILED":
                self.delete_module_instance(qp_collate_instance["id"])
                raise RuntimeError(hermes["error"])

            collate = self.poll_module_instance(qp_collate_instance["id"], *autopoll)

            return collate

        return [qp_prep_instance, hermes_instance, qp_collate_instance]

    def delete_module_instance(self, id: ModuleInstanceId):
        """
        Delete a module instance with a given ID.

        :param id: The ID of the module instance to be deleted.
        :return: The ID of the deleted module instance.
        :raise RuntimeError: If the operation fails.
        """
        response = self.client.execute(delete_module_instance, variable_values={"moduleInstanceId": id})

        taskId = response.get("delete_module_instance")
        if taskId:
            return ModuleInstanceId(taskId)
        else:
            raise RuntimeError(response)

    def poll_module_instance(self, id: ModuleInstanceId, n_retries: int = 10, poll_rate: int = 30) -> Any:
        """
        Poll a module instance until it is completed, with a specified number of retries and poll rate.

        :param id: The ID of the module instance to be polled.
        :param n_retries: The maximum number of retries. Default is 10.
        :param poll_rate: The poll rate in seconds. Default is 30.
        :return: The completed module instance.
        :raise Exception: If the module instance fails or polling times out.
        """
        n_try = 0

        while n_try < n_retries:
            n_try += 1
            response = self.client.execute(module_instance_query, variable_values={"id": id})
            module_instance: dict[str, Any] | None = response.get("module_instance")
            if module_instance and module_instance["status"] in ["COMPLETED", "FAILED"]:
                return module_instance

            time.sleep(poll_rate)

        raise Exception("Module polling timed out")

    def module_instance(self, id: ModuleInstanceId) -> Any:
        """
        Retrieve a module instance by its ID.

        :param id: The ID of the module instance to be retrieved.
        :return: The retrieved module instance.
        :raise Exception: If the module instance is not found.
        """
        response = self.client.execute(module_instance_query, variable_values={"id": id})
        module_instance: dict[str, Any] | None = response.get("module_instance")
        if module_instance:
            return module_instance

        raise Exception("Failed to find task")

    def upload_arg(self, file: Path) -> Arg:
        """
        Converts a file to bas64 and formats it for use as an argument

        :param file: The file to be uploaded.
        :return: The formatted file.
        """
        with open(file, "rb") as f:
            return Arg(None, value=base64.b64encode(f.read()).decode("utf-8"))

    def tag(
        self,
        tags: list[str],
        module_id: str | None = None,
        module_instance_id: str | None = None,
        argument_id: str | None = None,
    ) -> list[str]:
        """
        Add a list of tags to a module, module instance, or argument.

        :param tags: The list of tags to be added.
        :param module_id: The ID of the module to be tagged.
        :param module_instance_id: The ID of the module instance to be tagged.
        :param argument_id: The ID of the argument to be tagged.
        :return: The resulting full list of tags on the entity.
        """
        response = self.client.execute(
            tag,
            variable_values={
                "tags": tags,
                "moduleId": module_id,
                "moduleInstanceId": module_instance_id,
                "argumentId": argument_id,
            },
        )
        return response["tag"]

    def untag(
        self,
        tags: list[str],
        module_id: str | None = None,
        module_instance_id: str | None = None,
        argument_id: str | None = None,
    ) -> list[str]:
        """
        Remove a list of tags from a module, module instance, or argument.

        :param tags: The list of tags to be removed.
        :param module_id: The ID of the module to be untagged.
        :param module_instance_id: The ID of the module instance to be untagged.
        :param argument_id: The ID of the argument to be untagged.
        :return: The list of remaining tags.
        """
        response = self.client.execute(
            untag,
            variable_values={
                "tags": tags,
                "moduleId": module_id,
                "moduleInstanceId": module_instance_id,
                "argumentId": argument_id,
            },
        )
        return response["untag"]

    def upload(
        self,
        file: Path | str,
        typeinfo: dict[str, Any],
    ):
        with open(file, "rb") as f:
            response = self.client.execute(
                upload, variable_values={"file": f, "typeinfo": typeinfo}, upload_files=True
            )
        return response["upload"]

    def module_instances(
        self,
        before: str | None = None,
        after: str | None = None,
        first: int | None = None,
        last: int | None = None,
        path: str | None = None,
        name: str | None = None,
        status: Literal["CREATED", "ADMITTED", "QUEUED", "DISPATCHED", "COMPLETED", "FAILED"] | None = None,
        tags: list[str] | None = None,
    ) -> list[Any]:
        """
        Retrieve a list of module instancees filtered by the given parameters.

        :param before: Retrieve module instances before a certain cursor.
        :param after: Retrieve module instances after a certain cursor.
        :param first: Retrieve the first N module instances.
        :param last: Retrieve the last N module instances.
        :param tags: Retrieve module instancees with the given list of tags.
        :param status: Retrieve module instancees with the specified status ("CREATED", "ADMITTED", "QUEUED", "DISPATCHED", "COMPLETED", "FAILED").
        :return: A list of filtered module instancee.
        """
        response = self.client.execute(
            module_instances_query,
            variable_values={
                "first": first,
                "last": last,
                "before": before,
                "after": after,
                "path": path,
                "name": name,
                "status": status,
                "tags": tags,
            },
        )
        module_instances = response["me"]["account"]["module_instances"]

        if module_instances:
            module_instances = module_instances.get("nodes")
            if module_instances:
                return module_instances

        return []
