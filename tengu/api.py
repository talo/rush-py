import base64
import json
import time
import uuid
from dataclasses import dataclass
from functools import reduce
from io import IOBase
from pathlib import Path
from typing import Any, Generic, Iterable, Literal, TypeVar

import dataclasses_json
import requests
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

ArgId = uuid.UUID
ModuleInstanceId = uuid.UUID

Targets = Literal["GADI", "NIX", "NIX_SSH", "NIX_SSH_2", "SETONIX"]

tag = gql(
    """
mutation tag($moduleInstanceId: ModuleInstanceId, $argumentId: ArgumentId, $moduleId: ModuleId, $tags: [String!]!) {
    tag(module_instance: $moduleInstanceId, argument: $argumentId, module: $moduleId, tags: $tags)
}
"""
)

retry = gql(
    """
mutation retry($instance: ModuleInstanceId!, $resources: ModuleInstanceResourcesInput, $target: ModuleInstanceTarget) {
    retry(instance: $instance, resources: $resources, target: $target) {
        id
        tags
        target
    }
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

page_info = """
    pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
    }
"""


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
    """
    + page_info
    + """
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
    """
    + page_info
    + """
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
    """
    + page_info
    + """
        nodes {
            id
            path
            created_at
            deleted_at
            usage
            description
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
    """
    + page_info
    + """
      nodes { content id created_at }
    }
    stderr {
      nodes { content id created_at }
    }
    resource_utilization {
      sus
      gpu
      target
      storage
      walltime
      cputime
      mem
    }
    } }"""
)

module_instances_query = gql(
    """query($first: Int, $after: String, $last: Int, $before: String, $path: String, $name: String, $status: ModuleInstanceStatus, $tags: [String!]) {
    me { account {
    module_instances(first: $first, last: $last, after: $after, before: $before, path: $path, status: $status, name: $name, tags: $tags) {
    """
    + page_info
    + """
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

    def _query_with_pagination(
        self, query, variables: dict[str, Any], page_info_path: list[str], nodes_path: list[str]
    ):
        original_before = variables.get("before")
        page_info_res = {"hasPreviousPage": True, "endCursor": original_before}

        while page_info_res["hasPreviousPage"]:
            new_vars = variables | {"before": page_info_res["endCursor"]}
            result = self.client.execute(query, variable_values=new_vars)
            if result is None:
                raise RuntimeError(result)

            page_info_res = reduce(dict.get, page_info_path, result) or {"hasPreviousPage": False}
            yield reduce(dict.get, nodes_path, result) or []

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
    ) -> Iterable[Any]:
        """
        Retrieve a list of arguments.
        """
        variables = {
            "first": first,
            "after": after,
            "last": last,
            "before": before,
            "tags": tags,
        }
        return self._query_with_pagination(
            arguments_query,
            variables,
            ["me", "account", "arguments", "pageInfo"],
            ["me", "account", "arguments", "nodes"],
        )

    def object(self, id):
        """
        Retrieve an object from the database.

        :param id: The ID of the object.
        :return: The object.
        """
        response = self.client.execute(object_query, variable_values={"id": id})

        object = response.get("object")
        if object is None:
            raise RuntimeError(response)
        return object

    def download_object(self, id, filepath: Path):
        """
        Retrieve an object from the store: a wrapper for object with simpler behavior.

        :param id: The ID of the object.
        :param filepath: Where to download the object.
        """
        if isinstance(id, ArgId):
            id = str(id)
        obj = self.object(id)

        if "url" in obj:
            with requests.get(obj["url"], stream=True) as r:
                r.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content():
                        f.write(chunk)
        else:
            with open(filepath, "w") as f:
                f.write(obj)

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
        variables = {
            "first": first,
            "after": after,
            "last": last,
            "before": before,
            "names": names,
        }

        return self._query_with_pagination(
            latest_modules, variables, ["latest_modules", "pageInfo"], ["latest_modules", "nodes"]
        )

    def get_latest_module_paths(self, names: list[str] | None = None) -> dict[str, str]:
        """
        Get the latest module paths for a list of modules.

        :param names: The names of the modules.
        """
        ret = {}
        module_pages = self.latest_modules(names=names)
        for module_page in module_pages:
            for module in module_page:
                path = module["path"]
                name = module["path"].split("#")[-1]
                if path:
                    ret[name] = path
        return ret

    def load_module_paths(self, filename: Path) -> dict[str, str]:
        """
        Load all of the module versions from a file.

        :param filename: Json module version file
        """
        modules = None
        with open(filename, "r") as f:
            modules = json.load(f)
        return modules

    def save_module_paths(self, modules: dict[str, str], filename: Path | None = None):
        """
        Lock all of the module versions to a file.

        :param modules: The modules to lock.
        """
        filename = filename or Path(f'modules-{time.strftime("%Y%m%dT%H%M%S")}.json')
        with open(filename, "w") as f:
            json.dump(modules, f, indent=2)
        return filename

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
        variables = {
            "path": path,
            "name": name,
            "first": first,
            "after": after,
            "last": last,
            "before": before,
            "tags": tags,
        }

        return self._query_with_pagination(modules, variables, ["modules", "pageInfo"], ["modules", "nodes"])

    def run2(
        self,
        path: str,
        args: list[Arg],
        target: Targets | None = None,
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

        # TODO: less insane version of this
        def gen_arg_dict(input):
            arg = None
            if isinstance(input, Arg):
                arg = input
            elif isinstance(input, ArgId):
                arg = Arg(id=str(input))
            elif isinstance(input, Path):
                arg = Arg(value=base64.b64encode(input.read_bytes()).decode("utf-8"))
            elif isinstance(input, IOBase):
                data = input.read()
                # The only other case is bytes-like, i.e. isinstance(data, (bytes, bytearray))
                if isinstance(data, str):
                    data = data.encode("utf-8")
                arg = Arg(value=base64.b64encode(data).decode("utf-8"))
            else:
                arg = Arg(value=input)
            return arg.to_dict()

        arg_dicts = [gen_arg_dict(input) for input in args]
        response = self.client.execute(
            run_mutation,
            variable_values={
                "instance": {
                    "path": path,
                    "args": arg_dicts,
                    "target": target,
                    "resources": resources,
                    "tags": tags,
                    "out_tags": out_tags,
                }
            },
        )
        module_instance = response.get("run")
        if module_instance:
            out2 = {}
            # Convert IDs into ArgId type to keep them differentiated from vanilla strings
            out2["module_instance_id"] = ArgId(module_instance["id"])
            out2["output_ids"] = [ArgId(out["id"]) for out in module_instance["outs"]]
            return out2
        else:
            raise RuntimeError(response)

    def run(
        self,
        path: str,
        args: list[Arg],
        target: Targets | None = None,
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

    def run_qp(
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
        keywords: Arg[dict[str, Any]] = Arg(
            None, {"frag": frag_keywords, "scf": scf_keywords, "debug": {}, "export": {}, "guess": {}}
        ),
        amino_acids_of_interest: Arg[list[tuple[str, int]]] = Arg(None, None),
        target: Targets | None = None,
        resources: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        autopoll: tuple[int, int] | None = None,
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

        if resources is not None and "gpus" in resources and keywords.value is not None:
            keywords.value["frag"]["ngpus_per_node"] = resources["gpus"]

        qp_prep_instance = self.run(
            qp_gen_inputs_path,
            [pdb, gro, lig, lig_type, lig_res_id, model, keywords, amino_acids_of_interest],
            tags=tags,
            out_tags=([tags, tags, tags, tags] if tags else None),
        )
        print("launched qp_prep_instance", qp_prep_instance)
        try:
            hermes_instance = self.run(
                hermes_energy_path,
                [
                    Arg(qp_prep_instance["outs"][0]["id"], None),
                    Arg(qp_prep_instance["outs"][1]["id"], None),
                    Arg(qp_prep_instance["outs"][2]["id"], None),
                ],
                target,
                resources,
                tags=tags,
                out_tags=([tags, tags] if tags else None),
            )

            print("launched hermes_instance", hermes_instance)
        except Exception:
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
        except Exception:
            self.delete_module_instance(qp_prep_instance["id"])
            self.delete_module_instance(hermes_instance["id"])
            raise

        if autopoll:
            time.sleep(autopoll[0])
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

        We do exponential backoff from 0.5s, so a retry only counts once we hit the input poll rate.

        :param id: The ID of the module instance to be polled.
        :param n_retries: The maximum number of retries. Default is 10.
        :param poll_rate: The poll rate in seconds. Default is 30.
        :return: The completed module instance.
        :raise Exception: If the module instance fails or polling times out.
        """
        n_try = 0

        if isinstance(id, ModuleInstanceId):
            id = str(id)

        curr_poll_rate = 0.5
        while n_try < n_retries:
            time.sleep(curr_poll_rate)
            if curr_poll_rate == poll_rate:
                n_try += 1
            curr_poll_rate = min(curr_poll_rate * 2, poll_rate)
            response = self.client.execute(module_instance_query, variable_values={"id": id})
            module_instance: dict[str, Any] | None = response.get("module_instance")
            if module_instance and module_instance["status"] in ["COMPLETED", "FAILED"]:
                return module_instance

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
        Converts a file to base64 and formats it for use as an argument

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
        """
        Upload an Object with typeinfo and store as an Argument.

        :param file: The file to be uploaded.
        :param typeinfo: The typeinfo of the file.
        """
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
    ):
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
        variables = {
            "first": first,
            "last": last,
            "before": before,
            "after": after,
            "path": path,
            "name": name,
            "status": status,
            "tags": tags,
        }
        return self._query_with_pagination(
            module_instances_query,
            variables,
            ["me", "account", "module_instances", "pageInfo"],
            ["me", "account", "module_instances", "nodes"],
        )

    def retry(self, id: ModuleInstanceId, resources=None, target=None) -> ModuleInstanceId:
        """
        Retry a module instance.

        :param id: The ID of the module instance to be retried.
        :return: The ID of the new module instance.
        """
        response = self.client.execute(
            retry, variable_values={"instance": id, "resources": resources, "target": target}
        )
        taskId = response.get("retry")
        if taskId:
            return ModuleInstanceId(taskId["id"])
        else:
            raise RuntimeError(response)
