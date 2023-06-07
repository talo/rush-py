import base64
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Literal

import dataclasses_json
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

ModuleInstanceId = str

modules = gql(
    """
query ($first: Int, $after: String, $last: Int, $before: String, $path: String) {
    modules(first: $first, last: $last, after: $after, before: $before, path: $path) {
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
        deleteModuleInstance(module: $moduleInstanceId) { id }
    }
    """
)

# module instance fragment
module_instance_fragment = """
    id
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
    """query($first: Int, $after: String, $last: Int, $before: String, $path: String, $status: ModuleInstanceStatus) {
    me { account {
    module_instances(first: $first, last: $last, after: $after, before: $before, path: $path, status: $status) {
    nodes {
    """
    + module_instance_fragment
    + """
  } } } } }"""
)


class DataClassJsonMixin(dataclasses_json.DataClassJsonMixin):
    """Override dataclass mixin so that we don't have `"property": null,`s in our output"""

    dataclass_json_config = dataclasses_json.config(  # type: ignore
        undefined=dataclasses_json.Undefined.EXCLUDE,
        exclude=lambda f: f is None,  # type: ignore
    )["dataclasses_json"]


@dataclass
class Arg(DataClassJsonMixin):
    id: str | None
    value: Any | None


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

    def modules(
        self,
        path: str | None = None,
        name: str | None = None,
        first: int | None = None,
        after: str | None = None,
        last: int | None = None,
        before: str | None = None,
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
            },
        )
        return response.get("modules")

    def run(
        self,
        path: str,
        args: list[Arg],
        target: Literal["GADI", "NIX"] | None = None,
        resources: dict[str, Any] | None = None,
    ):
        """
        Run a module with the given inputs and outputs.
        """
        response = self.client.execute(
            run_mutation,
            variable_values={
                "instance": {
                    "path": path,
                    "args": [arg.to_dict() for arg in args],
                    "target": target,
                    "resources": resources,
                }
            },
        )
        module_instance = response.get("run")
        if module_instance:
            return module_instance
        else:
            raise RuntimeError(response)

    def qp_run(self, args: list[Arg], target: Literal["GADI", "NIX"] | None = None):
        """
        Construct the input and output module instance calls for QP run.
        """

        frag_keywords = {
            "dimer_cutoff": 10,
            "dimer_mp2_cutoff": 10,
            "fragmentation_level": 2,
            "method": "MBE",
            "monomer_cutoff": 20,
            "monomer_mp2_cutoff": 20,
            "ngpus_per_node": 1,
            "reference_fragment": 293,
            "trimer_cutoff": 10,
            "trimer_mp2_cutoff": 10,
            "lattice_energy_calc": True,
        }

        scf_keywords = {
            "convergence_metric": "diis",
            "dynamic_screening_threshold_exp": 10,
            "ndiis": 8,
            "niter": 40,
            "scf_conv": 0.000001,
        }

        qp_prep_instance = self.run(
            "github:talo/tengu-prelude/91e75238fb80e6fb92c9d678d84fd2778ff8e958#qp_gen_inputs",
            [
                self.upload_arg(Path("/home/ryanswart/Downloads/JAK2_3E64_lig22_md1_12ns.pdb")),
                self.upload_arg(Path("/home/ryanswart/Downloads/JAK2_3E64_lig22_GMX.gro")),
                self.upload_arg(Path("/home/ryanswart/Downloads/jak2_lig22.sdf")),
                Arg(None, "sdf"),
                Arg(None, "MOL"),
                Arg(
                    None,
                    None,
                ),  # {"model": "RHF", "basis": "6-31G*", "aux_basis": "6-31G*", "frag_enabled": True}),
                Arg(None, {"frag": frag_keywords, "scf": scf_keywords}),
                Arg(None, None),
            ],
        )

        print(qp_prep_instance)
        done = self.poll_module_instance(qp_prep_instance["id"])
        print(done)

        hermes_instance = self.run(
            "github:talo/tengu-prelude/0be073990adcee68020f6851f90c9404c12c8fc6#hermes_energy",
            [
                Arg(qp_prep_instance["outs"][0]["id"], None),
                Arg(qp_prep_instance["outs"][1]["id"], None),
                Arg(qp_prep_instance["outs"][2]["id"], None),
            ],
            "GADI",
            {"walltime": 120},
        )
        # qp_collate = self.run("", [])

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

    def module_instances(
        self,
        before: str | None = None,
        after: str | None = None,
        first: int | None = None,
        last: int | None = None,
        path: str | None = None,
        status: Literal["CREATED", "ADMITTED", "QUEUED", "DISPATCHED", "COMPLETED", "FAILED"] | None = None,
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
                "status": status,
            },
        )
        module_instances = response["me"]["account"]["module_instances"]

        if module_instances:
            module_instances = module_instances.get("nodes")
            if module_instances:
                return module_instances

        return []
