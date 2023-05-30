import base64
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
      nodes { content id createdAt }
    }
    stderr {
      nodes { content id createdAt }
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

    def run(self, path: str, args: dict[str, Any], target: Literal["GADI", "NIX"] | None = None):
        """
        Run a module with the given inputs and outputs.
        """
        response = self.client.execute(
            run_mutation, variable_values={"instance": {"path": path, "args": args, "target": target}}
        )
        module_instance = response.get("run")
        if module_instance:
            return module_instance
        else:
            raise RuntimeError(response)

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

    def upload_arg(self, file: Path) -> dict[str, str]:
        """
        Converts a file to bas64 and formats it for use as an argument

        :param file: The file to be uploaded.
        :return: The formatted file.
        """
        with open(file, "rb") as f:
            return {"value": base64.b64encode(f.read()).decode("utf-8")}

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
