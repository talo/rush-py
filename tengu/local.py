#!/usr/bin/env python3
import base64
import json
import os
from time import time
import uuid
import pathlib
from subprocess import run

from typing import Any

from .api import Arg, Provider

STATUS_PATH = "github:talo/tengu-module-flake-parts/6798b0c701ff0caa69a064d4e57431357fc3807c#dummy"


class LocalProvider:
    args = {}
    instances = {}

    def __init__(self, remote_provider: Provider | None = None):
        self.remote_provider = remote_provider
        # load the tengu config from the users xdg config directory
        config_file = pathlib.Path.home().joinpath(".config/tengu.json")
        if not config_file.exists():
            raise ValueError("Could not find tengu config file -- please run tengu-runtime install")
        self.config = json.loads(config_file.read_text())

    def upload_arg(self, file: pathlib.Path) -> Arg:
        """
        Converts a file to bas64 and formats it for use as an argument

        :param file: The file to be uploaded.
        :return: The formatted file.
        """
        with open(file, "rb") as f:
            return Arg(None, value=base64.b64encode(f.read()).decode("utf-8"))

    def resolve_arg(self, arg: Arg, retries: int = 0) -> Any:
        """
        Resolve a module path to a local path.
        :param path: The path to resolve.
        :return: The resolved path.
        """
        if arg.value is not None:
            return arg.value
        if self.args.get(arg.id) is not None:
            cached_arg = self.args[arg.id]
            if cached_arg["resolved"]:
                return self.args[arg.id]["value"]
            elif self.instances[cached_arg["instance_id"]]["status"] == "Done":
                self.args[arg.id]["value"] = self.instances[cached_arg["instance_id"]]["values"][
                    cached_arg["idx"]
                ]
                self.args[arg.id]["resolved"] = True
                return self.args[arg.id]["value"]

        if self.remote_provider is not None and arg.id:
            if retries > 5:
                raise ValueError(f"Argument is not yet ready {arg.id}")
            remote_arg = self.remote_provider.argument(arg.id)
            if remote_arg is not None:
                return self.resolve_arg(remote_arg, retries=retries + 1)
        raise ValueError(f"Could not resolve argument {arg.id}")

    def process_in(self, arg: Arg, in_a: dict[str, Any]) -> str:
        if in_a["k"] != "option" and arg.value is None:
            arg = self.resolve_arg(arg)

        if arg.value is not None:
            if in_a["k"] == "object":
                # copy to tengu store
                id = str(uuid.uuid4())
                if len(arg.value) == len(id):
                    # is probably already a tengu store object
                    pass
                else:
                    write_path = pathlib.Path(self.config["cache_dir"]).joinpath("store").joinpath(id)
                    with open(write_path, "w") as f:
                        if in_a["t"] == "bytes":
                            # decode the base64 bytes
                            decoded = base64.b64decode(arg.value)
                            f.write(decoded.decode("utf-8"))
                        else:
                            f.write(json.dumps(arg.value))
                return json.dumps(id)
            if in_a["k"] == "option":
                if in_a["t"]["k"] == "object":
                    return self.process_in(arg, in_a["t"])
                else:
                    return arg.value
            else:
                return json.dumps(arg.value)
        else:
            return "null"

    def process_ins(self, args: list[Arg], ins: list[dict[str, Any]]) -> list[str]:
        return [self.process_in(arg, in_a) for arg, in_a in zip(args, ins)]

    def run(
        self,
        path: str,
        args: list[Arg],
        _target: Any = None,
        _resources: Any | None = None,
        tags: list[str] | None = None,
        out_tags: list[list[str] | None] | None = None,
    ):
        """
        Run a module locally with the given inputs and outputs.
        :param path: The path of the module.
        :param args: The arguments to the module.
        :param target: The target to run the module on.
        :param resources: The resources to run the module with.
        :param tags: The tags to apply to the module.
        :param out_tags: The tags to apply to the outputs of the module.
                         If provided, must be the same length as the number of outputs.
        """
        manifest = json.loads(
            run(["nix", "run", path, "--", "manifest"], capture_output=True).stdout.decode("utf-8")
        )
        outs = [{"id": str(uuid.uuid4()), "typeinfo": out} for out in manifest["outs"]]

        resolved_args = self.process_ins(args, manifest["ins"])
        print("resolved args", resolved_args)

        instance_id = str(uuid.uuid4())
        # launch nix via subprocess to run the module
        run(
            ["nix", "run", path + "+nix+runner", "--"] + resolved_args + ['"result"', '"progress"'],
            env={"INSTANCE_ID": instance_id, "PATH": os.getenv("PATH") or ""},
        )
        self.instances[instance_id] = {"id": instance_id, "outs": outs}
        for idx, out in enumerate(outs):
            self.args[out["id"]] = {"idx": idx, "instance_id": instance_id, "value": None, "resolved": False}

        return self.instances[instance_id]

    def module_instance(
        self,
        id: str,
        path: str = STATUS_PATH,
    ):
        cache = self.instances.get(id)
        if cache is not None and cache.get("status") == "Done":
            return cache
        if cache is None:
            cache = {"outs": []}

        res = json.loads(
            run(
                ["nix", "run", path + "+nix+status"],
                env={"INSTANCE_ID": id, "PATH": os.getenv("PATH") or ""},
                capture_output=True,
            ).stdout.decode("utf-8")
        )
        self.instances[id]["status"] = res["status"]
        if res["status"] == "Done":
            self.instances[id]["outs"] = [
                {"value": value, "id": out["id"]} for (out, value) in zip(cache["outs"] or [], res["values"])
            ]
        return self.instances[id]

    def object(
        self,
        id: str,
    ):
        store_dir = pathlib.Path(self.config.get("cache_dir")).joinpath("store")
        if store_dir.exists():
            return store_dir.joinpath(id).read_text()

    def poll_module_instance(self, id: str, n_retries: int = 10, poll_rate: int = 30) -> Any:
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
            response = self.module_instance(id)
            if response and response["status"] in ["COMPLETED", "FAILED"]:
                return response

            time.sleep(poll_rate)

        raise Exception("Module polling timed out")
