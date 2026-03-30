"""Object store references and helpers."""

import json
import tarfile
import uuid
from dataclasses import dataclass
from functools import singledispatch
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, NewType, Self

import requests
import zstandard as zstd
from gql import FileVar, gql

from .convert import from_json
from .mol import TRC, Chains, Residues, Topology
from .session import _get_client, _get_config, _get_project_id

#: UUID identifying an object in the Rush object store.
ObjectID = NewType("ObjectID", str)


@dataclass(frozen=True)
class RushObject:
    """Reference to an object in the Rush object store."""

    #: UUID path in the object store.
    path: ObjectID
    #: Size in bytes.
    size: int
    #: Storage format.
    format: Literal["Json", "Bin"]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Construct from a raw GraphQL output dict.

        Requires ``path``, ``size``, and ``format`` keys.
        """
        try:
            return cls(
                path=ObjectID(d["path"]),
                size=d["size"],
                format=d["format"],
            )
        except KeyError as e:
            raise ValueError(
                f"RushObject dict missing required key {e}; got keys: {list(d.keys())}"
            ) from e

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "size": self.size, "format": self.format}

    def fetch(self, extract: bool = False) -> list[Any] | dict[str, Any] | bytes:
        """Download this object into memory."""
        if self.format.lower() == "json":
            return self.fetch_json()
        return self.fetch_bytes(extract=extract)

    def fetch_json(self) -> list[Any] | dict[str, Any]:
        """Download this JSON object into memory."""
        if self.format.lower() != "json":
            raise TypeError(f"{self.path} is not a Json object")
        return _fetch_json_object(self.path)

    def fetch_dict(self) -> dict[str, Any]:
        """Download this JSON object and require a dictionary payload."""
        if self.format.lower() != "json":
            raise TypeError(f"{self.path} is not a Json object")
        return _fetch_json_object_as_dict(self.path)

    def fetch_list(self) -> list[Any]:
        """Download this JSON object and require a list payload."""
        if self.format.lower() != "json":
            raise TypeError(f"{self.path} is not a Json object")
        return _fetch_json_object_as_list(self.path)

    def fetch_bytes(self, extract: bool = False) -> bytes:
        """Download this binary object into memory."""
        if self.format.lower() != "bin":
            raise TypeError(f"{self.path} is not a Bin object")
        return _fetch_bin_object(self.path, extract=extract)

    def save(
        self,
        filepath: Path | str | None = None,
        name: str | None = None,
        ext: str | None = None,
        extract: bool = False,
    ) -> Path:
        """Download this object and save to the workspace.

        The file type is derived from :attr:`format` automatically.
        Pass *ext* to override the file extension (e.g. ``"hdf5"``, ``"a3m"``).
        """
        if ext is None:
            ext = self.format.lower()

        workspace_project_dir = _get_config().workspace_dir / _get_project_id()
        if filepath is not None and name is None:
            filepath = Path(filepath) if isinstance(filepath, str) else filepath
        elif filepath is None and name is not None:
            filepath = workspace_project_dir / f"{name}.{ext}"
        elif filepath is None and name is None:
            filepath = workspace_project_dir / f"{self.path}.{ext}"
        else:
            raise Exception("Cannot specify both filepath and name")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.format.lower() == "json":
            d = self.fetch_json()
            with filepath.open("w") as f:
                json.dump(_clean_dict(d), f, indent=2)
        else:
            b = self.fetch_bytes(extract=extract)
            with filepath.open("wb") as f:
                f.write(b)

        return filepath


@dataclass(frozen=True)
class TRCPaths:
    """Workspace paths for a saved TRC triplet."""

    topology: Path
    residues: Path
    chains: Path


@dataclass(frozen=True)
class TRCRef:
    """Reference to a single TRC triplet in the Rush object store."""

    topology: RushObject
    residues: RushObject
    chains: RushObject

    @classmethod
    def upload(cls, trc: TRC) -> Self:
        return cls(
            RushObject.from_dict(upload_object(trc.topology.to_dict())),
            RushObject.from_dict(upload_object(trc.residues.to_dict())),
            RushObject.from_dict(upload_object(trc.chains.to_dict())),
        )

    def fetch(self) -> TRC:
        """Download and parse into a TRC."""
        topology = self.topology.fetch_dict()
        residues = self.residues.fetch_dict()
        chains = self.chains.fetch_dict()
        return from_json(
            {
                "topology": topology,
                "residues": residues,
                "chains": chains,
            }
        )

    def save(self) -> TRCPaths:
        """Download and save to the workspace."""
        return TRCPaths(
            topology=self.topology.save(),
            residues=self.residues.save(),
            chains=self.chains.save(),
        )


def upload_object(input: Path | str | dict[str, Any]) -> dict[str, Any]:
    """
    Upload an object at the filepath to the current project. Usually not necessary; the
    module functions should handle this automatically.
    """
    mutation = gql("""
        mutation UploadObject($file: Upload!, $typeinfo: Json!, $format: ObjectFormatEnum!, $project_id: String) {
            upload_object(file: $file, typeinfo: $typeinfo, format: $format, project_id: $project_id) {
                id
                object {
                    path
                    size
                    format
                }
                base_url
                url
            }
        }
     """)
    if isinstance(input, dict):
        t_f = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(input, t_f)
        t_f.close()
        return upload_object(t_f.name)

    filepath = Path(input) if isinstance(input, str) else input
    with filepath.open(mode="rb") as f:
        project_id = _get_project_id()
        if filepath.suffix == ".json":
            mutation.variable_values = {
                "file": FileVar(f),
                "format": "json",
                "typeinfo": {
                    "k": "record",
                    "t": {},
                },
                "project_id": project_id,
            }
        else:
            mutation.variable_values = {
                "file": FileVar(f),
                "format": "bin",
                "typeinfo": {
                    "k": "record",
                    "t": {
                        "size": "u32",
                        "path": {
                            "k": "@",
                            "t": "$Bytes",
                        },
                    },
                    "n": "Object",
                },
                "project_id": project_id,
            }
        result = _get_client().execute(mutation, upload_files=True)

    return result["upload_object"]["object"]


def _extract_object_archive(data: bytes) -> bytes:
    decompressed = zstd.ZstdDecompressor().decompress(data, max_output_size=int(1e9))
    with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
        tar_filenames = tar.getnames()

        # Handle empty tar archives
        if not tar_filenames:
            raise ValueError("Tar archive is empty - no files to extract")

        # Extract the appropriate file:
        # - If 1 file: extract that file
        # - If 2+ files: extract index 1 (skip index 0, which is often metadata)
        file_index = 1 if len(tar_filenames) >= 2 else 0
        member = tar.getmember(tar_filenames[file_index])

        # If we selected a directory, find the first actual file instead
        if member.isdir():
            file_index = None
            for i, name in enumerate(tar_filenames):
                m = tar.getmember(name)
                if not m.isdir():
                    file_index = i
                    break
            if file_index is None:
                raise ValueError(
                    "Tar archive contains only directories, no files to extract"
                )

        extracted_file = tar.extractfile(tar_filenames[file_index])
        if extracted_file is None:
            raise ValueError(
                f"Failed to extract file '{tar_filenames[file_index]}' from tar archive"
            )

        return extracted_file.read()


def fetch_object(
    path: str, extract: bool = False
) -> list[Any] | dict[str, Any] | bytes:
    """
    Fetch the contents of the given Rush object store path directly into memory.

    Be careful: if the contents are too large, they might not fit into memory.

    Args:
        path: The Rush object store path to fetch.
        extract: Automatically extract tar.zst archives in memory before returning.

    Returns:
        the data from the object store path, as a dict (JSON objects) or bytes (bin objects)
    """
    # TODO: enforce UUID type
    query = gql("""
        query GetObject($path: String!) {
            object_path(path: $path) {
                url
                object {
                    format
                    size
                }
            }
        }
    """)
    query.variable_values = {"path": path}

    result = _get_client().execute(query)
    obj_descriptor = result["object_path"]

    # Json
    if "contents" in obj_descriptor:
        contents = obj_descriptor["contents"]
        if not isinstance(contents, (list, dict)):
            raise TypeError(f"Expected JSON contents for {path}, got {type(contents)}")
        return contents
    # Bin
    elif "url" in obj_descriptor:
        response = requests.get(obj_descriptor["url"])
        response.raise_for_status()
        data = response.content
        return _extract_object_archive(data) if extract else data

    raise Exception(f"Object at path {path} has neither contents nor URL")


def _fetch_json_object(path: str, extract: bool = False) -> list[Any] | dict[str, Any]:
    raw = fetch_object(path, extract)
    if isinstance(raw, (str, bytes, bytearray)):
        raw = json.loads(raw)
    if not isinstance(raw, (list, dict)):
        raise TypeError(f"Expected JSON object for {path}, got {type(raw)}")
    return raw


def _fetch_json_object_as_list(path: str, extract: bool = False) -> list[Any]:
    raw = fetch_object(path, extract)
    if isinstance(raw, (str, bytes, bytearray)):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise TypeError(f"Expected JSON list object for {path}, got {type(raw)}")
    return raw


def _fetch_json_object_as_dict(path: str, extract: bool = False) -> dict[str, Any]:
    raw = fetch_object(path, extract)
    if isinstance(raw, (str, bytes, bytearray)):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected JSON dict object for {path}, got {type(raw)}")
    return raw


def _fetch_bin_object(path: str, extract: bool = False) -> bytes:
    raw = fetch_object(path, extract)
    if not isinstance(raw, (str, bytes, bytearray)):
        raise TypeError(f"Expected Bin object for {path}, got {type(raw)}")
    if isinstance(raw, str):
        raw = raw.encode()
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    return raw


def save_object(
    path: str,
    filepath: Path | str | None = None,
    name: str | None = None,
    type: Literal["json", "bin"] | None = None,
    ext: str | None = None,
    extract: bool = False,
) -> Path:
    """Save a Rush object store path to the workspace.

    Prefer :meth:`RushObject.save` when you have a ``RushObject``.
    This function infers the format from the *type* parameter.
    """
    if type is None:
        type = "json"
    format: Literal["Json", "Bin"] = "Json" if type == "json" else "Bin"
    obj = RushObject(path=ObjectID(path), size=0, format=format)
    return obj.save(filepath=filepath, name=name, ext=ext, extract=extract)


def _clean_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _clean_dict(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [_clean_dict(v) for v in d]
    return d


def _json_content_name(prefix: str, d: dict[str, Any]) -> str:
    payload = json.dumps(_clean_dict(d), sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_OID, payload)}"


def save_json(
    d: dict[str, Any],
    filepath: Path | str | None = None,
    name: str | None = None,
) -> Path:
    """
    Save a JSON file into the workspace folder.
    Convenient for saving non-object JSON output from a module run alongside
    the object outputs.
    """
    if filepath is not None and name is None:
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
    elif filepath is None and name is not None:
        filepath = _get_config().workspace_dir / _get_project_id() / f"{name}.json"
    else:
        raise Exception("Must specify either filepath or name")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        json.dump(_clean_dict(d), f, indent=2)
    return filepath


@singledispatch
def _to_topology_vobj(item: Any) -> dict[str, Any]:
    raise NotImplementedError(f"Cannot convert {type(item)} to a Topology vobj!")


@_to_topology_vobj.register
def _(trc: TRC) -> dict[str, Any]:
    return upload_object(trc.topology.to_dict())


@_to_topology_vobj.register
def _(trc_ref: TRCRef) -> dict[str, Any]:
    return trc_ref.topology.to_dict()


@_to_topology_vobj.register
def _(path: Path | str) -> dict[str, Any]:
    return upload_object(path)


@_to_topology_vobj.register
def _(object: RushObject) -> dict[str, Any]:
    return object.to_dict()


@_to_topology_vobj.register
def _(t: Topology) -> dict[str, Any]:
    return upload_object(t.to_dict())


@singledispatch
def _to_residues_vobj(item: Any) -> dict[str, Any]:
    raise NotImplementedError(f"Cannot convert {type(item)} to a Residues vobj!")


@_to_residues_vobj.register
def _(trc: TRC) -> dict[str, Any]:
    return upload_object(trc.residues.to_dict())


@_to_residues_vobj.register
def _(trc_ref: TRCRef) -> dict[str, Any]:
    return trc_ref.residues.to_dict()


@_to_residues_vobj.register
def _(path: Path | str) -> dict[str, Any]:
    return upload_object(path)


@_to_residues_vobj.register
def _(object: RushObject) -> dict[str, Any]:
    return object.to_dict()


@_to_residues_vobj.register
def _(r: Residues) -> dict[str, Any]:
    return upload_object(r.to_dict())


@singledispatch
def _to_chains_vobj(item: Any) -> dict[str, Any]:
    raise NotImplementedError(f"Cannot convert {type(item)} to a Chains vobj!")


@_to_chains_vobj.register
def _(trc: TRC) -> dict[str, Any]:
    return upload_object(trc.chains.to_dict())


@_to_chains_vobj.register
def _(trc_ref: TRCRef) -> dict[str, Any]:
    return trc_ref.chains.to_dict()


@_to_chains_vobj.register
def _(path: Path | str) -> dict[str, Any]:
    return upload_object(path)


@_to_chains_vobj.register
def _(object: RushObject) -> dict[str, Any]:
    return object.to_dict()


@_to_chains_vobj.register
def _(c: Chains) -> dict[str, Any]:
    return upload_object(c.to_dict())
