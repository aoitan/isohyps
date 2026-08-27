"""Versioned, deterministic contract for ``machine_index.json``.

The machine analysis pipeline keeps a richer, history-aware result for its
internal reports.  This module defines the smaller public projection that can
be consumed by downstream tools without exposing those implementation
details.
"""

from __future__ import annotations

import copy
import heapq
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


MACHINE_INDEX_SCHEMA_VERSION = "1.0"
MACHINE_INDEX_SCHEMA_MAJOR = 1

MACHINE_INDEX_TOP_LEVEL_FIELDS = (
    "schema_version",
    "files",
    "dependency_graph",
    "dependency_order",
)

MACHINE_INDEX_FILE_FIELDS = (
    "path",
    "hash",
    "size",
    "language",
    "kind",
    "public_symbols",
    "internal_symbols",
    "fan_in",
    "fan_out",
)

_SCHEMA_VERSION_PATTERN = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_SENTINELS = frozenset({"binary_skipped", "error"})
_FILE_KINDS = frozenset({"source", "test", "config", "doc", "other"})
_MISSING = object()


class MachineIndexContractError(ValueError):
    """Raised when a machine index violates its public contract."""


def _fail(location: str, reason: str) -> None:
    raise MachineIndexContractError(f"{location}: {reason}")


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(location, "expected an object")
    _validate_object_keys(value, location)
    return value


def _validate_object_keys(value: Mapping[Any, Any], location: str) -> None:
    for key in value:
        if not isinstance(key, str):
            _fail(location, "object keys must be strings")


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(location, "expected an array")
    return value


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    value = mapping.get(key, _MISSING)
    if value is _MISSING:
        _fail(f"{location}.{key}", "required field is missing")
    return value


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        _fail(location, "expected a string")
    return value


def _require_non_negative_integer(value: Any, location: str) -> int:
    if type(value) is not int or value < 0:
        _fail(location, "expected a non-negative integer")
    return value


def _validate_path(value: Any, location: str) -> str:
    path = _require_string(value, location)

    # ``machine_index.json`` uses repository-relative POSIX paths.  Reject
    # forms which PurePosixPath would silently normalize, since accepting them
    # would make two spellings refer to the same file identity.
    if (
        not path
        or "\\" in path
        or "\x00" in path
        or path.startswith("/")
        or PurePosixPath(path).is_absolute()
        or PureWindowsPath(path).drive
    ):
        _fail(location, "expected a repository-relative POSIX path")

    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        _fail(location, "path must not contain empty, '.' or '..' segments")

    if PurePosixPath(path).as_posix() != path:
        _fail(location, "path is not normalized")

    return path


def _validate_hash(value: Any, location: str) -> None:
    value = _require_string(value, location)
    if value not in _HASH_SENTINELS and _SHA256_PATTERN.fullmatch(value) is None:
        _fail(location, "expected a lowercase SHA-256 digest or a supported sentinel")


def _validate_string_array(value: Any, location: str) -> list[Any]:
    values = _require_list(value, location)
    for index, item in enumerate(values):
        _require_string(item, f"{location}[{index}]")
    return values


def _parse_schema_version(value: Any, location: str) -> tuple[int, int]:
    version = _require_string(value, location)
    match = _SCHEMA_VERSION_PATTERN.fullmatch(version)
    if match is None:
        _fail(location, "expected '<major>.<minor>' version string")
    return int(match.group("major")), int(match.group("minor"))


def _validate_file_entry(entry: Any, index: int) -> dict[str, Any]:
    location = f"files[{index}]"
    file_entry = _require_mapping(entry, location)

    path = _validate_path(_required(file_entry, "path", location), f"{location}.path")
    _validate_hash(_required(file_entry, "hash", location), f"{location}.hash")
    _require_non_negative_integer(
        _required(file_entry, "size", location), f"{location}.size"
    )
    _require_string(_required(file_entry, "language", location), f"{location}.language")

    kind = _require_string(_required(file_entry, "kind", location), f"{location}.kind")
    if kind not in _FILE_KINDS:
        _fail(f"{location}.kind", f"unsupported file kind: {kind!r}")

    _validate_string_array(
        _required(file_entry, "public_symbols", location),
        f"{location}.public_symbols",
    )
    _validate_string_array(
        _required(file_entry, "internal_symbols", location),
        f"{location}.internal_symbols",
    )
    _require_non_negative_integer(
        _required(file_entry, "fan_in", location), f"{location}.fan_in"
    )
    _require_non_negative_integer(
        _required(file_entry, "fan_out", location), f"{location}.fan_out"
    )

    return {"path": path, "kind": kind, "language": file_entry["language"]}


def _expected_dependency_order(graph: Mapping[str, list[str]]) -> list[str]:
    """Reproduce the producer's dependency-first order and cycle fallback."""

    dependents = {path: [] for path in graph}
    for source, dependencies in graph.items():
        for dependency in dependencies:
            dependents[dependency].append(source)

    in_degree = {path: 0 for path in dependents}
    for paths in dependents.values():
        for path in paths:
            in_degree[path] += 1

    queue = [path for path, degree in in_degree.items() if degree == 0]
    heapq.heapify(queue)
    sorted_dependents = {
        path: sorted(path_dependents) for path, path_dependents in dependents.items()
    }
    order: list[str] = []
    while queue:
        path = heapq.heappop(queue)
        order.append(path)
        for dependent in sorted_dependents[path]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                heapq.heappush(queue, dependent)

    order_set = set(order)
    order.extend(sorted(path for path in graph if path not in order_set))
    return order


def validate_machine_index(
    data: Mapping[str, Any], *, supported_major: int = MACHINE_INDEX_SCHEMA_MAJOR
) -> None:
    """Validate a machine index and raise on contract violations.

    Unknown fields are intentionally ignored.  They are the reader-side
    forward-compatibility boundary for a supported schema major; required v1
    fields and their semantics remain strict.
    """

    if type(supported_major) is not int or supported_major < 0:
        raise ValueError("supported_major must be a non-negative integer")

    root = _require_mapping(data, "root")
    major, _minor = _parse_schema_version(
        _required(root, "schema_version", "root"), "schema_version"
    )
    if major != supported_major:
        _fail(
            "schema_version",
            f"unsupported schema major {major}; expected {supported_major}",
        )

    files = _require_list(_required(root, "files", "root"), "files")
    file_descriptions: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    file_paths: list[str] = []
    for index, entry in enumerate(files):
        description = _validate_file_entry(entry, index)
        path = description["path"]
        if path in seen_paths:
            _fail(f"files[{index}].path", f"duplicate path: {path!r}")
        seen_paths.add(path)
        file_paths.append(path)
        file_descriptions.append(description)

    if file_paths != sorted(file_paths):
        _fail("files", "entries must be ordered by path")

    graph_value = _required(root, "dependency_graph", "root")
    graph_object = _require_mapping(graph_value, "dependency_graph")
    graph: dict[str, list[str]] = {}
    for source, value in graph_object.items():
        source_path = _validate_path(source, f"dependency_graph[{source!r}]")
        targets = _require_list(value, f"dependency_graph[{source!r}]")
        validated_targets: list[str] = []
        target_set: set[str] = set()
        for target_index, target in enumerate(targets):
            target_path = _validate_path(
                target, f"dependency_graph[{source!r}][{target_index}]"
            )
            if target_path in target_set:
                _fail(
                    f"dependency_graph[{source!r}]",
                    f"duplicate dependency: {target_path!r}",
                )
            target_set.add(target_path)
            validated_targets.append(target_path)

        if validated_targets != sorted(validated_targets):
            _fail(
                f"dependency_graph[{source!r}]",
                "dependencies must be ordered by path",
            )
        if source_path in target_set:
            _fail(
                f"dependency_graph[{source!r}]",
                "self-dependencies are not allowed",
            )
        graph[source_path] = validated_targets

    expected_graph_paths = {
        description["path"]
        for description in file_descriptions
        if description["kind"] == "source" and description["language"] != "unknown"
    }
    actual_graph_paths = set(graph)
    if actual_graph_paths != expected_graph_paths:
        missing = sorted(expected_graph_paths - actual_graph_paths)
        extra = sorted(actual_graph_paths - expected_graph_paths)
        _fail(
            "dependency_graph",
            f"keys do not match analyzable source files (missing={missing!r}, extra={extra!r})",
        )

    for source, targets in graph.items():
        for target_index, target in enumerate(targets):
            if target not in actual_graph_paths:
                _fail(
                    f"dependency_graph[{source!r}][{target_index}]",
                    f"unknown dependency path: {target!r}",
                )

    order = _require_list(_required(root, "dependency_order", "root"), "dependency_order")
    order_paths: list[str] = []
    order_seen: set[str] = set()
    for index, value in enumerate(order):
        path = _validate_path(value, f"dependency_order[{index}]")
        if path in order_seen:
            _fail(f"dependency_order[{index}]", f"duplicate path: {path!r}")
        order_seen.add(path)
        order_paths.append(path)

    if order_seen != actual_graph_paths:
        _fail(
            "dependency_order",
            "must contain every dependency graph key exactly once",
        )

    expected_order = _expected_dependency_order(graph)
    if order_paths != expected_order:
        _fail(
            "dependency_order",
            "does not match the deterministic dependency-first order and cycle fallback",
        )

    fan_in = {path: 0 for path in file_paths}
    for source, targets in graph.items():
        for target in targets:
            fan_in[target] += 1

    for index, entry in enumerate(files):
        path = file_paths[index]
        expected_fan_out = len(graph[path]) if path in graph else 0
        actual_fan_out = entry["fan_out"]
        if actual_fan_out != expected_fan_out:
            _fail(
                f"files[{index}].fan_out",
                f"expected {expected_fan_out}, got {actual_fan_out}",
            )
        expected_fan_in = fan_in[path]
        actual_fan_in = entry["fan_in"]
        if actual_fan_in != expected_fan_in:
            _fail(
                f"files[{index}].fan_in",
                f"expected {expected_fan_in}, got {actual_fan_in}",
            )


def build_machine_index_v1(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Build a v1 allowlist projection without mutating ``analysis``."""

    source = _require_mapping(analysis, "analysis")
    source_files = _require_list(_required(source, "files", "analysis"), "analysis.files")

    files: list[dict[str, Any]] = []
    for index, source_entry in enumerate(source_files):
        entry = _require_mapping(source_entry, f"analysis.files[{index}]")
        path = _validate_path(
            _required(entry, "path", f"analysis.files[{index}]"),
            f"analysis.files[{index}].path",
        )
        projected = {
            field: copy.deepcopy(
                _required(entry, field, f"analysis.files[{index}]")
            )
            for field in MACHINE_INDEX_FILE_FIELDS
        }
        # Validate before sorting so malformed paths fail with a contract
        # location instead of an incidental Python sorting error.
        projected["path"] = path
        files.append(projected)
    files.sort(key=lambda entry: entry["path"])

    source_graph = _require_mapping(
        _required(source, "dependency_graph", "analysis"),
        "analysis.dependency_graph",
    )
    dependency_graph: dict[str, list[str]] = {}
    for source_path, targets_value in source_graph.items():
        source_path = _validate_path(
            source_path, f"analysis.dependency_graph[{source_path!r}]"
        )
        targets = _require_list(
            targets_value, f"analysis.dependency_graph[{source_path!r}]"
        )
        copied_targets = [
            _validate_path(
                copy.deepcopy(target),
                f"analysis.dependency_graph[{source_path!r}][{index}]",
            )
            for index, target in enumerate(targets)
        ]
        dependency_graph[source_path] = sorted(copied_targets)
    dependency_graph = {
        path: dependency_graph[path] for path in sorted(dependency_graph)
    }

    source_order = _require_list(
        _required(source, "dependency_order", "analysis"),
        "analysis.dependency_order",
    )
    dependency_order = [
        _validate_path(copy.deepcopy(path), f"analysis.dependency_order[{index}]")
        for index, path in enumerate(source_order)
    ]

    projected_index = {
        "schema_version": MACHINE_INDEX_SCHEMA_VERSION,
        "files": files,
        "dependency_graph": dependency_graph,
        "dependency_order": dependency_order,
    }
    validate_machine_index(projected_index)
    return projected_index


def serialize_machine_index(data: Mapping[str, Any]) -> str:
    """Serialize a valid index with the canonical UTF-8 JSON representation."""

    validate_machine_index(data)
    try:
        encoded = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MachineIndexContractError(f"root: cannot serialize JSON: {exc}") from exc
    return encoded + "\n"


def load_machine_index(
    path: Path, *, supported_major: int = MACHINE_INDEX_SCHEMA_MAJOR
) -> dict[str, Any]:
    """Load and validate a UTF-8 machine index from ``path``."""

    source_path = Path(path)
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MachineIndexContractError(
            f"{source_path}: cannot read valid UTF-8 JSON: {exc}"
        ) from exc

    validate_machine_index(data, supported_major=supported_major)
    return data


def write_machine_index(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically write a canonical machine index beside its destination."""

    destination = Path(path)
    payload = serialize_machine_index(data)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


# Keep an explicitly named alias available to callers that want to emphasize
# the atomicity guarantee at the call site.
write_machine_index_atomic = write_machine_index


__all__ = [
    "MACHINE_INDEX_FILE_FIELDS",
    "MACHINE_INDEX_SCHEMA_MAJOR",
    "MACHINE_INDEX_SCHEMA_VERSION",
    "MACHINE_INDEX_TOP_LEVEL_FIELDS",
    "MachineIndexContractError",
    "build_machine_index_v1",
    "load_machine_index",
    "serialize_machine_index",
    "validate_machine_index",
    "write_machine_index",
    "write_machine_index_atomic",
]
