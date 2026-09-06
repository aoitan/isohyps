"""Versioned, deterministic contract for ``machine_index.json``.

The machine analysis pipeline keeps a richer, history-aware result for its
internal reports.  This module defines the smaller public projection that can
be consumed by downstream tools without exposing those implementation
details.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import os
import re
import stat as stat_module
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from isohyps.attention import AttentionContractError, validate_attention
from isohyps.doc_freshness import (
    DOC_FRESHNESS_SCHEMA_MAJOR,
    DocFreshnessContractError,
    canonical_doc_freshness_bytes,
    validate_doc_freshness,
)


MACHINE_INDEX_SCHEMA_VERSION = "1.0"
MACHINE_INDEX_SCHEMA_MAJOR = 1
MACHINE_INDEX_V2_LEGACY_SCHEMA_VERSION = "2.0"
MACHINE_INDEX_V2_SCHEMA_VERSION = "2.1"

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
MACHINE_INDEX_DOC_FRESHNESS_FIELDS = (
    "path",
    "schema_version",
    "sha256",
    "counts",
)
MACHINE_INDEX_FRESHNESS_COUNT_FIELDS = ("missing", "fresh", "stale", "unknown")

_SCHEMA_VERSION_PATTERN = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_SENTINELS = frozenset({"binary_skipped", "error"})
_FILE_KINDS = frozenset({"source", "test", "config", "doc", "other"})
_MACHINE_INDEX_REFERENCE_CHUNK_SIZE = 1024 * 1024
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


def _validate_doc_freshness_counts(value: Any, location: str) -> dict[str, int]:
    counts = _require_mapping(value, location)
    if set(counts) != set(MACHINE_INDEX_FRESHNESS_COUNT_FIELDS):
        _fail(
            location,
            "expected exactly the freshness status count fields "
            f"{list(MACHINE_INDEX_FRESHNESS_COUNT_FIELDS)!r}",
        )
    return {
        status: _require_non_negative_integer(
            counts[status], f"{location}.{status}"
        )
        for status in MACHINE_INDEX_FRESHNESS_COUNT_FIELDS
    }


def _validate_doc_freshness_reference(
    value: Any, location: str = "doc_freshness"
) -> dict[str, Any]:
    reference = _require_mapping(value, location)
    path = _validate_path(
        _required(reference, "path", location), f"{location}.path"
    )
    schema_version = _require_string(
        _required(reference, "schema_version", location),
        f"{location}.schema_version",
    )
    freshness_major, _freshness_minor = _parse_schema_version(
        schema_version, f"{location}.schema_version"
    )
    if freshness_major != DOC_FRESHNESS_SCHEMA_MAJOR:
        _fail(
            f"{location}.schema_version",
            f"unsupported freshness schema major {freshness_major}; "
            f"expected {DOC_FRESHNESS_SCHEMA_MAJOR}",
        )

    sha256 = _require_string(
        _required(reference, "sha256", location), f"{location}.sha256"
    )
    if _SHA256_PATTERN.fullmatch(sha256) is None:
        _fail(
            f"{location}.sha256",
            "expected a lowercase 64-character SHA-256 digest",
        )
    counts = _validate_doc_freshness_counts(
        _required(reference, "counts", location), f"{location}.counts"
    )
    return {
        "path": path,
        "schema_version": schema_version,
        "sha256": sha256,
        "counts": counts,
    }


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
    major, minor = _parse_schema_version(
        _required(root, "schema_version", "root"), "schema_version"
    )
    if major != supported_major:
        _fail(
            "schema_version",
            f"unsupported schema major {major}; expected {supported_major}",
        )

    if major not in (1, 2):
        _fail("schema_version", f"unsupported schema major {major}")

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

    if major == 2:
        attention = _require_list(_required(root, "attention", "root"), "attention")
        try:
            validate_attention(attention)
        except AttentionContractError as exc:
            _fail("attention", str(exc))
        if minor >= 1:
            _validate_doc_freshness_reference(
                _required(root, "doc_freshness", "root")
            )

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


def build_machine_index_v2(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Build the v2.1 projection with attention and freshness reference."""

    projected = build_machine_index_v1(analysis)
    source = _require_mapping(analysis, "analysis")
    attention = copy.deepcopy(
        _require_list(_required(source, "attention", "analysis"), "analysis.attention")
    )
    try:
        validate_attention(attention)
    except AttentionContractError as exc:
        _fail("analysis.attention", str(exc))
    projected["schema_version"] = MACHINE_INDEX_V2_SCHEMA_VERSION
    projected["attention"] = attention
    freshness = _require_mapping(
        _required(source, "doc_freshness", "analysis"),
        "analysis.doc_freshness",
    )
    projected["doc_freshness"] = {
        field: copy.deepcopy(
            _required(freshness, field, "analysis.doc_freshness")
        )
        for field in MACHINE_INDEX_DOC_FRESHNESS_FIELDS
    }
    validate_machine_index(projected, supported_major=2)
    return projected


def serialize_machine_index(
    data: Mapping[str, Any], *, supported_major: int = MACHINE_INDEX_SCHEMA_MAJOR
) -> str:
    """Serialize a valid index with the canonical UTF-8 JSON representation."""

    validate_machine_index(data, supported_major=supported_major)
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


def _resolve_freshness_reference_path(index_path: Path, reference_path: str) -> Path:
    """Resolve a freshness reference without leaving the index directory."""

    index_directory = Path(index_path).resolve().parent
    candidate = index_directory / PurePosixPath(reference_path)
    try:
        candidate.resolve(strict=False).relative_to(index_directory)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MachineIndexContractError(
            "doc_freshness.path: reference escapes the machine index directory"
        ) from exc

    # A lexical path check is not sufficient when a reference or one of its
    # parent directories is a symlink.  Reject the whole path rather than
    # resolving it to an output outside the index directory.
    current = candidate
    while True:
        if current.is_symlink():
            raise MachineIndexContractError(
                "doc_freshness.path: symlink references are not allowed"
            )
        if current == index_directory:
            break
        parent = current.parent
        if parent == current:
            raise MachineIndexContractError(
                "doc_freshness.path: reference is not contained by the index directory"
            )
        current = parent

    if not candidate.exists():
        raise MachineIndexContractError(
            f"doc_freshness.path: referenced artifact does not exist: {reference_path!r}"
        )
    if not candidate.is_file():
        raise MachineIndexContractError(
            f"doc_freshness.path: referenced artifact is not a regular file: {reference_path!r}"
        )
    return candidate


def _reference_file_identity(stat_result: os.stat_result) -> tuple[Any, ...]:
    """Return metadata used to detect reference replacement or mutation."""

    return (
        getattr(stat_result, "st_dev", None),
        getattr(stat_result, "st_ino", None),
        getattr(stat_result, "st_size", None),
        getattr(stat_result, "st_mtime_ns", None),
    )


def _read_freshness_reference_bytes(
    index_path: Path, freshness_path: Path
) -> bytes:
    """Read a freshness reference through no-follow directory descriptors.

    The path checks in ``_resolve_freshness_reference_path`` are useful for
    diagnostics, but they cannot make a later path-based read race-free.  Keep
    the index directory and every parent component open, reject symlinks while
    opening, and compare visible identities before returning the bytes.  A
    non-blocking final open also prevents a raced FIFO from hanging the
    resolver before its non-regular type can be rejected.
    """

    index_directory = Path(index_path).resolve().parent
    try:
        relative_path = Path(freshness_path).relative_to(index_directory)
    except ValueError as exc:
        raise MachineIndexContractError(
            "doc_freshness.path: referenced artifact is not contained by the "
            "machine index directory"
        ) from exc

    parts = relative_path.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise MachineIndexContractError(
            "doc_freshness.path: referenced artifact has an unsafe relative path"
        )

    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    nonblock_flag = getattr(os, "O_NONBLOCK", 0)
    if (
        os.open not in supports_dir_fd
        or os.stat not in supports_dir_fd
        or not directory_flag
        or not nofollow_flag
        or not nonblock_flag
    ):
        raise MachineIndexContractError(
            "doc_freshness.path: safe descriptor-relative reference reads are "
            "unavailable on this platform"
        )

    directory_flags = (
        os.O_RDONLY
        | directory_flag
        | nofollow_flag
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | nofollow_flag
        | nonblock_flag
        | getattr(os, "O_CLOEXEC", 0)
    )

    descriptors: list[int] = []
    try:
        try:
            root_fd = os.open(index_directory, directory_flags)
            descriptors.append(root_fd)
            root_identity = _reference_file_identity(os.fstat(root_fd))
            parent_fd = os.dup(root_fd)
            descriptors.append(parent_fd)
        except OSError as exc:
            raise MachineIndexContractError(
                "doc_freshness.path: cannot open the machine index directory "
                "safely"
            ) from exc

        for component in parts[:-1]:
            try:
                before_stat = os.stat(
                    component,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError as exc:
                raise MachineIndexContractError(
                    f"doc_freshness.path: referenced artifact does not exist: "
                    f"{freshness_path.name!r}"
                ) from exc
            except OSError as exc:
                raise MachineIndexContractError(
                    "doc_freshness.path: cannot inspect the referenced artifact "
                    "parent safely"
                ) from exc
            if stat_module.S_ISLNK(before_stat.st_mode):
                raise MachineIndexContractError(
                    "doc_freshness.path: symlink references are not allowed"
                )
            if not stat_module.S_ISDIR(before_stat.st_mode):
                raise MachineIndexContractError(
                    "doc_freshness.path: reference parent is not a directory"
                )

            next_fd: int | None = None
            try:
                next_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_fd,
                )
                opened_stat = os.fstat(next_fd)
                if _reference_file_identity(before_stat) != _reference_file_identity(
                    opened_stat
                ):
                    raise MachineIndexContractError(
                        "doc_freshness.path: reference parent changed during read"
                    )
            except MachineIndexContractError:
                if next_fd is not None:
                    os.close(next_fd)
                raise
            except OSError as exc:
                if next_fd is not None:
                    os.close(next_fd)
                raise MachineIndexContractError(
                    "doc_freshness.path: reference parent cannot be opened safely"
                ) from exc

            descriptors.append(next_fd)
            descriptors.remove(parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd

        final_name = parts[-1]
        try:
            before_file_stat = os.stat(
                final_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise MachineIndexContractError(
                f"doc_freshness.path: referenced artifact does not exist: "
                f"{freshness_path.name!r}"
            ) from exc
        except OSError as exc:
            raise MachineIndexContractError(
                "doc_freshness.path: cannot inspect the referenced artifact safely"
            ) from exc
        if stat_module.S_ISLNK(before_file_stat.st_mode):
            raise MachineIndexContractError(
                "doc_freshness.path: symlink references are not allowed"
            )
        if not stat_module.S_ISREG(before_file_stat.st_mode):
            raise MachineIndexContractError(
                "doc_freshness.path: referenced artifact is not a regular file"
            )

        try:
            file_fd = os.open(final_name, file_flags, dir_fd=parent_fd)
            descriptors.append(file_fd)
            opened_file_stat = os.fstat(file_fd)
        except OSError as exc:
            raise MachineIndexContractError(
                "doc_freshness.path: referenced artifact cannot be opened safely"
            ) from exc
        if not stat_module.S_ISREG(opened_file_stat.st_mode):
            raise MachineIndexContractError(
                "doc_freshness.path: referenced artifact is not a regular file"
            )
        if _reference_file_identity(before_file_stat) != _reference_file_identity(
            opened_file_stat
        ):
            raise MachineIndexContractError(
                "doc_freshness.path: referenced artifact changed during read"
            )

        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(file_fd, _MACHINE_INDEX_REFERENCE_CHUNK_SIZE)
            except OSError as exc:
                raise MachineIndexContractError(
                    "doc_freshness.path: cannot read referenced artifact safely"
                ) from exc
            if not chunk:
                break
            chunks.append(chunk)

        after_file_stat = os.fstat(file_fd)
        if _reference_file_identity(opened_file_stat) != _reference_file_identity(
            after_file_stat
        ):
            raise MachineIndexContractError(
                "doc_freshness.path: referenced artifact changed during read"
            )

        try:
            visible_file_stat = os.stat(
                final_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except (FileNotFoundError, OSError) as exc:
            raise MachineIndexContractError(
                "doc_freshness.path: referenced artifact changed during read"
            ) from exc
        if (
            stat_module.S_ISLNK(visible_file_stat.st_mode)
            or not stat_module.S_ISREG(visible_file_stat.st_mode)
            or _reference_file_identity(visible_file_stat)
            != _reference_file_identity(after_file_stat)
        ):
            raise MachineIndexContractError(
                "doc_freshness.path: referenced artifact changed during read"
            )

        try:
            visible_root_stat = os.stat(index_directory, follow_symlinks=False)
        except OSError as exc:
            raise MachineIndexContractError(
                "doc_freshness.path: machine index directory changed during read"
            ) from exc
        if (
            not stat_module.S_ISDIR(visible_root_stat.st_mode)
            or _reference_file_identity(visible_root_stat) != root_identity
            or _reference_file_identity(os.fstat(root_fd)) != root_identity
        ):
            raise MachineIndexContractError(
                "doc_freshness.path: machine index directory changed during read"
            )

        # Re-run the visible-path checks after reading.  The descriptor keeps
        # the read itself on the originally opened, no-follow path; this
        # second check turns any parent replacement into a fail-closed result.
        try:
            _resolve_freshness_reference_path(
                index_path, relative_path.as_posix()
            )
        except MachineIndexContractError as exc:
            raise MachineIndexContractError(
                "doc_freshness.path: reference path changed during read"
            ) from exc

        return b"".join(chunks)
    except MachineIndexContractError:
        raise
    except OSError as exc:
        raise MachineIndexContractError(
            "doc_freshness.path: cannot read referenced artifact safely"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def resolve_machine_index_freshness(
    index_path: Path,
    *,
    index_data: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load the freshness artifact bound to a v2.1 machine index.

    A v2.0 index is still a valid readable index, but it predates the
    freshness reference and therefore returns ``None``.  For v2.1 and later,
    the referenced file must be contained by the index directory and its
    exact bytes, freshness schema version, and status counts must all match
    the reference before the document is returned.
    """

    source_path = Path(index_path)
    if index_data is None:
        index = load_machine_index(source_path, supported_major=2)
    else:
        validate_machine_index(index_data, supported_major=2)
        index = dict(index_data)

    _major, minor = _parse_schema_version(
        _required(index, "schema_version", "root"), "schema_version"
    )
    if minor < 1:
        return None

    reference = _validate_doc_freshness_reference(
        _required(index, "doc_freshness", "root")
    )
    freshness_path = _resolve_freshness_reference_path(
        source_path, reference["path"]
    )
    payload = _read_freshness_reference_bytes(source_path, freshness_path)

    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != reference["sha256"]:
        raise MachineIndexContractError(
            "doc_freshness.sha256: referenced artifact digest does not match "
            f"(expected {reference['sha256']}, got {actual_digest})"
        )

    try:
        freshness_data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MachineIndexContractError(
            "doc_freshness.path: referenced artifact is not valid UTF-8 JSON"
        ) from exc

    try:
        validate_doc_freshness(freshness_data)
        canonical_payload = canonical_doc_freshness_bytes(freshness_data)
    except (DocFreshnessContractError, TypeError, ValueError) as exc:
        raise MachineIndexContractError(
            f"doc_freshness.path: referenced artifact violates its contract: {exc}"
        ) from exc

    if canonical_payload != payload:
        raise MachineIndexContractError(
            "doc_freshness.path: referenced artifact is not canonical JSON"
        )
    if freshness_data["schema_version"] != reference["schema_version"]:
        raise MachineIndexContractError(
            "doc_freshness.schema_version: reference does not match the "
            "referenced artifact"
        )
    if freshness_data["counts"] != reference["counts"]:
        raise MachineIndexContractError(
            "doc_freshness.counts: reference does not match the referenced artifact"
        )
    return freshness_data


# Explicit aliases keep the resolver discoverable for callers that name the
# relationship as either a machine-index or document-freshness operation.
resolve_machine_index_doc_freshness = resolve_machine_index_freshness
load_machine_index_doc_freshness = resolve_machine_index_freshness


def write_machine_index(
    path: Path,
    data: Mapping[str, Any],
    *,
    supported_major: int = MACHINE_INDEX_SCHEMA_MAJOR,
) -> None:
    """Atomically write a canonical machine index beside its destination."""

    destination = Path(path)
    payload = serialize_machine_index(data, supported_major=supported_major)
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
    "MACHINE_INDEX_DOC_FRESHNESS_FIELDS",
    "MACHINE_INDEX_FILE_FIELDS",
    "MACHINE_INDEX_FRESHNESS_COUNT_FIELDS",
    "MACHINE_INDEX_SCHEMA_MAJOR",
    "MACHINE_INDEX_SCHEMA_VERSION",
    "MACHINE_INDEX_TOP_LEVEL_FIELDS",
    "MACHINE_INDEX_V2_LEGACY_SCHEMA_VERSION",
    "MACHINE_INDEX_V2_SCHEMA_VERSION",
    "MachineIndexContractError",
    "build_machine_index_v1",
    "build_machine_index_v2",
    "load_machine_index",
    "load_machine_index_doc_freshness",
    "resolve_machine_index_doc_freshness",
    "resolve_machine_index_freshness",
    "serialize_machine_index",
    "validate_machine_index",
    "write_machine_index",
    "write_machine_index_atomic",
]
