"""Versioned contracts for document freshness and producer provenance.

This module owns the shape and persistence boundary for the two artifacts used
by the document-freshness pipeline:

* ``doc_freshness.json`` records the evidence and result for every source
  target; and
* ``doc_provenance.json`` records the source/doc byte relationship asserted by
  a document producer.

File observation and mapping resolution live here as deterministic, fail-closed
boundaries.  Freshness evaluation remains a later pipeline component.  Keeping
the boundaries independent means a consumer can validate and reproduce the
artifact without depending on filesystem timestamps or analyzer internals.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat as stat_module
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, TypedDict


DOC_FRESHNESS_SCHEMA_VERSION = "1.0"
DOC_FRESHNESS_SCHEMA_MAJOR = 1
DOC_PROVENANCE_SCHEMA_VERSION = "1.0"
DOC_PROVENANCE_SCHEMA_MAJOR = 1

DOC_FRESHNESS_HASH_ALGORITHM = "sha256"
DOC_FRESHNESS_MAPPING_RULE = "source-suffix-and-replaced-suffix-v1"
DOC_PROVENANCE_PRODUCER = "isohyps.project_analysis"
DOC_PROVENANCE_PRODUCER_VERSION = "1"

DOC_FRESHNESS_TOP_LEVEL_FIELDS = (
    "schema_version",
    "hash_algorithm",
    "mapping_rule",
    "counts",
    "entries",
    "diagnostics",
)
DOC_FRESHNESS_ENTRY_FIELDS = (
    "source_path",
    "doc_path",
    "expected_doc_paths",
    "current_source_hash",
    "recorded_source_hash",
    "doc_hash",
    "status",
    "reason",
    "diagnostics",
)
DOC_PROVENANCE_TOP_LEVEL_FIELDS = (
    "schema_version",
    "hash_algorithm",
    "assertions",
)
DOC_PROVENANCE_ASSERTION_FIELDS = (
    "source_path",
    "doc_path",
    "recorded_source_hash",
    "recorded_doc_hash",
    "producer",
    "producer_version",
)


FreshnessStatus = Literal["missing", "fresh", "stale", "unknown"]
FreshnessReason = Literal[
    "source_hash_match",
    "source_hash_mismatch",
    "doc_missing",
    "recorded_source_hash_missing",
    "ambiguous_doc_mapping",
    "source_hash_unavailable",
    "source_changed_during_scan",
    "doc_hash_unavailable",
    "doc_changed_during_scan",
    "unsafe_source_path",
    "unsafe_doc_path",
    "doc_identity_collision",
    "provenance_artifact_invalid",
    "provenance_assertion_invalid",
    "provenance_binding_mismatch",
    "provenance_doc_hash_mismatch",
    "invalid_source_identity",
    "duplicate_source_identity",
]


FRESHNESS_STATUSES: tuple[FreshnessStatus, ...] = (
    "missing",
    "fresh",
    "stale",
    "unknown",
)
FRESHNESS_REASONS: tuple[str, ...] = (
    "source_hash_match",
    "source_hash_mismatch",
    "doc_missing",
    "recorded_source_hash_missing",
    "ambiguous_doc_mapping",
    "source_hash_unavailable",
    "source_changed_during_scan",
    "doc_hash_unavailable",
    "doc_changed_during_scan",
    "unsafe_source_path",
    "unsafe_doc_path",
    "doc_identity_collision",
    "provenance_artifact_invalid",
    "provenance_assertion_invalid",
    "provenance_binding_mismatch",
    "provenance_doc_hash_mismatch",
    "invalid_source_identity",
    "duplicate_source_identity",
)

PROVENANCE_FAILURE_REASONS: tuple[str, ...] = (
    "recorded_source_hash_missing",
    "provenance_artifact_invalid",
    "provenance_assertion_invalid",
)

# Diagnostics are stable codes rather than free-form exception text.  The
# primary reason vocabulary is also useful as a secondary diagnostic, while
# this code is specific to an assertion which no longer has a current target.
DIAGNOSTIC_CODES: tuple[str, ...] = tuple(
    sorted((*FRESHNESS_REASONS, "orphan_provenance_assertion"))
)

_SCHEMA_VERSION_PATTERN = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()

HASH_CHUNK_SIZE = 1024 * 1024
_HASH_MAX_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class HashObservation:
    """The result of one safe exact-byte file-hash observation.

    ``digest`` is ``None`` whenever the path cannot be safely observed.  The
    reason values describe the observation boundary and are intentionally
    separate from the freshness artifact's source/doc-specific reason enum;
    the evaluator can map the same observation to either ``source_*`` or
    ``doc_*`` diagnostics without making the file reader guess its role.
    """

    digest: str | None
    reason: str | None = None
    attempts: int = 1

    @property
    def available(self) -> bool:
        """Whether a stable digest was obtained."""

        return self.digest is not None

    @property
    def hash(self) -> str | None:
        """Compatibility spelling for callers that call the value ``hash``."""

        return self.digest

    @property
    def sha256(self) -> str | None:
        """Explicit algorithm spelling for callers inspecting the result."""

        return self.digest


@dataclass(frozen=True, slots=True)
class _ReadObservation:
    """The result of a safe exact-byte file read."""

    payload: bytes | None
    reason: str | None = None
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class CoverageTarget:
    """The source identity consumed by the document mapping resolver."""

    source_path: str
    current_source_hash: Any = None
    doc_hash: Any = None

    def __init__(
        self,
        source_path: str | None = None,
        *,
        path: str | None = None,
        current_source_hash: Any = None,
        source_hash: Any = None,
        hash: Any = None,
        doc_hash: Any = None,
    ):
        if source_path is None:
            source_path = path
        elif path is not None and path != source_path:
            raise ValueError("source_path and path must identify the same target")
        if source_path is None:
            raise TypeError("CoverageTarget requires source_path or path")

        supplied_hashes = [
            value
            for value in (current_source_hash, source_hash, hash)
            if value is not None
        ]
        if supplied_hashes and any(value != supplied_hashes[0] for value in supplied_hashes[1:]):
            raise ValueError("current_source_hash, source_hash, and hash must agree")
        if current_source_hash is None and source_hash is not None:
            current_source_hash = source_hash
        if current_source_hash is None and hash is not None:
            current_source_hash = hash
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "current_source_hash", current_source_hash)
        object.__setattr__(self, "doc_hash", doc_hash)

    @property
    def path(self) -> str:
        """Alias matching the path key used by machine-analysis metadata."""

        return self.source_path

    @property
    def source_hash(self) -> Any:
        """Alias for the current source observation used by the evaluator."""

        return self.current_source_hash

    @property
    def hash(self) -> Any:
        """Compatibility alias for machine metadata's source ``hash`` key."""

        return self.current_source_hash


@dataclass(frozen=True, slots=True)
class DocMapping:
    """Deterministic mapping result for one source target."""

    source_path: str
    expected_doc_paths: tuple[str, str]
    doc_path: str | None
    existing_doc_paths: tuple[str, ...]
    reason: str | None = None
    diagnostics: tuple[str, ...] = ()
    doc_hash: Any = None
    current_source_hash: Any = None

    @property
    def selected_doc_path(self) -> str | None:
        """Alias used by callers that distinguish expected and selected paths."""

        return self.doc_path

    @property
    def candidates(self) -> tuple[str, str]:
        """Return the two candidates in mapping-rule order."""

        return self.expected_doc_paths


@dataclass(frozen=True, slots=True)
class MappingResolution:
    """All mapping results, sorted by source identity and globally diagnosed."""

    entries: tuple[DocMapping, ...]
    diagnostics: tuple[str, ...] = ()
    source_hashes: Mapping[str, Any] | None = None
    doc_hashes: Mapping[str, Any] | None = None

    @property
    def mappings(self) -> tuple[DocMapping, ...]:
        """Alias for code that names the result collection ``mappings``."""

        return self.entries

    @property
    def by_source_path(self) -> dict[str, DocMapping]:
        """Return a convenient source-path lookup without changing ownership."""

        return {entry.source_path: entry for entry in self.entries}

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, key: int | str) -> DocMapping:
        if isinstance(key, int):
            return self.entries[key]
        return self.by_source_path[key]


@dataclass(frozen=True, slots=True)
class _CandidateObservation:
    path: str
    state: Literal["missing", "regular", "unsafe", "unavailable"]
    reason: str | None = None


class DocFreshnessContractError(ValueError):
    """Raised when a freshness or provenance document violates its contract."""


class DocProvenanceContractError(DocFreshnessContractError):
    """More specific alias for callers validating provenance documents."""


class DocProvenanceRecordingError(DocProvenanceContractError):
    """Raised when a producer cannot safely create a provenance assertion."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        observation: HashObservation | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.observation = observation


class DocFreshnessEntry(TypedDict):
    source_path: str
    doc_path: str | None
    expected_doc_paths: list[str]
    current_source_hash: str | None
    recorded_source_hash: str | None
    doc_hash: str | None
    status: FreshnessStatus
    reason: FreshnessReason
    diagnostics: list[str]


class DocFreshnessDocument(TypedDict):
    schema_version: str
    hash_algorithm: str
    mapping_rule: str
    counts: dict[str, int]
    entries: list[DocFreshnessEntry]
    diagnostics: list[str]


class ProvenanceAssertion(TypedDict):
    source_path: str
    doc_path: str
    recorded_source_hash: str
    recorded_doc_hash: str
    producer: str
    producer_version: str


class DocProvenanceDocument(TypedDict):
    schema_version: str
    hash_algorithm: str
    assertions: list[ProvenanceAssertion]


@dataclass(frozen=True, slots=True)
class ProvenanceFailure:
    """A fail-closed provenance read result consumed by the evaluator."""

    reason: str
    diagnostics: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ProvenanceReadResult:
    """Non-throwing result of reading producer-owned provenance."""

    document: dict[str, Any] | None = None
    failure: ProvenanceFailure | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether a validated provenance document was read."""

        return self.document is not None and self.failure is None

    @property
    def available(self) -> bool:
        """Alias for callers that treat provenance as an optional observation."""

        return self.valid

    @property
    def reason(self) -> str | None:
        """Return the stable failure reason, if the read did not succeed."""

        return self.failure.reason if self.failure is not None else None

    @property
    def error(self) -> str | None:
        """Return diagnostic exception text without using it as a contract code."""

        return self.failure.error if self.failure is not None else None


def _fail(location: str, reason: str, *, provenance: bool = False) -> None:
    error_type = DocProvenanceContractError if provenance else DocFreshnessContractError
    raise error_type(f"{location}: {reason}")


def _require_mapping(value: Any, location: str, *, provenance: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(location, "expected an object", provenance=provenance)
    for key in value:
        if not isinstance(key, str):
            _fail(location, "object keys must be strings", provenance=provenance)
    return value


def _required(mapping: Mapping[str, Any], key: str, location: str, *, provenance: bool = False) -> Any:
    value = mapping.get(key, _MISSING)
    if value is _MISSING:
        _fail(f"{location}.{key}", "required field is missing", provenance=provenance)
    return value


def _require_list(value: Any, location: str, *, provenance: bool = False) -> list[Any]:
    if not isinstance(value, list):
        _fail(location, "expected an array", provenance=provenance)
    return value


def _require_string(
    value: Any,
    location: str,
    *,
    non_empty: bool = False,
    provenance: bool = False,
) -> str:
    if not isinstance(value, str):
        _fail(location, "expected a string", provenance=provenance)
    if non_empty and not value:
        _fail(location, "expected a non-empty string", provenance=provenance)
    return value


def _require_non_negative_integer(value: Any, location: str) -> int:
    if type(value) is not int or value < 0:
        _fail(location, "expected a non-negative integer")
    return value


def _parse_schema_version(
    value: Any,
    location: str,
    *,
    provenance: bool = False,
) -> tuple[int, int]:
    version = _require_string(value, location, provenance=provenance)
    match = _SCHEMA_VERSION_PATTERN.fullmatch(version)
    if match is None:
        _fail(location, "expected '<major>.<minor>' version string", provenance=provenance)
    return int(match.group("major")), int(match.group("minor"))


def validate_repository_relative_path(
    value: Any, *, location: str = "path", provenance: bool = False
) -> str:
    """Validate and return a normalized repository-relative POSIX path.

    The validator intentionally rejects spellings that a path library could
    silently normalize.  This keeps path strings suitable for use as stable
    artifact identities; filesystem containment and symlink policy are enforced
    by ``safe_hash_regular_file`` and the mapping resolver below.
    """

    path = _require_string(value, location, provenance=provenance)
    if (
        not path
        or "\\" in path
        or "\x00" in path
        or path.startswith("/")
        or PurePosixPath(path).is_absolute()
        or PureWindowsPath(path).drive
    ):
        _fail(
            location,
            "expected a repository-relative POSIX path",
            provenance=provenance,
        )

    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        _fail(
            location,
            "path must not contain empty, '.' or '..' segments",
            provenance=provenance,
        )
    if PurePosixPath(path).as_posix() != path:
        _fail(location, "path is not normalized", provenance=provenance)
    return path


def expected_doc_paths(source_path: str) -> tuple[str, str]:
    """Return the two v1 documentation candidates for ``source_path``."""

    source_path = validate_repository_relative_path(source_path, location="source_path")
    appended = f"{source_path}.md"
    try:
        replaced = PurePosixPath(source_path).with_suffix(".md").as_posix()
    except ValueError as exc:
        raise DocFreshnessContractError(
            f"source_path: cannot derive documentation candidate: {exc}"
        ) from exc
    validate_repository_relative_path(appended, location="expected_doc_paths[0]")
    validate_repository_relative_path(replaced, location="expected_doc_paths[1]")
    return appended, replaced


def _path_text(value: Any) -> str | None:
    """Return a text path without coercing arbitrary objects to strings."""

    try:
        raw = os.fspath(value)
    except (TypeError, ValueError):
        return None
    if isinstance(raw, bytes) or not isinstance(raw, str):
        return None
    return raw


def _has_unsafe_path_spelling(raw_path: str) -> bool:
    """Reject spellings that path libraries could silently normalize."""

    if not raw_path or "\\" in raw_path or "\x00" in raw_path:
        return True
    if PureWindowsPath(raw_path).drive:
        return True

    is_absolute = raw_path.startswith("/")
    if is_absolute:
        # A double leading slash has implementation-defined POSIX semantics.
        if raw_path.startswith("//"):
            return True
        segments = raw_path[1:].split("/")
    else:
        segments = raw_path.split("/")
    return any(segment in ("", ".", "..") for segment in segments)


def _file_identity(stat_result: os.stat_result) -> tuple[Any, ...]:
    """Return metadata used to detect replacement or in-place mutation."""

    return (
        getattr(stat_result, "st_dev", None),
        getattr(stat_result, "st_ino", None),
        getattr(stat_result, "st_size", None),
        getattr(stat_result, "st_mtime_ns", None),
    )


def _prepare_safe_path(
    path: Path | str, allowed_root: Path | str
) -> tuple[Path, Path] | str:
    """Resolve a path lexically and reject root escapes and symlink components.

    The return value is either ``(resolved_root, candidate)`` or a stable
    observation reason.  A missing final component is not a preparation error;
    callers need to distinguish a genuinely missing document from an unsafe
    or unreadable one.
    """

    raw_path = _path_text(path)
    if raw_path is None or _has_unsafe_path_spelling(raw_path):
        return "unsafe_path"

    try:
        root = Path(allowed_root).resolve(strict=True)
        root_stat = os.stat(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return "allowed_root_unavailable"
    if not stat_module.S_ISDIR(root_stat.st_mode):
        return "allowed_root_unavailable"

    candidate = Path(raw_path) if Path(raw_path).is_absolute() else root / raw_path
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return "unsafe_path"
    if not relative.parts:
        return "unsafe_path"

    # This catches symlinked parents as well as a symlink at the final path.
    # If a parent is not present yet, lstat stops here and the final check can
    # report the candidate as missing.
    current = root
    for component in relative.parts:
        current /= component
        try:
            component_stat = os.lstat(current)
        except FileNotFoundError:
            break
        except (NotADirectoryError, PermissionError):
            return "unsafe_path"
        except OSError:
            return "hash_unavailable"
        if stat_module.S_ISLNK(component_stat.st_mode):
            return "unsafe_path"

    # Resolve once more immediately before opening/inspecting the file.  This
    # closes the common parent-symlink escape window; the hash path additionally
    # uses O_NOFOLLOW and compares identities before and after reading.
    try:
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return "unsafe_path"

    return root, candidate


def _hash_regular_file_once(candidate: Path, root: Path) -> HashObservation:
    """Attempt one stable hash observation for an already prepared path."""

    try:
        before_path_stat = os.lstat(candidate)
    except FileNotFoundError:
        return HashObservation(None, "file_missing")
    except (NotADirectoryError, PermissionError, OSError):
        return HashObservation(None, "hash_unavailable")

    if stat_module.S_ISLNK(before_path_stat.st_mode):
        return HashObservation(None, "unsafe_path")
    if not stat_module.S_ISREG(before_path_stat.st_mode):
        return HashObservation(None, "non_regular_file")

    try:
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return HashObservation(None, "unsafe_path")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(candidate, flags)
    except FileNotFoundError:
        return HashObservation(None, "file_missing")
    except OSError as exc:
        if exc.errno == getattr(errno, "ELOOP", -1):
            return HashObservation(None, "unsafe_path")
        if exc.errno == getattr(errno, "ENOTDIR", -1):
            return HashObservation(None, "non_regular_file")
        return HashObservation(None, "hash_unavailable")

    try:
        opened_stat = os.fstat(file_descriptor)
        if not stat_module.S_ISREG(opened_stat.st_mode):
            return HashObservation(None, "non_regular_file")
        if _file_identity(before_path_stat) != _file_identity(opened_stat):
            return HashObservation(None, "file_changed_during_scan")

        digest = hashlib.sha256()
        with os.fdopen(file_descriptor, "rb", buffering=0) as handle:
            file_descriptor = None
            while True:
                chunk = handle.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
            after_read_stat = os.fstat(handle.fileno())
            if _file_identity(opened_stat) != _file_identity(after_read_stat):
                return HashObservation(None, "file_changed_during_scan")

        try:
            after_path_stat = os.lstat(candidate)
        except FileNotFoundError:
            return HashObservation(None, "file_changed_during_scan")
        except OSError:
            return HashObservation(None, "hash_unavailable")
        if stat_module.S_ISLNK(after_path_stat.st_mode):
            return HashObservation(None, "unsafe_path")
        if not stat_module.S_ISREG(after_path_stat.st_mode):
            return HashObservation(None, "file_changed_during_scan")
        if _file_identity(after_path_stat) != _file_identity(after_read_stat):
            return HashObservation(None, "file_changed_during_scan")

        # Re-check parent and target components after reading.  In addition to
        # the final lstat identity comparison above, this prevents a parent
        # directory from being swapped for a symlink to a location outside the
        # allowed root while the descriptor was open.
        post_read_path = _prepare_safe_path(candidate, root)
        if isinstance(post_read_path, str):
            if post_read_path == "unsafe_path":
                return HashObservation(None, "unsafe_path")
            return HashObservation(None, "file_changed_during_scan")

        return HashObservation(digest.hexdigest())
    except (OSError, ValueError):
        return HashObservation(None, "hash_unavailable")
    finally:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass


def safe_hash_regular_file(
    path: Path | str, allowed_root: Path | str
) -> HashObservation:
    """Hash exact bytes of a regular, non-symlink file under ``allowed_root``.

    The function never follows a target symlink, never accepts a lexical root
    escape, and returns an unavailable observation for unsafe/unreadable input.
    A metadata race is retried once; a second unstable observation remains
    unavailable instead of returning a potentially non-reproducible digest.
    """

    last_observation = HashObservation(None, "hash_unavailable", 0)
    for attempt in range(1, _HASH_MAX_ATTEMPTS + 1):
        prepared = _prepare_safe_path(path, allowed_root)
        if isinstance(prepared, str):
            return HashObservation(None, prepared, attempt)
        root, candidate = prepared
        observation = _hash_regular_file_once(candidate, root)
        last_observation = HashObservation(
            observation.digest,
            observation.reason,
            attempt,
        )
        if observation.reason != "file_changed_during_scan":
            return last_observation
    return last_observation


def _node_identity(stat_result: os.stat_result) -> tuple[Any, Any]:
    """Return the stable filesystem identity portion of a stat result."""

    return (
        getattr(stat_result, "st_dev", None),
        getattr(stat_result, "st_ino", None),
    )


def _read_regular_file_once(candidate: Path, root: Path) -> _ReadObservation:
    """Read one regular file through contained, no-follow descriptors.

    Provenance is an authority input to the evaluator, so it must use the
    same fail-closed boundary as the other filesystem observations.  Keeping
    the directory descriptors open prevents a parent replacement from
    redirecting the descriptor-relative read.  The visible path and file
    identities are checked again before the bytes are returned.
    """

    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return _ReadObservation(None, "unsafe_path")
    parts = relative.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        return _ReadObservation(None, "unsafe_path")

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
        return _ReadObservation(None, "safe_read_unavailable")

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
    root_fd: int | None = None
    parent_fd: int | None = None
    try:
        try:
            root_fd = os.open(root, directory_flags)
            descriptors.append(root_fd)
            root_identity = _node_identity(os.fstat(root_fd))
            parent_fd = os.dup(root_fd)
            descriptors.append(parent_fd)
        except OSError as exc:
            if exc.errno == getattr(errno, "ENOENT", -1):
                return _ReadObservation(None, "file_missing")
            return _ReadObservation(None, "read_unavailable")

        for component in parts[:-1]:
            try:
                before_stat = os.stat(
                    component,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return _ReadObservation(None, "file_missing")
            except OSError:
                return _ReadObservation(None, "read_unavailable")
            if stat_module.S_ISLNK(before_stat.st_mode):
                return _ReadObservation(None, "unsafe_path")
            if not stat_module.S_ISDIR(before_stat.st_mode):
                return _ReadObservation(None, "unsafe_path")

            next_fd: int | None = None
            try:
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                opened_stat = os.fstat(next_fd)
            except FileNotFoundError:
                if next_fd is not None:
                    os.close(next_fd)
                return _ReadObservation(None, "file_missing")
            except OSError as exc:
                if next_fd is not None:
                    os.close(next_fd)
                if exc.errno == getattr(errno, "ELOOP", -1):
                    return _ReadObservation(None, "unsafe_path")
                return _ReadObservation(None, "read_unavailable")
            if _node_identity(before_stat) != _node_identity(opened_stat):
                os.close(next_fd)
                return _ReadObservation(None, "file_changed_during_scan")

            descriptors.append(next_fd)
            assert parent_fd is not None
            descriptors.remove(parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd

        assert parent_fd is not None
        final_name = parts[-1]
        try:
            before_file_stat = os.stat(
                final_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return _ReadObservation(None, "file_missing")
        except OSError:
            return _ReadObservation(None, "read_unavailable")
        if stat_module.S_ISLNK(before_file_stat.st_mode):
            return _ReadObservation(None, "unsafe_path")
        if not stat_module.S_ISREG(before_file_stat.st_mode):
            return _ReadObservation(None, "non_regular_file")

        try:
            file_fd = os.open(final_name, file_flags, dir_fd=parent_fd)
            descriptors.append(file_fd)
            opened_file_stat = os.fstat(file_fd)
        except FileNotFoundError:
            return _ReadObservation(None, "file_missing")
        except OSError as exc:
            if exc.errno == getattr(errno, "ELOOP", -1):
                return _ReadObservation(None, "unsafe_path")
            return _ReadObservation(None, "read_unavailable")
        if not stat_module.S_ISREG(opened_file_stat.st_mode):
            return _ReadObservation(None, "non_regular_file")
        if _file_identity(before_file_stat) != _file_identity(opened_file_stat):
            return _ReadObservation(None, "file_changed_during_scan")

        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(file_fd, HASH_CHUNK_SIZE)
            except OSError:
                return _ReadObservation(None, "read_unavailable")
            if not chunk:
                break
            chunks.append(chunk)

        after_file_stat = os.fstat(file_fd)
        if _file_identity(opened_file_stat) != _file_identity(after_file_stat):
            return _ReadObservation(None, "file_changed_during_scan")

        try:
            visible_file_stat = os.stat(
                final_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except (FileNotFoundError, OSError):
            return _ReadObservation(None, "file_changed_during_scan")
        if stat_module.S_ISLNK(visible_file_stat.st_mode):
            return _ReadObservation(None, "unsafe_path")
        if (
            not stat_module.S_ISREG(visible_file_stat.st_mode)
            or _file_identity(visible_file_stat) != _file_identity(after_file_stat)
        ):
            return _ReadObservation(None, "file_changed_during_scan")

        post_prepared = _prepare_safe_path(candidate, root)
        if isinstance(post_prepared, str):
            if post_prepared == "unsafe_path":
                return _ReadObservation(None, "unsafe_path")
            return _ReadObservation(None, "file_changed_during_scan")

        post_parent = root.joinpath(*parts[:-1]) if len(parts) > 1 else root
        try:
            visible_parent_stat = os.stat(post_parent, follow_symlinks=False)
        except OSError:
            return _ReadObservation(None, "file_changed_during_scan")
        if (
            not stat_module.S_ISDIR(visible_parent_stat.st_mode)
            or _node_identity(visible_parent_stat) != _node_identity(os.fstat(parent_fd))
        ):
            return _ReadObservation(None, "file_changed_during_scan")

        try:
            visible_root_stat = os.stat(root, follow_symlinks=False)
        except OSError:
            return _ReadObservation(None, "file_changed_during_scan")
        if (
            not stat_module.S_ISDIR(visible_root_stat.st_mode)
            or _node_identity(visible_root_stat) != root_identity
            or _node_identity(os.fstat(root_fd)) != root_identity
        ):
            return _ReadObservation(None, "file_changed_during_scan")

        return _ReadObservation(b"".join(chunks))
    except (OSError, ValueError):
        return _ReadObservation(None, "read_unavailable")
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_regular_file_bytes(
    path: Path | str, allowed_root: Path | str
) -> _ReadObservation:
    """Read a regular file safely, retrying one detected replacement race."""

    last_observation = _ReadObservation(None, "read_unavailable", 0)
    for attempt in range(1, _HASH_MAX_ATTEMPTS + 1):
        prepared = _prepare_safe_path(path, allowed_root)
        if isinstance(prepared, str):
            reason = "unsafe_path" if prepared == "unsafe_path" else "read_unavailable"
            return _ReadObservation(None, reason, attempt)
        root, candidate = prepared
        observation = _read_regular_file_once(candidate, root)
        last_observation = _ReadObservation(
            observation.payload,
            observation.reason,
            attempt,
        )
        if observation.reason != "file_changed_during_scan":
            return last_observation
    return last_observation


def _safe_directory_open_flags() -> int:
    """Return flags required for race-safe directory-relative traversal."""

    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if not directory_flag or not nofollow_flag:
        raise DocProvenanceContractError(
            "write_path: directory no-follow flags are unavailable"
        )
    return (
        os.O_RDONLY
        | directory_flag
        | nofollow_flag
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_safe_parent_directory(
    root: Path,
    relative_parts: tuple[str, ...],
) -> tuple[int, int, tuple[Any, Any]]:
    """Open/create parent components without resolving path strings.

    Every component is opened relative to the previously opened directory fd,
    with ``O_NOFOLLOW``.  Once a directory fd is acquired, replacing its path
    name with a symlink cannot redirect the relative open through that symlink;
    the caller also re-checks the visible parent identity before writing.  The
    root fd remains open so preparation files can be written in the allowed
    root even if a nested destination parent is moved during the operation.
    """

    if not relative_parts:
        raise DocProvenanceContractError("write_path: destination path is empty")

    directory_flags = _safe_directory_open_flags()
    root_fd: int | None = None
    parent_fd: int | None = None
    try:
        root_fd = os.open(root, directory_flags)
        root_identity = _node_identity(os.fstat(root_fd))
        # Keep distinct descriptors: traversal is allowed to replace the
        # parent descriptor while the root descriptor anchors temp files.
        parent_fd = os.dup(root_fd)
        for component in relative_parts[:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o777, dir_fd=parent_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        return parent_fd, root_fd, root_identity
    except Exception:
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass
        raise


def _destination_lstat_at(parent_fd: int, name: str) -> os.stat_result | None:
    """Inspect the final component without following a symlink."""

    try:
        return os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None


def _destination_stat_at(parent_fd: int, name: str) -> os.stat_result | None:
    """Inspect a regular-file destination without following a symlink."""

    destination_stat = _destination_lstat_at(parent_fd, name)
    if destination_stat is None:
        return None
    if stat_module.S_ISLNK(destination_stat.st_mode):
        raise DocProvenanceContractError(
            "write_path: destination must not be a symlink"
        )
    if not stat_module.S_ISREG(destination_stat.st_mode):
        raise DocProvenanceContractError(
            "write_path: destination must be a regular file"
        )
    return destination_stat


def _open_temporary_file_at(directory_fd: int, destination_name: str) -> tuple[int, str]:
    """Create a private preparation file relative to an already-open root fd."""

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    prefix = hashlib.sha256(destination_name.encode("utf-8")).hexdigest()[:16]
    for attempt in range(100):
        name = f".{prefix}.{os.getpid()}.{attempt}.tmp"
        try:
            return os.open(name, flags, 0o666, dir_fd=directory_fd), name
        except FileExistsError:
            continue
        except OSError as exc:
            raise DocProvenanceContractError(
                "write_path: temporary destination cannot be created safely"
            ) from exc
    raise DocProvenanceContractError(
        "write_path: temporary destination name allocation failed"
    )


def _link_destination_backup_at(
    source_directory_fd: int,
    destination_directory_fd: int,
    destination_name: str,
) -> str:
    """Keep a root-local hard-link backup of an existing destination."""

    prefix = hashlib.sha256(destination_name.encode("utf-8")).hexdigest()[:16]
    for attempt in range(100):
        backup_name = f".{prefix}.{os.getpid()}.{attempt}.bak"
        try:
            os.link(
                destination_name,
                backup_name,
                src_dir_fd=destination_directory_fd,
                dst_dir_fd=source_directory_fd,
                follow_symlinks=False,
            )
            return backup_name
        except FileExistsError:
            continue
        except OSError as exc:
            raise DocProvenanceContractError(
                "write_path: existing destination cannot be backed up safely"
            ) from exc
    raise DocProvenanceContractError(
        "write_path: destination backup name allocation failed"
    )


def _remove_temporary_file(directory_fd: int, name: str | None) -> None:
    """Best-effort cleanup of a preparation file relative to its root fd."""

    if name is None:
        return
    try:
        os.unlink(name, dir_fd=directory_fd)
    except (FileNotFoundError, OSError):
        pass


def _verify_safe_write_location(
    *,
    root: Path,
    destination: Path,
    relative: Path,
    root_fd: int,
    parent_fd: int,
    root_identity: tuple[Any, Any],
    destination_name: str,
    previous_stat: os.stat_result | None,
    expected_destination_identity: tuple[Any, Any] | None = None,
) -> None:
    """Verify visible paths still refer to the descriptors used for commit."""

    post_prepared = _prepare_safe_path(destination, root)
    if isinstance(post_prepared, str):
        raise DocProvenanceContractError(
            "write_path: destination is no longer safely contained"
        )
    post_root, _post_destination = post_prepared
    if post_root != root:
        raise DocProvenanceContractError(
            "write_path: allowed root changed during inspection"
        )

    post_parent = post_root.joinpath(*relative.parts[:-1])
    post_parent_stat = os.stat(post_parent, follow_symlinks=False)
    if not stat_module.S_ISDIR(post_parent_stat.st_mode):
        raise DocProvenanceContractError(
            "write_path: destination parent must be a directory"
        )
    if _node_identity(post_parent_stat) != _node_identity(os.fstat(parent_fd)):
        raise DocProvenanceContractError(
            "write_path: destination parent changed during inspection"
        )

    post_destination_stat = _destination_stat_at(parent_fd, destination_name)
    if expected_destination_identity is not None:
        if (
            post_destination_stat is None
            or _node_identity(post_destination_stat) != expected_destination_identity
        ):
            raise DocProvenanceContractError(
                "write_path: committed destination changed during inspection"
            )
    elif previous_stat is None:
        if post_destination_stat is not None:
            raise DocProvenanceContractError(
                "write_path: destination changed during inspection"
            )
    elif (
        post_destination_stat is None
        or _node_identity(post_destination_stat) != _node_identity(previous_stat)
    ):
        raise DocProvenanceContractError(
            "write_path: destination changed during inspection"
        )

    if _node_identity(os.fstat(root_fd)) != root_identity:
        raise DocProvenanceContractError(
            "write_path: allowed root changed during inspection"
        )
    root_stat = os.stat(root, follow_symlinks=False)
    if (
        not stat_module.S_ISDIR(root_stat.st_mode)
        or _node_identity(root_stat) != root_identity
    ):
        raise DocProvenanceContractError(
            "write_path: allowed root changed during inspection"
        )


def write_text_regular_file(
    path: Path | str,
    text: str,
    allowed_root: Path | str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write text without following an unsafe destination path.

    The destination parent is traversed through directory file descriptors,
    while payload bytes are prepared in a temporary file relative to the
    allowed-root fd.  The destination is replaced only after the visible root,
    parent, and destination identities still match the descriptors.  This
    prevents a nested parent moved outside the root during preparation from
    receiving payload bytes through an already-open destination fd.
    """

    if not isinstance(text, str):
        raise TypeError("text must be str")

    prepared = _prepare_safe_path(path, allowed_root)
    if isinstance(prepared, str):
        raise DocProvenanceContractError(
            f"write_path: destination is not safely contained by allowed_root ({prepared})"
        )
    root, destination = prepared
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise DocProvenanceContractError(
            "write_path: destination is no longer safely contained"
        ) from exc

    payload = text.encode(encoding)
    parent_fd: int | None = None
    root_fd: int | None = None
    temporary_fd: int | None = None
    temporary_name: str | None = None
    backup_name: str | None = None
    try:
        parent_fd, root_fd, root_identity = _open_safe_parent_directory(
            root,
            tuple(relative.parts),
        )
        parent_path = root.joinpath(*relative.parts[:-1])
        parent_stat = os.stat(parent_path, follow_symlinks=False)
        if not stat_module.S_ISDIR(parent_stat.st_mode):
            raise DocProvenanceContractError(
                "write_path: destination parent must be a directory"
            )
        if _node_identity(parent_stat) != _node_identity(os.fstat(parent_fd)):
            raise DocProvenanceContractError(
                "write_path: destination parent changed during inspection"
            )

        destination_name = relative.parts[-1]
        previous_stat = _destination_stat_at(parent_fd, destination_name)
        temporary_fd, temporary_name = _open_temporary_file_at(
            root_fd, destination_name
        )
        temporary_identity = _node_identity(os.fstat(temporary_fd))
        if previous_stat is not None:
            os.fchmod(temporary_fd, stat_module.S_IMODE(previous_stat.st_mode))

        # Keep this truncate as part of the preparation phase.  A parent swap
        # injected at this boundary must be detected before payload bytes are
        # written anywhere that could have moved outside the root.
        os.ftruncate(temporary_fd, 0)
        _verify_safe_write_location(
            root=root,
            destination=destination,
            relative=relative,
            root_fd=root_fd,
            parent_fd=parent_fd,
            root_identity=root_identity,
            destination_name=destination_name,
            previous_stat=previous_stat,
        )

        if previous_stat is not None:
            backup_name = _link_destination_backup_at(
                root_fd,
                parent_fd,
                destination_name,
            )

        with os.fdopen(temporary_fd, "wb", closefd=True) as handle:
            temporary_fd = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            prepared_temporary_stat = os.stat(
                temporary_name,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
        except (FileNotFoundError, OSError) as exc:
            raise DocProvenanceContractError(
                "write_path: temporary destination changed during preparation"
            ) from exc
        if (
            not stat_module.S_ISREG(prepared_temporary_stat.st_mode)
            or _node_identity(prepared_temporary_stat) != temporary_identity
        ):
            raise DocProvenanceContractError(
                "write_path: temporary destination changed during preparation"
            )

        _verify_safe_write_location(
            root=root,
            destination=destination,
            relative=relative,
            root_fd=root_fd,
            parent_fd=parent_fd,
            root_identity=root_identity,
            destination_name=destination_name,
            previous_stat=previous_stat,
        )
        committed = False
        try:
            os.replace(
                temporary_name,
                destination_name,
                src_dir_fd=root_fd,
                dst_dir_fd=parent_fd,
            )
            committed = True
            # A directory fd remains usable after its path is renamed.  The
            # post-commit check is therefore required: an adversary can move
            # the visible parent between the preflight check and rename, in
            # which case the kernel quite correctly commits into the directory
            # referred to by the fd.  Treat that combination as a failed
            # publication.  The source pathname may also have been replaced
            # after the preparation check, so the committed inode is not
            # necessarily the payload prepared by this call.
            _verify_safe_write_location(
                root=root,
                destination=destination,
                relative=relative,
                root_fd=root_fd,
                parent_fd=parent_fd,
                root_identity=root_identity,
                destination_name=destination_name,
                previous_stat=None,
                expected_destination_identity=temporary_identity,
            )
        except Exception:
            if committed:
                try:
                    committed_stat = _destination_lstat_at(parent_fd, destination_name)
                    if backup_name is not None:
                        # Replacing from the root-local hard-link backup is
                        # atomic and removes either our payload or an
                        # unexpected inode without an identity-to-unlink race.
                        os.replace(
                            backup_name,
                            destination_name,
                            src_dir_fd=root_fd,
                            dst_dir_fd=parent_fd,
                        )
                        backup_name = None
                    elif committed_stat is not None:
                        # There was no prior destination.  Do not leave an
                        # unexpected regular file (or symlink) published after
                        # a failed commit.  Directories are intentionally not
                        # recursively removed; the original failure remains
                        # the result in that case.
                        os.unlink(destination_name, dir_fd=parent_fd)
                except (OSError, DocProvenanceContractError):
                    # If cleanup itself cannot be proved safe, the caller still
                    # receives the fail-closed error.  In particular, do not
                    # unlink a destination after a failed inspection unless the
                    # operation above has identified a removable node.
                    pass
            raise
        else:
            temporary_name = None
            _remove_temporary_file(root_fd, backup_name)
            backup_name = None
    except DocProvenanceContractError:
        raise
    except (OSError, ValueError) as exc:
        raise DocProvenanceContractError(
            "write_path: destination cannot be written safely"
        ) from exc
    finally:
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
        if root_fd is not None:
            _remove_temporary_file(root_fd, temporary_name)
            _remove_temporary_file(root_fd, backup_name)
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass


def _portable_path_identity(path: str) -> str:
    """Return the cross-platform identity used for collision diagnostics."""

    return unicodedata.normalize("NFC", path).casefold()


def _target_source_path(target: Any, index: int) -> str:
    """Extract and validate a source identity from supported target shapes."""

    if isinstance(target, CoverageTarget):
        value = target.source_path
    elif isinstance(target, (str, os.PathLike)):
        value = target
    elif isinstance(target, Mapping):
        if "source_path" in target:
            value = target["source_path"]
        elif "path" in target:
            value = target["path"]
        else:
            _fail(f"targets[{index}]", "source_path or path is required")
    else:
        value = getattr(target, "source_path", _MISSING)
        if value is _MISSING:
            value = getattr(target, "path", _MISSING)
        if value is _MISSING:
            _fail(f"targets[{index}]", "source_path or path is required")

    return validate_repository_relative_path(
        _path_text(value),
        location=f"targets[{index}].source_path",
    )


def _inspect_doc_candidate(
    candidate_path: str, output_root: Path | str
) -> _CandidateObservation:
    """Classify one candidate without reading its contents."""

    prepared = _prepare_safe_path(candidate_path, output_root)
    if isinstance(prepared, str):
        if prepared == "unsafe_path":
            return _CandidateObservation(candidate_path, "unsafe", "unsafe_doc_path")
        return _CandidateObservation(
            candidate_path,
            "unavailable",
            "doc_hash_unavailable",
        )
    _root, candidate = prepared

    try:
        candidate_stat = os.lstat(candidate)
    except FileNotFoundError:
        return _CandidateObservation(candidate_path, "missing")
    except (NotADirectoryError, PermissionError, OSError):
        return _CandidateObservation(
            candidate_path,
            "unavailable",
            "doc_hash_unavailable",
        )

    if stat_module.S_ISLNK(candidate_stat.st_mode):
        return _CandidateObservation(candidate_path, "unsafe", "unsafe_doc_path")
    if not stat_module.S_ISREG(candidate_stat.st_mode):
        return _CandidateObservation(candidate_path, "unsafe", "unsafe_doc_path")
    return _CandidateObservation(candidate_path, "regular")


def resolve_doc_mappings(
    targets: Sequence[CoverageTarget | Mapping[str, Any] | str | os.PathLike[str]],
    output_root: Path | str,
) -> MappingResolution:
    """Resolve source/document candidates with global collision detection.

    A source ``a/b.py`` has candidates ``a/b.py.md`` and ``a/b.md`` in that
    order.  Exactly one safe regular candidate is selected.  Zero candidates,
    two candidates, unsafe candidates, and global identity collisions are
    represented explicitly and never resolved by enumeration order.

    Exact duplicate source identities violate the artifact contract and raise
    ``DocFreshnessContractError``.  Case-folded or Unicode-normalization source
    collisions are retained as entries with ``doc_identity_collision`` so no
    document can be selected for either identity.
    """

    source_paths: list[str] = []
    seen_sources: set[str] = set()
    for index, target in enumerate(targets):
        source_path = _target_source_path(target, index)
        if source_path in seen_sources:
            raise DocFreshnessContractError(
                f"targets[{index}].source_path: duplicate_source_identity: {source_path!r}"
            )
        seen_sources.add(source_path)
        source_paths.append(source_path)

    source_paths.sort()
    observations_by_source: dict[str, tuple[_CandidateObservation, ...]] = {}
    expected_by_source: dict[str, tuple[str, str]] = {}
    for source_path in source_paths:
        expected = expected_doc_paths(source_path)
        expected_by_source[source_path] = expected
        unique_candidates = tuple(dict.fromkeys(expected))
        observations_by_source[source_path] = tuple(
            _inspect_doc_candidate(candidate_path, output_root)
            for candidate_path in unique_candidates
        )

    portable_source_owners: dict[str, set[str]] = {}
    for source_path in source_paths:
        portable_source_owners.setdefault(
            _portable_path_identity(source_path), set()
        ).add(source_path)
    source_collision_paths = {
        source_path
        for owners in portable_source_owners.values()
        if len(owners) > 1
        for source_path in owners
    }

    portable_doc_owners: dict[str, set[str]] = {}
    for source_path in source_paths:
        # Build the reverse index from every expected identity, not only from
        # currently present files.  A missing candidate can become ambiguous
        # on a case-folding or Unicode-normalizing filesystem, so it must not
        # be selected merely because this checkout is case-sensitive today.
        for candidate_path in dict.fromkeys(expected_by_source[source_path]):
            portable_doc_owners.setdefault(
                _portable_path_identity(candidate_path), set()
            ).add(source_path)
    doc_collision_paths = {
        source_path
        for owners in portable_doc_owners.values()
        if len(owners) > 1
        for source_path in owners
    }
    collision_paths = source_collision_paths | doc_collision_paths

    entries: list[DocMapping] = []
    for source_path in source_paths:
        expected = expected_by_source[source_path]
        observations = observations_by_source[source_path]
        existing_doc_paths = tuple(
            observation.path
            for observation in observations
            if observation.state == "regular"
        )

        reason: str | None = None
        doc_path: str | None = None
        if source_path in collision_paths:
            reason = "doc_identity_collision"
        elif any(observation.state == "unsafe" for observation in observations):
            reason = "unsafe_doc_path"
        elif any(
            observation.state == "unavailable" for observation in observations
        ):
            reason = "doc_hash_unavailable"
        elif not existing_doc_paths:
            reason = "doc_missing"
        elif len(existing_doc_paths) > 1:
            reason = "ambiguous_doc_mapping"
        else:
            doc_path = existing_doc_paths[0]

        diagnostics = (reason,) if reason is not None else ()
        entries.append(
            DocMapping(
                source_path=source_path,
                expected_doc_paths=expected,
                doc_path=doc_path,
                existing_doc_paths=existing_doc_paths,
                reason=reason,
                diagnostics=diagnostics,
            )
        )

    diagnostics = tuple(
        sorted({diagnostic for entry in entries for diagnostic in entry.diagnostics})
    )
    return MappingResolution(entries=tuple(entries), diagnostics=diagnostics)


def validate_sha256(
    value: Any, *, location: str = "hash", provenance: bool = False
) -> str:
    """Validate and return a lowercase, exact-file-bytes SHA-256 digest."""

    digest = _require_string(value, location, provenance=provenance)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        _fail(
            location,
            "expected a lowercase 64-character SHA-256 digest",
            provenance=provenance,
        )
    return digest


def _validate_optional_sha256(value: Any, location: str) -> str | None:
    if value is None:
        return None
    return validate_sha256(value, location=location)


def _validate_diagnostics(
    value: Any, location: str, *, provenance: bool = False
) -> list[str]:
    diagnostics = _require_list(value, location, provenance=provenance)
    validated: list[str] = []
    for index, diagnostic in enumerate(diagnostics):
        code = _require_string(
            diagnostic, f"{location}[{index}]", provenance=provenance
        )
        if code not in DIAGNOSTIC_CODES:
            _fail(
                f"{location}[{index}]",
                f"unsupported diagnostic code: {code!r}",
                provenance=provenance,
            )
        if code in validated:
            _fail(
                f"{location}[{index}]",
                f"duplicate diagnostic code: {code!r}",
                provenance=provenance,
            )
        validated.append(code)
    if validated != sorted(validated):
        _fail(
            location,
            "diagnostics must be ordered lexicographically",
            provenance=provenance,
        )
    return validated


def _validate_freshness_entry(entry: Any, index: int) -> dict[str, Any]:
    location = f"entries[{index}]"
    entry_object = _require_mapping(entry, location)

    source_path = validate_repository_relative_path(
        _required(entry_object, "source_path", location),
        location=f"{location}.source_path",
    )
    expected = _require_list(
        _required(entry_object, "expected_doc_paths", location),
        f"{location}.expected_doc_paths",
    )
    validated_expected = [
        validate_repository_relative_path(
            candidate,
            location=f"{location}.expected_doc_paths[{candidate_index}]",
        )
        for candidate_index, candidate in enumerate(expected)
    ]
    if len(validated_expected) != 2:
        _fail(f"{location}.expected_doc_paths", "expected exactly two documentation candidates")
    expected_by_rule = list(expected_doc_paths(source_path))
    if validated_expected != expected_by_rule:
        _fail(
            f"{location}.expected_doc_paths",
            "does not match the v1 mapping rule and order",
        )

    doc_path_value = _required(entry_object, "doc_path", location)
    if doc_path_value is None:
        doc_path = None
    else:
        doc_path = validate_repository_relative_path(
            doc_path_value,
            location=f"{location}.doc_path",
        )
        if doc_path not in validated_expected:
            _fail(
                f"{location}.doc_path",
                "must be one of expected_doc_paths",
            )

    current_source_hash = _validate_optional_sha256(
        _required(entry_object, "current_source_hash", location),
        f"{location}.current_source_hash",
    )
    recorded_source_hash = _validate_optional_sha256(
        _required(entry_object, "recorded_source_hash", location),
        f"{location}.recorded_source_hash",
    )
    doc_hash = _validate_optional_sha256(
        _required(entry_object, "doc_hash", location),
        f"{location}.doc_hash",
    )

    if doc_path is None and doc_hash is not None:
        _fail(
            f"{location}.doc_hash",
            "must be null when doc_path is null",
        )

    status = _require_string(_required(entry_object, "status", location), f"{location}.status")
    if status not in FRESHNESS_STATUSES:
        _fail(f"{location}.status", f"unsupported freshness status: {status!r}")
    reason = _require_string(_required(entry_object, "reason", location), f"{location}.reason")
    if reason not in FRESHNESS_REASONS:
        _fail(f"{location}.reason", f"unsupported freshness reason: {reason!r}")
    diagnostics = _validate_diagnostics(
        _required(entry_object, "diagnostics", location),
        f"{location}.diagnostics",
    )

    expected_status = {
        "source_hash_match": "fresh",
        "source_hash_mismatch": "stale",
        "doc_missing": "missing",
    }.get(reason, "unknown")
    if status != expected_status:
        _fail(
            f"{location}.status",
            f"status {status!r} is inconsistent with reason {reason!r}",
        )

    if reason == "doc_missing":
        if doc_path is not None or doc_hash is not None:
            _fail(
                location,
                "doc_missing entries must not select a document or document hash",
            )
    elif reason in ("source_hash_match", "source_hash_mismatch"):
        if doc_path is None or current_source_hash is None or recorded_source_hash is None or doc_hash is None:
            _fail(
                location,
                "hash comparison results require source/doc paths and all hashes",
            )
        hashes_match = current_source_hash == recorded_source_hash
        if (reason == "source_hash_match") != hashes_match:
            _fail(
                location,
                "reason does not match the current and recorded source hashes",
            )

    return {
        "source_path": source_path,
        "doc_path": doc_path,
        "expected_doc_paths": validated_expected,
        "current_source_hash": current_source_hash,
        "recorded_source_hash": recorded_source_hash,
        "doc_hash": doc_hash,
        "status": status,
        "reason": reason,
        "diagnostics": diagnostics,
    }


def validate_doc_freshness(
    data: Mapping[str, Any], *, supported_major: int = DOC_FRESHNESS_SCHEMA_MAJOR
) -> None:
    """Validate a v1 ``doc_freshness.json`` document.

    Unknown object fields are tolerated within a supported major so readers can
    survive additive minor extensions.  Required fields, enum values, path and
    hash formats, entry ordering, identity uniqueness, and status counts remain
    strict.
    """

    if type(supported_major) is not int or supported_major < 0:
        raise ValueError("supported_major must be a non-negative integer")

    root = _require_mapping(data, "root")
    major, _minor = _parse_schema_version(
        _required(root, "schema_version", "root"),
        "schema_version",
    )
    if major != supported_major or major != DOC_FRESHNESS_SCHEMA_MAJOR:
        _fail(
            "schema_version",
            f"unsupported schema major {major}; expected {DOC_FRESHNESS_SCHEMA_MAJOR}",
        )

    hash_algorithm = _require_string(
        _required(root, "hash_algorithm", "root"),
        "hash_algorithm",
    )
    if hash_algorithm != DOC_FRESHNESS_HASH_ALGORITHM:
        _fail("hash_algorithm", f"unsupported hash algorithm: {hash_algorithm!r}")
    mapping_rule = _require_string(
        _required(root, "mapping_rule", "root"),
        "mapping_rule",
    )
    if mapping_rule != DOC_FRESHNESS_MAPPING_RULE:
        _fail("mapping_rule", f"unsupported mapping rule: {mapping_rule!r}")

    counts_object = _require_mapping(_required(root, "counts", "root"), "counts")
    if set(counts_object) != set(FRESHNESS_STATUSES):
        _fail(
            "counts",
            f"expected exactly the status keys {list(FRESHNESS_STATUSES)!r}",
        )
    counts = {
        status: _require_non_negative_integer(counts_object[status], f"counts.{status}")
        for status in FRESHNESS_STATUSES
    }

    entries_value = _require_list(_required(root, "entries", "root"), "entries")
    validated_entries: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    source_paths: list[str] = []
    for index, entry in enumerate(entries_value):
        validated = _validate_freshness_entry(entry, index)
        source_path = validated["source_path"]
        if source_path in seen_sources:
            _fail(
                f"entries[{index}].source_path",
                f"duplicate source identity: {source_path!r}",
            )
        seen_sources.add(source_path)
        source_paths.append(source_path)
        validated_entries.append(validated)

    if source_paths != sorted(source_paths):
        _fail("entries", "entries must be ordered by source_path")

    actual_counts = {status: 0 for status in FRESHNESS_STATUSES}
    for entry in validated_entries:
        actual_counts[entry["status"]] += 1
    if counts != actual_counts:
        _fail(
            "counts",
            f"does not match entries (expected {actual_counts!r}, got {counts!r})",
        )

    _validate_diagnostics(
        _required(root, "diagnostics", "root"),
        "diagnostics",
    )


def _validate_provenance_assertion(entry: Any, index: int) -> dict[str, str]:
    location = f"assertions[{index}]"
    assertion = _require_mapping(entry, location, provenance=True)
    source_path = validate_repository_relative_path(
        _required(assertion, "source_path", location, provenance=True),
        location=f"{location}.source_path",
        provenance=True,
    )
    doc_path = validate_repository_relative_path(
        _required(assertion, "doc_path", location, provenance=True),
        location=f"{location}.doc_path",
        provenance=True,
    )
    recorded_source_hash = validate_sha256(
        _required(assertion, "recorded_source_hash", location, provenance=True),
        location=f"{location}.recorded_source_hash",
        provenance=True,
    )
    recorded_doc_hash = validate_sha256(
        _required(assertion, "recorded_doc_hash", location, provenance=True),
        location=f"{location}.recorded_doc_hash",
        provenance=True,
    )
    producer = _require_string(
        _required(assertion, "producer", location, provenance=True),
        f"{location}.producer",
        non_empty=True,
        provenance=True,
    )
    producer_version = _require_string(
        _required(assertion, "producer_version", location, provenance=True),
        f"{location}.producer_version",
        non_empty=True,
        provenance=True,
    )
    return {
        "source_path": source_path,
        "doc_path": doc_path,
        "recorded_source_hash": recorded_source_hash,
        "recorded_doc_hash": recorded_doc_hash,
        "producer": producer,
        "producer_version": producer_version,
    }


def validate_doc_provenance(
    data: Mapping[str, Any], *, supported_major: int = DOC_PROVENANCE_SCHEMA_MAJOR
) -> None:
    """Validate a v1 producer-owned ``doc_provenance.json`` document."""

    if type(supported_major) is not int or supported_major < 0:
        raise ValueError("supported_major must be a non-negative integer")

    root = _require_mapping(data, "root", provenance=True)
    major, _minor = _parse_schema_version(
        _required(root, "schema_version", "root", provenance=True),
        "schema_version",
        provenance=True,
    )
    if major != supported_major or major != DOC_PROVENANCE_SCHEMA_MAJOR:
        _fail(
            "schema_version",
            f"unsupported schema major {major}; expected {DOC_PROVENANCE_SCHEMA_MAJOR}",
            provenance=True,
        )

    hash_algorithm = _require_string(
        _required(root, "hash_algorithm", "root", provenance=True),
        "hash_algorithm",
        provenance=True,
    )
    if hash_algorithm != DOC_FRESHNESS_HASH_ALGORITHM:
        _fail(
            "hash_algorithm",
            f"unsupported hash algorithm: {hash_algorithm!r}",
            provenance=True,
        )

    assertions_value = _require_list(
        _required(root, "assertions", "root", provenance=True),
        "assertions",
        provenance=True,
    )
    validated_assertions: list[dict[str, str]] = []
    seen_identity: set[tuple[str, str]] = set()
    identities: list[tuple[str, str]] = []
    for index, assertion in enumerate(assertions_value):
        validated = _validate_provenance_assertion(assertion, index)
        identity = (validated["source_path"], validated["doc_path"])
        if identity in seen_identity:
            _fail(
                f"assertions[{index}]",
                f"duplicate assertion identity: {identity!r}",
                provenance=True,
            )
        seen_identity.add(identity)
        identities.append(identity)
        validated_assertions.append(validated)

    if identities != sorted(identities):
        _fail(
            "assertions",
            "assertions must be ordered by (source_path, doc_path)",
            provenance=True,
        )

    if "diagnostics" in root:
        _validate_diagnostics(root["diagnostics"], "diagnostics", provenance=True)


def _serialize(
    data: Mapping[str, Any], *, validator: Any, error_type: type[ValueError]
) -> str:
    validator(data)
    try:
        encoded = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise error_type(f"root: cannot serialize JSON: {exc}") from exc
    return encoded + "\n"


def serialize_doc_freshness(data: Mapping[str, Any]) -> str:
    """Return canonical UTF-8 JSON text for a valid freshness document."""

    return _serialize(
        data,
        validator=validate_doc_freshness,
        error_type=DocFreshnessContractError,
    )


def canonical_doc_freshness_bytes(data: Mapping[str, Any]) -> bytes:
    """Return canonical UTF-8 JSON bytes for a valid freshness document."""

    return serialize_doc_freshness(data).encode("utf-8")


def serialize_doc_provenance(data: Mapping[str, Any]) -> str:
    """Return canonical UTF-8 JSON text for a valid provenance document."""

    return _serialize(
        data,
        validator=validate_doc_provenance,
        error_type=DocProvenanceContractError,
    )


def canonical_doc_provenance_bytes(data: Mapping[str, Any]) -> bytes:
    """Return canonical UTF-8 JSON bytes for a valid provenance document."""

    return serialize_doc_provenance(data).encode("utf-8")


def _load_json(path: Path, *, validator: Any, error_type: type[ValueError]) -> dict[str, Any]:
    source_path = Path(path)
    observation = _read_regular_file_bytes(source_path.name, source_path.parent)
    if observation.payload is None:
        raise error_type(
            f"{source_path}: cannot read valid UTF-8 JSON safely: "
            f"{observation.reason or 'read_unavailable'}"
        )
    try:
        data = json.loads(observation.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error_type(
            f"{source_path}: cannot read valid UTF-8 JSON: {exc}"
        ) from exc
    try:
        validator(data)
    except DocFreshnessContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise error_type(f"{source_path}: invalid contract: {exc}") from exc
    if not isinstance(data, dict):
        # The validators reject this, but retain a precise return type for
        # static callers if a future validator changes its implementation.
        raise error_type(f"{source_path}: expected a JSON object")
    return data


def load_doc_freshness(
    path: Path, *, supported_major: int = DOC_FRESHNESS_SCHEMA_MAJOR
) -> dict[str, Any]:
    """Load and validate a freshness document from ``path``."""

    return _load_json(
        Path(path),
        validator=lambda data: validate_doc_freshness(data, supported_major=supported_major),
        error_type=DocFreshnessContractError,
    )


def load_doc_provenance(
    path: Path, *, supported_major: int = DOC_PROVENANCE_SCHEMA_MAJOR
) -> dict[str, Any]:
    """Load and validate a provenance document from ``path``."""

    return _load_json(
        Path(path),
        validator=lambda data: validate_doc_provenance(data, supported_major=supported_major),
        error_type=DocProvenanceContractError,
    )


def _provenance_error_reason(error: BaseException) -> str:
    """Map validator/read failures to the stable provenance reason enum."""

    message = str(error)
    if re.search(r"(?:^|:)assertions\[", message):
        return "provenance_assertion_invalid"
    return "provenance_artifact_invalid"


def _validated_root_diagnostics(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract optional producer diagnostics after the document was validated."""

    value = data.get("diagnostics", ())
    if not isinstance(value, list):
        return ()
    return tuple(sorted(set(value)))


def read_doc_provenance(
    path: Path | str,
    *,
    allowed_root: Path | str | None = None,
    target_paths: Sequence[CoverageTarget | Mapping[str, Any] | str | os.PathLike[str]]
    | None = None,
    targets: Sequence[CoverageTarget | Mapping[str, Any] | str | os.PathLike[str]]
    | None = None,
    mappings: MappingResolution
    | Mapping[str, Any]
    | Sequence[DocMapping | Mapping[str, Any] | None]
    | None = None,
) -> ProvenanceReadResult:
    """Read provenance without allowing an invalid artifact to fail open.

    ``load_doc_provenance`` remains the strict contract loader and raises for
    callers that need an exception.  The machine-scan boundary uses this
    non-throwing reader so a missing, malformed, or unsupported provenance file
    can become a per-entry ``unknown`` result while the rest of the scan is
    still inspectable.
    """

    if target_paths is not None and targets is not None:
        raise ValueError("target_paths and targets are aliases; provide only one")
    known_targets = target_paths if target_paths is not None else targets
    source_path = Path(path)
    read_path: Path | str
    read_root: Path | str
    if allowed_root is None:
        # A path-only call still gets a no-follow final component check.  The
        # machine scan passes its output root explicitly so parent components
        # are also traversed through contained directory descriptors.
        read_path = source_path.name
        read_root = source_path.parent
    else:
        read_path = source_path
        read_root = allowed_root

    observation = _read_regular_file_bytes(read_path, read_root)
    if observation.reason == "file_missing":
        failure = ProvenanceFailure(
            "recorded_source_hash_missing",
            diagnostics=("recorded_source_hash_missing",),
            error=f"{source_path}: file is missing",
        )
        return ProvenanceReadResult(failure=failure)
    if observation.payload is None:
        failure = ProvenanceFailure(
            "provenance_artifact_invalid",
            diagnostics=("provenance_artifact_invalid",),
            error=(
                f"{source_path}: provenance artifact cannot be read safely "
                f"({observation.reason or 'read_unavailable'})"
            ),
        )
        return ProvenanceReadResult(failure=failure)

    try:
        data = json.loads(observation.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        failure = ProvenanceFailure(
            "provenance_artifact_invalid",
            diagnostics=("provenance_artifact_invalid",),
            error=str(exc),
        )
        return ProvenanceReadResult(failure=failure)

    try:
        validate_doc_provenance(data)
    except (DocFreshnessContractError, TypeError, ValueError) as exc:
        reason = _provenance_error_reason(exc)
        failure = ProvenanceFailure(reason, diagnostics=(reason,), error=str(exc))
        return ProvenanceReadResult(failure=failure)

    if not isinstance(data, dict):
        failure = ProvenanceFailure(
            "provenance_artifact_invalid",
            diagnostics=("provenance_artifact_invalid",),
            error="provenance root must be a JSON object",
        )
        return ProvenanceReadResult(failure=failure)
    diagnostics = set(_validated_root_diagnostics(data))
    if known_targets is not None or mappings is not None:
        try:
            if mappings is not None:
                known_mappings = _normalise_mapping_resolution(mappings)
                known_entries = known_mappings.entries
            else:
                known_entries = tuple(
                    DocMapping(
                        source_path=source_path_value,
                        expected_doc_paths=expected_doc_paths(source_path_value),
                        doc_path=None,
                        existing_doc_paths=(),
                    )
                    for index, target in enumerate(known_targets or ())
                    for source_path_value in (
                        _target_source_path(target, index),
                    )
                )
            if _orphan_provenance_diagnostic(data, known_entries):
                diagnostics.add("orphan_provenance_assertion")
        except (DocFreshnessContractError, TypeError, ValueError):
            # The provenance file is valid on its own; an invalid optional
            # caller context must not turn that valid artifact into a false
            # provenance failure.
            pass
    return ProvenanceReadResult(document=data, diagnostics=tuple(sorted(diagnostics)))


@dataclass(frozen=True, slots=True)
class _ProvenanceState:
    document: dict[str, Any] | None
    failure: ProvenanceFailure | None
    diagnostics: tuple[str, ...]


def _coerce_provenance_state(
    provenance: Mapping[str, Any]
    | ProvenanceReadResult
    | ProvenanceFailure
    | Path
    | str
    | None,
) -> _ProvenanceState:
    """Normalize all supported provenance inputs for the pure evaluator."""

    if isinstance(provenance, (Path, str)):
        result = read_doc_provenance(provenance)
        return _ProvenanceState(result.document, result.failure, result.diagnostics)

    if isinstance(provenance, ProvenanceReadResult):
        if provenance.document is not None and provenance.failure is None:
            try:
                validate_doc_provenance(provenance.document)
            except (DocFreshnessContractError, TypeError, ValueError) as exc:
                reason = _provenance_error_reason(exc)
                failure = ProvenanceFailure(reason, diagnostics=(reason,), error=str(exc))
                return _ProvenanceState(None, failure, (reason,))
            return _ProvenanceState(
                provenance.document,
                None,
                tuple(sorted(set(provenance.diagnostics))),
            )
        failure = provenance.failure or ProvenanceFailure("recorded_source_hash_missing")
        diagnostics = tuple(sorted(set((*provenance.diagnostics, *failure.diagnostics))))
        return _ProvenanceState(None, failure, diagnostics)

    if isinstance(provenance, ProvenanceFailure):
        if provenance.reason not in PROVENANCE_FAILURE_REASONS:
            failure = ProvenanceFailure(
                "provenance_artifact_invalid",
                diagnostics=("provenance_artifact_invalid",),
                error=provenance.error,
            )
            return _ProvenanceState(None, failure, failure.diagnostics)
        diagnostics = tuple(sorted(set((*provenance.diagnostics, provenance.reason))))
        failure = ProvenanceFailure(provenance.reason, diagnostics, provenance.error)
        return _ProvenanceState(None, failure, diagnostics)

    if provenance is None:
        failure = ProvenanceFailure(
            "recorded_source_hash_missing",
            diagnostics=("recorded_source_hash_missing",),
        )
        return _ProvenanceState(None, failure, failure.diagnostics)

    if isinstance(provenance, Mapping):
        try:
            validate_doc_provenance(provenance)
        except (DocFreshnessContractError, TypeError, ValueError) as exc:
            reason = _provenance_error_reason(exc)
            failure = ProvenanceFailure(reason, diagnostics=(reason,), error=str(exc))
            return _ProvenanceState(None, failure, failure.diagnostics)
        if not isinstance(provenance, dict):
            # A Mapping is valid input to the validator but the evaluator keeps
            # the result JSON-compatible and does not retain a live custom map.
            provenance = dict(provenance)
        return _ProvenanceState(
            provenance,
            None,
            _validated_root_diagnostics(provenance),
        )

    failure = ProvenanceFailure(
        "provenance_artifact_invalid",
        diagnostics=("provenance_artifact_invalid",),
        error=f"unsupported provenance input: {type(provenance).__name__}",
    )
    return _ProvenanceState(None, failure, failure.diagnostics)


def _mapping_value(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    """Return the first explicitly present value from a mapping."""

    if mapping is None:
        return _MISSING
    for key in keys:
        if key in mapping:
            return mapping[key]
    return _MISSING


def _object_value(value: Any, *keys: str) -> Any:
    """Read a field from a mapping or object without coercing arbitrary values."""

    if isinstance(value, Mapping):
        result = _mapping_value(value, *keys)
        if result is not _MISSING:
            return result
    for key in keys:
        result = getattr(value, key, _MISSING)
        if result is not _MISSING:
            return result
    return _MISSING


def _observation_reason(value: Any) -> str | None:
    if isinstance(value, HashObservation):
        return value.reason
    if isinstance(value, Mapping):
        reason = value.get("reason", _MISSING)
        return reason if isinstance(reason, str) else None
    reason = getattr(value, "reason", None)
    return reason if isinstance(reason, str) else None


def _observation_digest(value: Any) -> Any:
    if isinstance(value, HashObservation):
        return value.digest
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        result = _mapping_value(value, "digest", "sha256", "hash")
        return None if result is _MISSING else result
    result = _object_value(value, "digest", "sha256", "hash")
    return None if result is _MISSING else result


def _hash_reason_for_role(value: Any, role: Literal["source", "doc"]) -> str:
    """Project a role-neutral hash observation failure into the contract enum."""

    observed_reason = _observation_reason(value)
    if observed_reason in ("file_changed_during_scan", "source_changed_during_scan"):
        return "source_changed_during_scan" if role == "source" else "doc_changed_during_scan"
    if observed_reason in ("unsafe_path", "unsafe_source_path"):
        return "unsafe_source_path" if role == "source" else "unsafe_doc_path"
    if role == "doc" and observed_reason in ("non_regular_file", "unsafe_doc_path"):
        return "unsafe_doc_path"
    return "source_hash_unavailable" if role == "source" else "doc_hash_unavailable"


def _normalise_hash_observation(
    value: Any, role: Literal["source", "doc"]
) -> tuple[str | None, str | None]:
    """Return a validated digest and stable failure reason for one observation."""

    if value is _MISSING or value is None:
        return None, _hash_reason_for_role(value, role)
    digest = _observation_digest(value)
    if digest is None:
        return None, _hash_reason_for_role(value, role)
    try:
        return validate_sha256(digest), None
    except (DocFreshnessContractError, TypeError, ValueError):
        return None, _hash_reason_for_role(value, role)


def _target_source_hash(target: Any, source_path: str) -> Any:
    """Extract an evaluator input from machine metadata/target aliases."""

    if isinstance(target, CoverageTarget):
        return target.current_source_hash
    result = _object_value(
        target,
        "current_source_hash",
        "source_hash",
        "hash",
    )
    return None if result is _MISSING else result


def _target_doc_hash(target: Any) -> Any:
    result = _object_value(target, "doc_hash", "current_doc_hash")
    return None if result is _MISSING else result


def _normalise_mapping_entry(value: Any, source_key: str | None = None) -> DocMapping:
    """Accept the public dataclass and small mapping-shaped test fixtures."""

    if isinstance(value, DocMapping):
        return value
    if isinstance(value, Mapping):
        source_value = value.get("source_path", source_key)
        if source_value is _MISSING or source_value is None:
            _fail("mappings", "source_path is required")
        source_path = validate_repository_relative_path(
            source_value, location="mappings.source_path"
        )
        expected_value = value.get("expected_doc_paths", expected_doc_paths(source_path))
        expected = tuple(expected_value)
        doc_path = value.get("doc_path", value.get("selected_doc_path"))
        existing_value = value.get("existing_doc_paths", ())
        diagnostics_value = value.get("diagnostics", ())
        if isinstance(diagnostics_value, str):
            diagnostics_value = (diagnostics_value,)
        return DocMapping(
            source_path=source_path,
            expected_doc_paths=expected,  # validated below
            doc_path=doc_path,
            existing_doc_paths=tuple(existing_value),
            reason=value.get("reason"),
            diagnostics=tuple(diagnostics_value),
            doc_hash=value.get("doc_hash", value.get("current_doc_hash")),
            current_source_hash=value.get("current_source_hash", value.get("source_hash")),
        )
    if value is None and source_key is not None:
        source_path = validate_repository_relative_path(source_key, location="mappings.source_path")
        return DocMapping(
            source_path=source_path,
            expected_doc_paths=expected_doc_paths(source_path),
            doc_path=None,
            existing_doc_paths=(),
            reason="doc_missing",
            diagnostics=("doc_missing",),
        )
    if isinstance(value, str) and source_key is not None:
        source_path = validate_repository_relative_path(source_key, location="mappings.source_path")
        return DocMapping(
            source_path=source_path,
            expected_doc_paths=expected_doc_paths(source_path),
            doc_path=value,
            existing_doc_paths=(value,),
        )
    raise DocFreshnessContractError(
        f"mappings: unsupported mapping entry {type(value).__name__}"
    )


def _validate_mapping_entry(entry: DocMapping, index: int) -> DocMapping:
    """Validate a mapping before allowing it to influence a freshness result."""

    source_path = validate_repository_relative_path(
        entry.source_path, location=f"mappings[{index}].source_path"
    )
    expected = tuple(entry.expected_doc_paths)
    expected_by_rule = expected_doc_paths(source_path)
    if expected != expected_by_rule:
        raise DocFreshnessContractError(
            f"mappings[{index}].expected_doc_paths: does not match the v1 mapping rule"
        )

    doc_path = entry.doc_path
    if doc_path is not None:
        doc_path = validate_repository_relative_path(
            doc_path, location=f"mappings[{index}].doc_path"
        )
        if doc_path not in expected_by_rule:
            raise DocFreshnessContractError(
                f"mappings[{index}].doc_path: must be one of expected_doc_paths"
            )

    existing_doc_paths = tuple(entry.existing_doc_paths)
    for path_index, candidate in enumerate(existing_doc_paths):
        validate_repository_relative_path(
            candidate,
            location=f"mappings[{index}].existing_doc_paths[{path_index}]",
        )
    if len(set(existing_doc_paths)) != len(existing_doc_paths):
        raise DocFreshnessContractError(
            f"mappings[{index}].existing_doc_paths: duplicate documentation path"
        )

    reason = entry.reason
    if reason is not None and reason not in FRESHNESS_REASONS:
        raise DocFreshnessContractError(
            f"mappings[{index}].reason: unsupported mapping reason {reason!r}"
        )
    if reason == "doc_missing" and (doc_path is not None or existing_doc_paths):
        raise DocFreshnessContractError(
            f"mappings[{index}]: doc_missing mapping must not select a document"
        )
    if reason in {
        "ambiguous_doc_mapping",
        "doc_identity_collision",
        "doc_hash_unavailable",
        "unsafe_doc_path",
    } and doc_path is not None:
        raise DocFreshnessContractError(
            f"mappings[{index}]: {reason} mapping must not select a document"
        )

    diagnostics = tuple(entry.diagnostics)
    _validate_diagnostics(list(diagnostics), f"mappings[{index}].diagnostics")
    return DocMapping(
        source_path=source_path,
        expected_doc_paths=expected_by_rule,
        doc_path=doc_path,
        existing_doc_paths=existing_doc_paths,
        reason=reason,
        diagnostics=diagnostics,
        doc_hash=entry.doc_hash,
        current_source_hash=entry.current_source_hash,
    )


def _normalise_mapping_resolution(
    mappings: MappingResolution
    | Mapping[str, Any]
    | Sequence[DocMapping | Mapping[str, Any] | None],
) -> MappingResolution:
    if isinstance(mappings, MappingResolution):
        entries = [
            _validate_mapping_entry(entry, index)
            for index, entry in enumerate(mappings.entries)
        ]
        seen_sources: set[str] = set()
        for index, entry in enumerate(entries):
            if entry.source_path in seen_sources:
                raise DocFreshnessContractError(
                    f"mappings[{index}].source_path: duplicate source identity: {entry.source_path!r}"
                )
            seen_sources.add(entry.source_path)
        return MappingResolution(
            entries=tuple(sorted(entries, key=lambda entry: entry.source_path)),
            diagnostics=tuple(sorted(set(mappings.diagnostics))),
            source_hashes=mappings.source_hashes,
            doc_hashes=mappings.doc_hashes,
        )

    raw_entries: list[tuple[str | None, Any]] = []
    if isinstance(mappings, Mapping):
        # A single mapping-shaped object is accepted in addition to the
        # source-path -> mapping form used by compact callers.
        if "source_path" in mappings or "doc_path" in mappings:
            raw_entries.append((None, mappings))
        else:
            raw_entries.extend(mappings.items())
    elif isinstance(mappings, Sequence) and not isinstance(mappings, (str, bytes, bytearray)):
        raw_entries.extend((None, value) for value in mappings)
    else:
        raise DocFreshnessContractError("mappings: expected MappingResolution, object, or array")

    entries = [
        _normalise_mapping_entry(value, source_key) for source_key, value in raw_entries
    ]
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if entry.source_path in seen:
            raise DocFreshnessContractError(
                f"mappings[{index}].source_path: duplicate source identity: {entry.source_path!r}"
            )
        seen.add(entry.source_path)
        entries[index] = _validate_mapping_entry(entry, index)
    return MappingResolution(
        entries=tuple(sorted(entries, key=lambda entry: entry.source_path)),
        diagnostics=tuple(
            sorted(
                set(
                    diagnostic
                    for entry in entries
                    for diagnostic in entry.diagnostics
                )
            )
        ),
    )


def _lookup_hash_override(
    values: Mapping[str, Any] | None, *keys: str
) -> Any:
    if values is None:
        return _MISSING
    return _mapping_value(values, *keys)


def _provenance_assertion_index(
    document: Mapping[str, Any] | None,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    if document is None:
        return by_identity, by_source
    assertions = document.get("assertions", [])
    if not isinstance(assertions, list):
        return by_identity, by_source
    for assertion in assertions:
        if not isinstance(assertion, Mapping):
            continue
        source_path = assertion.get("source_path")
        doc_path = assertion.get("doc_path")
        if not isinstance(source_path, str) or not isinstance(doc_path, str):
            continue
        by_identity[(source_path, doc_path)] = assertion
        by_source.setdefault(source_path, []).append(assertion)
    return by_identity, by_source


def _orphan_provenance_diagnostic(
    document: Mapping[str, Any] | None,
    mapping_entries: Sequence[DocMapping],
) -> bool:
    if document is None:
        return False
    expected_by_source = {
        entry.source_path: set(entry.expected_doc_paths) for entry in mapping_entries
    }
    assertions = document.get("assertions", [])
    if not isinstance(assertions, list):
        return False
    for assertion in assertions:
        if not isinstance(assertion, Mapping):
            return True
        source_path = assertion.get("source_path")
        doc_path = assertion.get("doc_path")
        if source_path not in expected_by_source or doc_path not in expected_by_source[source_path]:
            return True
    return False


def _entry_diagnostics(*values: str | None) -> list[str]:
    return sorted({value for value in values if value in DIAGNOSTIC_CODES})


def _make_freshness_entry(
    mapping: DocMapping,
    *,
    current_source_hash: str | None,
    doc_path: str | None,
    doc_hash: str | None,
    recorded_source_hash: str | None,
    status: FreshnessStatus,
    reason: FreshnessReason,
    diagnostics: Sequence[str] = (),
) -> DocFreshnessEntry:
    return {
        "source_path": mapping.source_path,
        "doc_path": doc_path,
        "expected_doc_paths": list(mapping.expected_doc_paths),
        "current_source_hash": current_source_hash,
        "recorded_source_hash": recorded_source_hash,
        "doc_hash": doc_hash,
        "status": status,
        "reason": reason,
        "diagnostics": sorted(
            {
                diagnostic
                for diagnostic in diagnostics
                if diagnostic in DIAGNOSTIC_CODES
                and diagnostic != reason
            }
        ),
    }


def evaluate_doc_freshness(
    targets: Sequence[CoverageTarget | Mapping[str, Any] | str | os.PathLike[str]],
    mappings: MappingResolution
    | Mapping[str, Any]
    | Sequence[DocMapping | Mapping[str, Any] | None],
    provenance: Mapping[str, Any]
    | ProvenanceReadResult
    | ProvenanceFailure
    | Path
    | str
    | None = None,
    source_hashes: Mapping[str, Any] | None = None,
    doc_hashes: Mapping[str, Any] | None = None,
    *,
    current_source_hashes: Mapping[str, Any] | None = None,
    current_doc_hashes: Mapping[str, Any] | None = None,
    source_observations: Mapping[str, Any] | None = None,
    doc_observations: Mapping[str, Any] | None = None,
) -> DocFreshnessDocument:
    """Evaluate freshness from byte observations and validated provenance.

    The evaluator is deliberately pure with respect to freshness: it never
    reads filesystem mtimes, previous scan output, or Git history.  Current
    source/doc observations can be supplied explicitly through the optional
    mappings, or embedded in target/mapping fixtures for small callers.  A
    valid producer assertion is the only input that can provide
    ``recorded_source_hash``.

    The precedence is source observation, mapping state, document observation,
    provenance validity/binding, and finally source-hash equality.  Every
    output entry keeps the nullable evidence fields even when an earlier guard
    prevents a comparison.
    """

    target_values = list(targets)
    source_paths: list[str] = []
    seen_sources: set[str] = set()
    for index, target in enumerate(target_values):
        source_path = _target_source_path(target, index)
        if source_path in seen_sources:
            raise DocFreshnessContractError(
                f"targets[{index}].source_path: duplicate_source_identity: {source_path!r}"
            )
        seen_sources.add(source_path)
        source_paths.append(source_path)
    source_paths.sort()

    target_by_source = {
        _target_source_path(target, index): target
        for index, target in enumerate(target_values)
    }
    resolution = _normalise_mapping_resolution(mappings)
    mappings_by_source = {entry.source_path: entry for entry in resolution.entries}
    if set(mappings_by_source) != set(source_paths):
        missing = sorted(set(source_paths) - set(mappings_by_source))
        extra = sorted(set(mappings_by_source) - set(source_paths))
        raise DocFreshnessContractError(
            f"mappings: source identities do not match targets (missing={missing!r}, extra={extra!r})"
        )

    provenance_state = _coerce_provenance_state(provenance)
    provenance_document = provenance_state.document
    assertions_by_identity, assertions_by_source = _provenance_assertion_index(
        provenance_document
    )

    effective_source_hashes: dict[str, Any] = {}
    if isinstance(resolution.source_hashes, Mapping):
        effective_source_hashes.update(resolution.source_hashes)
    if source_hashes is not None:
        effective_source_hashes.update(source_hashes)
    if source_observations is not None:
        effective_source_hashes.update(source_observations)
    if current_source_hashes is not None:
        effective_source_hashes.update(current_source_hashes)

    effective_doc_hashes: dict[str, Any] = {}
    if isinstance(resolution.doc_hashes, Mapping):
        effective_doc_hashes.update(resolution.doc_hashes)
    if doc_hashes is not None:
        effective_doc_hashes.update(doc_hashes)
    if doc_observations is not None:
        effective_doc_hashes.update(doc_observations)
    if current_doc_hashes is not None:
        effective_doc_hashes.update(current_doc_hashes)

    entries: list[DocFreshnessEntry] = []
    for source_path in source_paths:
        target = target_by_source[source_path]
        mapping = mappings_by_source[source_path]

        source_value = _lookup_hash_override(effective_source_hashes, source_path)
        if source_value is _MISSING:
            source_value = mapping.current_source_hash
        if source_value is None:
            source_value = _target_source_hash(target, source_path)
        current_source_hash, source_reason = _normalise_hash_observation(
            source_value, "source"
        )

        doc_path = mapping.doc_path
        doc_hash: str | None = None
        recorded_source_hash: str | None = None
        diagnostics = _entry_diagnostics(*mapping.diagnostics, source_reason, mapping.reason)

        if source_reason is not None:
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=doc_path,
                    doc_hash=None,
                    recorded_source_hash=None,
                    status="unknown",
                    reason=source_reason,  # type: ignore[arg-type]
                    diagnostics=diagnostics,
                )
            )
            continue

        mapping_reason = mapping.reason
        if mapping_reason is not None and mapping_reason != "doc_missing":
            safe_reason = mapping_reason if mapping_reason in FRESHNESS_REASONS else "doc_identity_collision"
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=doc_path,
                    doc_hash=None,
                    recorded_source_hash=None,
                    status="unknown",
                    reason=safe_reason,  # type: ignore[arg-type]
                    diagnostics=diagnostics,
                )
            )
            continue

        if doc_path is None:
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=None,
                    doc_hash=None,
                    recorded_source_hash=None,
                    status="missing",
                    reason="doc_missing",
                    diagnostics=diagnostics or ("doc_missing",),
                )
            )
            continue

        doc_value = _lookup_hash_override(effective_doc_hashes, doc_path, source_path)
        if doc_value is _MISSING:
            doc_value = mapping.doc_hash
        if doc_value is _MISSING or doc_value is None:
            target_doc_value = _target_doc_hash(target)
            doc_value = target_doc_value
        doc_hash, doc_reason = _normalise_hash_observation(doc_value, "doc")
        if doc_reason is not None:
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=doc_path,
                    doc_hash=None,
                    recorded_source_hash=None,
                    status="unknown",
                    reason=doc_reason,  # type: ignore[arg-type]
                    diagnostics=_entry_diagnostics(*diagnostics, doc_reason),
                )
            )
            continue

        if provenance_state.failure is not None:
            provenance_reason = provenance_state.failure.reason
            if provenance_reason not in PROVENANCE_FAILURE_REASONS:
                provenance_reason = "provenance_artifact_invalid"
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=doc_path,
                    doc_hash=doc_hash,
                    recorded_source_hash=None,
                    status="unknown",
                    reason=provenance_reason,  # type: ignore[arg-type]
                    diagnostics=_entry_diagnostics(
                        *diagnostics,
                        *provenance_state.diagnostics,
                        provenance_reason,
                    ),
                )
            )
            continue

        assertion = assertions_by_identity.get((source_path, doc_path))
        if assertion is None:
            if assertions_by_source.get(source_path):
                reason: FreshnessReason = "provenance_binding_mismatch"
            else:
                reason = "recorded_source_hash_missing"
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=doc_path,
                    doc_hash=doc_hash,
                    recorded_source_hash=None,
                    status="unknown",
                    reason=reason,
                    diagnostics=_entry_diagnostics(*diagnostics, reason),
                )
            )
            continue

        assertion_doc_hash = assertion.get("recorded_doc_hash")
        if assertion_doc_hash != doc_hash:
            reason = "provenance_doc_hash_mismatch"
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=doc_path,
                    doc_hash=doc_hash,
                    recorded_source_hash=None,
                    status="unknown",
                    reason=reason,
                    diagnostics=_entry_diagnostics(*diagnostics, reason),
                )
            )
            continue

        candidate_recorded_hash = assertion.get("recorded_source_hash")
        try:
            recorded_source_hash = validate_sha256(candidate_recorded_hash)
        except (DocFreshnessContractError, TypeError, ValueError):
            reason = "provenance_assertion_invalid"
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=doc_path,
                    doc_hash=doc_hash,
                    recorded_source_hash=None,
                    status="unknown",
                    reason=reason,
                    diagnostics=_entry_diagnostics(*diagnostics, reason),
                )
            )
            continue

        if current_source_hash == recorded_source_hash:
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=doc_path,
                    doc_hash=doc_hash,
                    recorded_source_hash=recorded_source_hash,
                    status="fresh",
                    reason="source_hash_match",
                    diagnostics=diagnostics,
                )
            )
        else:
            entries.append(
                _make_freshness_entry(
                    mapping,
                    current_source_hash=current_source_hash,
                    doc_path=doc_path,
                    doc_hash=doc_hash,
                    recorded_source_hash=recorded_source_hash,
                    status="stale",
                    reason="source_hash_mismatch",
                    diagnostics=diagnostics,
                )
            )

    entries.sort(key=lambda entry: entry["source_path"])
    counts = {status: 0 for status in FRESHNESS_STATUSES}
    for entry in entries:
        counts[entry["status"]] += 1

    diagnostics = set(resolution.diagnostics)
    diagnostics.update(provenance_state.diagnostics)
    if _orphan_provenance_diagnostic(provenance_document, resolution.entries):
        diagnostics.add("orphan_provenance_assertion")

    document: DocFreshnessDocument = {
        "schema_version": DOC_FRESHNESS_SCHEMA_VERSION,
        "hash_algorithm": DOC_FRESHNESS_HASH_ALGORITHM,
        "mapping_rule": DOC_FRESHNESS_MAPPING_RULE,
        "counts": counts,
        "entries": entries,
        "diagnostics": sorted(
            diagnostic for diagnostic in diagnostics if diagnostic in DIAGNOSTIC_CODES
        ),
    }
    validate_doc_freshness(document)
    return document


FRESHNESS_TO_DOC_STATUS: dict[FreshnessStatus, str] = {
    "fresh": "current",
    "stale": "stale",
    "missing": "missing",
    "unknown": "unavailable",
}


def project_doc_statuses(
    document: Mapping[str, Any],
) -> dict[str, str]:
    """Project freshness statuses into the legacy attention vocabulary."""

    validate_doc_freshness(document)
    return {
        entry["source_path"]: FRESHNESS_TO_DOC_STATUS[entry["status"]]
        for entry in document["entries"]
    }


def project_freshness(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the deterministic legacy coverage/attention projection."""

    validate_doc_freshness(document)
    entries = document["entries"]
    missing_docs = [
        entry["source_path"] for entry in entries if entry["status"] == "missing"
    ]
    stale_docs = [
        entry["source_path"] for entry in entries if entry["status"] == "stale"
    ]
    valid_docs = [
        entry["source_path"] for entry in entries if entry["status"] == "fresh"
    ]
    unknown_docs = [
        entry["source_path"] for entry in entries if entry["status"] == "unknown"
    ]
    total_targets = len(entries)
    documented_count = total_targets - len(missing_docs)
    coverage_summary = {
        "coverage_target_files": total_targets,
        "documented_files": documented_count,
        "missing_docs": len(missing_docs),
        "stale_docs": len(stale_docs),
        "fresh_docs": len(valid_docs),
        "valid_docs": len(valid_docs),
        "unknown_docs": len(unknown_docs),
        "coverage_percent": (
            documented_count / total_targets * 100 if total_targets > 0 else 100.0
        ),
    }
    attention_diagnostics: list[dict[str, str | None]] = []
    for entry in entries:
        if entry["status"] == "unknown":
            attention_diagnostics.append(
                {
                    "detector": "doc_status",
                    "code": entry["reason"],
                    "path": entry["source_path"],
                }
            )
    for diagnostic in document["diagnostics"]:
        if diagnostic == "orphan_provenance_assertion":
            attention_diagnostics.append(
                {"detector": "doc_status", "code": diagnostic, "path": None}
            )
    attention_diagnostics.sort(
        key=lambda item: (str(item["code"]), str(item["path"] or ""))
    )
    statuses = project_doc_statuses(document)
    return {
        "doc_status_by_path": statuses,
        "doc_statuses": statuses,
        "missing_docs": missing_docs,
        "stale_docs": stale_docs,
        "valid_docs": valid_docs,
        "unknown_docs": unknown_docs,
        "coverage_summary": coverage_summary,
        "attention_diagnostics": attention_diagnostics,
    }


# Names used by later machine-analysis integration and by callers that prefer
# an explicit direction in the helper name.
freshness_to_doc_statuses = project_doc_statuses
project_doc_freshness = project_freshness


def _safe_write_root_for_destination(destination: Path) -> Path:
    """Find an existing ancestor from which a destination can be created.

    The safe writer deliberately requires an existing allowed root.  Contract
    writers historically created missing parent directories, so choose the
    nearest existing *directory* without resolving symlinks.  If an existing
    ancestor is a symlink, walk above it; the safe writer then observes and
    rejects that component instead of silently making the resolved target its
    root.
    """

    if not destination.is_absolute():
        return Path.cwd()

    candidate = destination.parent
    while True:
        try:
            candidate_stat = os.lstat(candidate)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise DocProvenanceContractError(
                "write_path: destination parent cannot be inspected"
            ) from exc
        else:
            if stat_module.S_ISDIR(candidate_stat.st_mode) and not stat_module.S_ISLNK(
                candidate_stat.st_mode
            ):
                return candidate

        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent


def _write_atomic(path: Path, payload: str) -> None:
    """Atomically publish a contract payload through the safe writer."""

    destination = Path(path)
    write_text_regular_file(
        destination,
        payload,
        _safe_write_root_for_destination(destination),
    )


def write_doc_freshness_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Validate and atomically write a canonical freshness artifact."""

    _write_atomic(Path(path), serialize_doc_freshness(data))


def write_doc_provenance_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Validate and atomically write a canonical provenance artifact."""

    destination = Path(path)
    write_text_regular_file(
        destination,
        serialize_doc_provenance(data),
        _safe_write_root_for_destination(destination),
    )


def _recording_digest(
    observation: Any, role: Literal["source", "doc"]
) -> str:
    """Return a validated digest or expose a stable producer failure."""

    digest = _observation_digest(observation)
    if digest is not None:
        try:
            return validate_sha256(digest)
        except (DocFreshnessContractError, TypeError, ValueError):
            pass

    reason = _hash_reason_for_role(observation, role)
    raise DocProvenanceRecordingError(
        f"{role} file could not be safely hashed ({reason})",
        reason=reason,
        observation=observation if isinstance(observation, HashObservation) else None,
    )


def _provenance_destination(path: Path | str, output_root: Path | str) -> Path:
    """Validate a producer artifact destination under the output root."""

    prepared = _prepare_safe_path(path, output_root)
    if isinstance(prepared, str):
        raise DocProvenanceContractError(
            f"provenance_path: destination is not safely contained by output_root ({prepared})"
        )
    _root, destination = prepared
    return destination


def record_doc_provenance(
    provenance_path: Path | str,
    *,
    source_root: Path | str,
    output_root: Path | str,
    source_path: str,
    doc_path: str,
    producer: str,
    producer_version: str,
) -> None:
    """Record one producer assertion after two stable file observations.

    ``source_path`` is relative to ``source_root`` and ``doc_path`` is relative
    to ``output_root``.  Both paths are validated before hashing.  The
    provenance file is loaded only after both hashes succeed, then the matching
    identity is replaced and the complete document is atomically published.
    Consequently, a missing/unreadable/unstable source or document cannot
    advance an existing assertion.

    The producer may pass an actual collision-adjusted ``doc_path``.  The
    provenance contract keeps that byte-bound assertion for auditability; the
    freshness mapping resolver will safely treat a path outside its expected
    candidates as unbound rather than treating it as fresh.
    """

    validated_source_path = validate_repository_relative_path(
        source_path,
        location="source_path",
        provenance=True,
    )
    validated_doc_path = validate_repository_relative_path(
        doc_path,
        location="doc_path",
        provenance=True,
    )
    validated_producer = _require_string(
        producer,
        "producer",
        non_empty=True,
        provenance=True,
    )
    validated_producer_version = _require_string(
        producer_version,
        "producer_version",
        non_empty=True,
        provenance=True,
    )
    destination = _provenance_destination(provenance_path, output_root)

    source_observation = safe_hash_regular_file(validated_source_path, source_root)
    current_source_hash = _recording_digest(source_observation, "source")
    doc_observation = safe_hash_regular_file(validated_doc_path, output_root)
    current_doc_hash = _recording_digest(doc_observation, "doc")

    if destination.exists():
        if not destination.is_file():
            raise DocProvenanceContractError(
                f"provenance_path: expected a regular file: {destination}"
            )
        existing = load_doc_provenance(destination)
    else:
        existing = {
            "schema_version": DOC_PROVENANCE_SCHEMA_VERSION,
            "hash_algorithm": DOC_FRESHNESS_HASH_ALGORITHM,
            "assertions": [],
        }

    assertions = [
        dict(assertion)
        for assertion in existing["assertions"]
        if (
            assertion["source_path"],
            assertion["doc_path"],
        )
        != (validated_source_path, validated_doc_path)
    ]
    assertions.append(
        {
            "source_path": validated_source_path,
            "doc_path": validated_doc_path,
            "recorded_source_hash": current_source_hash,
            "recorded_doc_hash": current_doc_hash,
            "producer": validated_producer,
            "producer_version": validated_producer_version,
        }
    )
    assertions.sort(key=lambda assertion: (assertion["source_path"], assertion["doc_path"]))

    updated = dict(existing)
    updated["schema_version"] = existing.get(
        "schema_version", DOC_PROVENANCE_SCHEMA_VERSION
    )
    updated["hash_algorithm"] = DOC_FRESHNESS_HASH_ALGORITHM
    updated["assertions"] = assertions
    write_doc_provenance_atomic(destination, updated)


# Explicit short aliases mirror the existing machine-index contract API and
# make the atomicity guarantee visible at call sites.
write_doc_freshness = write_doc_freshness_atomic
write_doc_provenance = write_doc_provenance_atomic


def hash_bytes(data: bytes) -> str:
    """Return the contract's SHA-256 representation for exact bytes.

    This tiny helper is useful to future producer and evaluator code.  It does
    not perform filesystem observation; that safety boundary belongs to the
    later safe-hashing work item.
    """

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "DIAGNOSTIC_CODES",
    "DOC_FRESHNESS_ENTRY_FIELDS",
    "DOC_FRESHNESS_HASH_ALGORITHM",
    "DOC_FRESHNESS_MAPPING_RULE",
    "DOC_FRESHNESS_SCHEMA_MAJOR",
    "DOC_FRESHNESS_SCHEMA_VERSION",
    "DOC_FRESHNESS_TOP_LEVEL_FIELDS",
    "FRESHNESS_TO_DOC_STATUS",
    "DOC_PROVENANCE_ASSERTION_FIELDS",
    "DOC_PROVENANCE_PRODUCER",
    "DOC_PROVENANCE_PRODUCER_VERSION",
    "DOC_PROVENANCE_SCHEMA_MAJOR",
    "DOC_PROVENANCE_SCHEMA_VERSION",
    "DOC_PROVENANCE_TOP_LEVEL_FIELDS",
    "CoverageTarget",
    "DocMapping",
    "DocFreshnessContractError",
    "DocFreshnessDocument",
    "DocFreshnessEntry",
    "DocProvenanceContractError",
    "DocProvenanceRecordingError",
    "DocProvenanceDocument",
    "HASH_CHUNK_SIZE",
    "HashObservation",
    "MappingResolution",
    "FreshnessReason",
    "FreshnessStatus",
    "FRESHNESS_REASONS",
    "FRESHNESS_STATUSES",
    "PROVENANCE_FAILURE_REASONS",
    "ProvenanceAssertion",
    "ProvenanceFailure",
    "ProvenanceReadResult",
    "canonical_doc_freshness_bytes",
    "canonical_doc_provenance_bytes",
    "evaluate_doc_freshness",
    "expected_doc_paths",
    "freshness_to_doc_statuses",
    "hash_bytes",
    "load_doc_freshness",
    "load_doc_provenance",
    "project_doc_freshness",
    "project_doc_statuses",
    "project_freshness",
    "read_doc_provenance",
    "record_doc_provenance",
    "serialize_doc_freshness",
    "serialize_doc_provenance",
    "resolve_doc_mappings",
    "safe_hash_regular_file",
    "write_text_regular_file",
    "validate_doc_freshness",
    "validate_doc_provenance",
    "validate_repository_relative_path",
    "validate_sha256",
    "write_doc_freshness",
    "write_doc_freshness_atomic",
    "write_doc_provenance",
    "write_doc_provenance_atomic",
]
