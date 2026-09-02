"""Structured attention signals and deterministic severity classification.

The machine-analysis pipeline collects repository facts in a normalized
``AttentionSignalSnapshot`` before calling :func:`classify_attention`.  This
module deliberately has no filesystem, Git, or previous-result dependencies;
the classifier only interprets the snapshot it receives.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal, TypedDict


JSONPrimitive = str | int | float | bool | None
Severity = Literal["critical", "high", "medium", "low"]
DocStatus = Literal["current", "missing", "stale", "unavailable"]
AttentionKind = Literal[
    "large_file",
    "test_missing",
    "todo_increase",
    "high_fan_in",
    "high_fan_out",
    "doc_missing",
    "doc_stale",
]
DiagnosticDetector = Literal["metadata", "test_index", "entrypoint", "doc_status"]


SEVERITY_ORDER: tuple[Severity, ...] = ("critical", "high", "medium", "low")
SEVERITY_RANK: dict[Severity, int] = {
    severity: rank for rank, severity in enumerate(SEVERITY_ORDER)
}
ATTENTION_KINDS: tuple[AttentionKind, ...] = (
    "large_file",
    "test_missing",
    "todo_increase",
    "high_fan_in",
    "high_fan_out",
    "doc_missing",
    "doc_stale",
)
DOC_STATUSES: tuple[DocStatus, ...] = (
    "current",
    "missing",
    "stale",
    "unavailable",
)

LINE_COUNT_THRESHOLD = 300
FAN_IN_THRESHOLD = 5
FAN_OUT_THRESHOLD = 15

ATTENTION_ENTRY_FIELDS = (
    "severity",
    "kind",
    "path",
    "reason",
    "evidence",
)

ATTENTION_REASONS: dict[AttentionKind, str] = {
    "large_file": "source file exceeds the line-count threshold",
    "test_missing": "Python source has no matching test file",
    "todo_increase": "TODO/FIXME count increased since the prior snapshot",
    "high_fan_in": "file is imported by at least the fan-in threshold",
    "high_fan_out": "file imports at least the fan-out threshold",
    "doc_missing": "coverage target has no matching documentation",
    "doc_stale": "matching documentation is older than changed source",
}


class AttentionEntry(TypedDict):
    """The exact public shape of one structured attention point."""

    severity: Severity
    kind: AttentionKind
    path: str
    reason: str
    evidence: dict[str, JSONPrimitive]


@dataclass(frozen=True, slots=True)
class AttentionSignalSnapshot:
    """Normalized, path-level facts consumed by the pure classifier.

    Defaults make small unit fixtures convenient, while callers that build a
    production snapshot should provide every value explicitly.  The
    ``todo_increased`` flag is the normalized detector result; counts are
    retained so that the resulting evidence remains inspectable.
    """

    path: str
    kind: str = "source"
    language: str = "python"
    readable: bool = True
    line_count: int | None = None
    fan_in: int = 0
    fan_out: int = 0
    test_missing: bool = False
    todo_current: int = 0
    todo_previous: int | None = None
    todo_increased: bool = False
    is_entrypoint: bool = False
    doc_status: DocStatus = "current"


@dataclass(frozen=True, slots=True)
class AttentionDiagnostic:
    """Stable detector failure information kept separate from findings."""

    detector: DiagnosticDetector
    code: str
    path: str | None = None


class AttentionContractError(ValueError):
    """Raised when normalized signals or attention entries violate the contract."""


def _fail(location: str, reason: str) -> None:
    raise AttentionContractError(f"{location}: {reason}")


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(location, "expected an object")
    for key in value:
        if not isinstance(key, str):
            _fail(location, "object keys must be strings")
    return value


def _require_string(value: Any, location: str, *, non_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(location, "expected a string")
    if non_empty and not value:
        _fail(location, "expected a non-empty string")
    return value


def _require_boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        _fail(location, "expected a boolean")
    return value


def _require_non_negative_integer(value: Any, location: str) -> int:
    # ``bool`` is an ``int`` subclass, but it is not valid numeric evidence.
    if type(value) is not int or value < 0:
        _fail(location, "expected a non-negative integer")
    return value


def validate_repository_relative_path(value: Any, *, location: str = "path") -> str:
    """Validate and return a normalized repository-relative POSIX path."""

    path = _require_string(value, location)
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


def _validate_snapshot(snapshot: Any, index: int) -> AttentionSignalSnapshot:
    location = f"snapshots[{index}]"
    if not isinstance(snapshot, AttentionSignalSnapshot):
        _fail(location, "expected an AttentionSignalSnapshot")

    validate_repository_relative_path(snapshot.path, location=f"{location}.path")
    _require_string(snapshot.kind, f"{location}.kind", non_empty=True)
    _require_string(snapshot.language, f"{location}.language", non_empty=True)
    _require_boolean(snapshot.readable, f"{location}.readable")

    if snapshot.line_count is not None:
        _require_non_negative_integer(snapshot.line_count, f"{location}.line_count")
    _require_non_negative_integer(snapshot.fan_in, f"{location}.fan_in")
    _require_non_negative_integer(snapshot.fan_out, f"{location}.fan_out")
    _require_boolean(snapshot.test_missing, f"{location}.test_missing")
    _require_non_negative_integer(snapshot.todo_current, f"{location}.todo_current")
    if snapshot.todo_previous is not None:
        _require_non_negative_integer(
            snapshot.todo_previous, f"{location}.todo_previous"
        )
    _require_boolean(snapshot.todo_increased, f"{location}.todo_increased")
    _require_boolean(snapshot.is_entrypoint, f"{location}.is_entrypoint")
    if snapshot.doc_status not in DOC_STATUSES:
        _fail(
            f"{location}.doc_status",
            f"unsupported document status: {snapshot.doc_status!r}",
        )
    return snapshot


def _make_entry(
    severity: Severity,
    kind: AttentionKind,
    path: str,
    evidence: dict[str, JSONPrimitive],
) -> AttentionEntry:
    return {
        "severity": severity,
        "kind": kind,
        "path": path,
        "reason": ATTENTION_REASONS[kind],
        "evidence": evidence,
    }


def _entries_for_snapshot(snapshot: AttentionSignalSnapshot) -> list[AttentionEntry]:
    """Build all signal entries for one already-validated snapshot."""

    is_source = snapshot.kind == "source"
    is_python_source = is_source and snapshot.language == "python"
    high_fan_in = snapshot.fan_in >= FAN_IN_THRESHOLD
    large_file = (
        is_source
        and snapshot.readable
        and snapshot.line_count is not None
        and snapshot.line_count > LINE_COUNT_THRESHOLD
    )

    entries: list[AttentionEntry] = []

    if large_file:
        entries.append(
            _make_entry(
                "critical" if high_fan_in else "medium",
                "large_file",
                snapshot.path,
                {
                    "line_count": snapshot.line_count,
                    "threshold_exclusive": LINE_COUNT_THRESHOLD,
                },
            )
        )

    if is_python_source and snapshot.test_missing:
        entries.append(
            _make_entry(
                "medium",
                "test_missing",
                snapshot.path,
                {"test_missing": True},
            )
        )

    if is_source and snapshot.readable and snapshot.todo_increased:
        entries.append(
            _make_entry(
                "low",
                "todo_increase",
                snapshot.path,
                {
                    "current_count": snapshot.todo_current,
                    "previous_count": snapshot.todo_previous,
                },
            )
        )

    if high_fan_in:
        entries.append(
            _make_entry(
                "critical" if large_file else "high",
                "high_fan_in",
                snapshot.path,
                {
                    "fan_in": snapshot.fan_in,
                    "threshold_inclusive": FAN_IN_THRESHOLD,
                    "is_entrypoint": snapshot.is_entrypoint,
                    "doc_status": snapshot.doc_status,
                },
            )
        )

    if snapshot.fan_out >= FAN_OUT_THRESHOLD:
        entries.append(
            _make_entry(
                "medium",
                "high_fan_out",
                snapshot.path,
                {
                    "fan_out": snapshot.fan_out,
                    "threshold_inclusive": FAN_OUT_THRESHOLD,
                },
            )
        )

    if is_source and snapshot.doc_status == "missing":
        missing_severity: Severity
        if snapshot.is_entrypoint:
            missing_severity = "critical"
        elif high_fan_in:
            missing_severity = "high"
        else:
            missing_severity = "medium"
        entries.append(
            _make_entry(
                missing_severity,
                "doc_missing",
                snapshot.path,
                {
                    "doc_status": "missing",
                    "is_entrypoint": snapshot.is_entrypoint,
                    "fan_in": snapshot.fan_in,
                },
            )
        )

    if is_source and snapshot.doc_status == "stale":
        entries.append(
            _make_entry(
                "high" if snapshot.is_entrypoint else "medium",
                "doc_stale",
                snapshot.path,
                {
                    "doc_status": "stale",
                    "is_entrypoint": snapshot.is_entrypoint,
                    "fan_in": snapshot.fan_in,
                },
            )
        )

    return entries


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[int, str, str]:
    return (SEVERITY_RANK[entry["severity"]], entry["path"], entry["kind"])


def _validate_evidence(
    kind: AttentionKind, evidence_value: Any, location: str
) -> dict[str, JSONPrimitive]:
    evidence = _require_mapping(evidence_value, location)
    expected_keys = {
        "large_file": {"line_count", "threshold_exclusive"},
        "test_missing": {"test_missing"},
        "todo_increase": {"current_count", "previous_count"},
        "high_fan_in": {
            "fan_in",
            "threshold_inclusive",
            "is_entrypoint",
            "doc_status",
        },
        "high_fan_out": {"fan_out", "threshold_inclusive"},
        "doc_missing": {"doc_status", "is_entrypoint", "fan_in"},
        "doc_stale": {"doc_status", "is_entrypoint", "fan_in"},
    }[kind]
    if set(evidence) != expected_keys:
        _fail(
            location,
            f"expected exactly evidence keys {sorted(expected_keys)!r}",
        )

    if kind == "large_file":
        _require_non_negative_integer(evidence["line_count"], f"{location}.line_count")
        threshold = _require_non_negative_integer(
            evidence["threshold_exclusive"], f"{location}.threshold_exclusive"
        )
        if threshold != LINE_COUNT_THRESHOLD:
            _fail(
                f"{location}.threshold_exclusive",
                f"expected {LINE_COUNT_THRESHOLD}",
            )
    elif kind == "test_missing":
        if _require_boolean(evidence["test_missing"], f"{location}.test_missing") is not True:
            _fail(f"{location}.test_missing", "must be true for a test_missing entry")
    elif kind == "todo_increase":
        _require_non_negative_integer(
            evidence["current_count"], f"{location}.current_count"
        )
        previous = evidence["previous_count"]
        if previous is not None:
            _require_non_negative_integer(previous, f"{location}.previous_count")
    elif kind == "high_fan_in":
        _require_non_negative_integer(evidence["fan_in"], f"{location}.fan_in")
        threshold = _require_non_negative_integer(
            evidence["threshold_inclusive"], f"{location}.threshold_inclusive"
        )
        if threshold != FAN_IN_THRESHOLD:
            _fail(
                f"{location}.threshold_inclusive",
                f"expected {FAN_IN_THRESHOLD}",
            )
        _require_boolean(evidence["is_entrypoint"], f"{location}.is_entrypoint")
        doc_status = _require_string(evidence["doc_status"], f"{location}.doc_status")
        if doc_status not in DOC_STATUSES:
            _fail(f"{location}.doc_status", "unsupported document status")
    elif kind == "high_fan_out":
        _require_non_negative_integer(evidence["fan_out"], f"{location}.fan_out")
        threshold = _require_non_negative_integer(
            evidence["threshold_inclusive"], f"{location}.threshold_inclusive"
        )
        if threshold != FAN_OUT_THRESHOLD:
            _fail(
                f"{location}.threshold_inclusive",
                f"expected {FAN_OUT_THRESHOLD}",
            )
    else:
        doc_status = _require_string(evidence["doc_status"], f"{location}.doc_status")
        expected_status = "missing" if kind == "doc_missing" else "stale"
        if doc_status != expected_status:
            _fail(
                f"{location}.doc_status",
                f"expected {expected_status!r}",
            )
        _require_boolean(evidence["is_entrypoint"], f"{location}.is_entrypoint")
        _require_non_negative_integer(evidence["fan_in"], f"{location}.fan_in")

    return dict(evidence)


def _validate_entry(value: Any, index: int) -> AttentionEntry:
    location = f"entries[{index}]"
    entry = _require_mapping(value, location)
    if set(entry) != set(ATTENTION_ENTRY_FIELDS):
        _fail(
            location,
            f"expected exactly fields {list(ATTENTION_ENTRY_FIELDS)!r}",
        )

    severity = _require_string(entry["severity"], f"{location}.severity")
    if severity not in SEVERITY_RANK:
        _fail(f"{location}.severity", f"unsupported severity: {severity!r}")
    kind = _require_string(entry["kind"], f"{location}.kind")
    if kind not in ATTENTION_KINDS:
        _fail(f"{location}.kind", f"unsupported attention kind: {kind!r}")
    path = validate_repository_relative_path(entry["path"], location=f"{location}.path")
    reason = _require_string(entry["reason"], f"{location}.reason", non_empty=True)
    if reason != ATTENTION_REASONS[kind]:
        _fail(f"{location}.reason", "does not match the stable reason for kind")
    evidence = _validate_evidence(kind, entry["evidence"], f"{location}.evidence")

    return {
        "severity": severity,
        "kind": kind,
        "path": path,
        "reason": reason,
        "evidence": evidence,
    }


def _validated_entries(entries: Sequence[Mapping[str, Any]]) -> list[AttentionEntry]:
    if not isinstance(entries, Sequence):
        raise AttentionContractError("entries: expected an array")

    validated: list[AttentionEntry] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(entries):
        entry = _validate_entry(value, index)
        identity = (entry["path"], entry["kind"])
        if identity in seen:
            _fail(
                f"entries[{index}]",
                f"duplicate attention identity: {identity!r}",
            )
        seen.add(identity)
        validated.append(entry)
    return validated


def _deduplicate_candidates(entries: Sequence[AttentionEntry]) -> list[AttentionEntry]:
    by_identity: dict[tuple[str, str], AttentionEntry] = {}
    for index, candidate in enumerate(entries):
        entry = _validate_entry(candidate, index)
        identity = (entry["path"], entry["kind"])
        existing = by_identity.get(identity)
        if existing is None:
            by_identity[identity] = entry
        elif existing != entry:
            _fail(
                f"entries[{index}]",
                f"conflicting attention identity: {identity!r}",
            )
        # Exact duplicate candidates are intentionally ignored.
    return sorted(by_identity.values(), key=_entry_sort_key)


def classify_attention(
    snapshots: Sequence[AttentionSignalSnapshot],
) -> list[AttentionEntry]:
    """Classify normalized snapshots into canonical, structured entries.

    The input sequence is never modified.  Snapshot paths must be unique;
    repeated identical snapshots are harmless and are deduplicated.  A
    conflicting repeated path is rejected instead of being resolved by input
    order.
    """

    if not isinstance(snapshots, Sequence):
        raise AttentionContractError("snapshots: expected an array")

    snapshots_by_path: dict[str, AttentionSignalSnapshot] = {}
    for index, value in enumerate(snapshots):
        snapshot = _validate_snapshot(value, index)
        existing = snapshots_by_path.get(snapshot.path)
        if existing is not None and existing != snapshot:
            _fail(
                f"snapshots[{index}].path",
                f"conflicting normalized snapshot for path {snapshot.path!r}",
            )
        snapshots_by_path.setdefault(snapshot.path, snapshot)

    candidates: list[AttentionEntry] = []
    for snapshot in snapshots_by_path.values():
        candidates.extend(_entries_for_snapshot(snapshot))

    return _deduplicate_candidates(candidates)


def sort_attention_entries(
    entries: Sequence[Mapping[str, Any]],
) -> list[AttentionEntry]:
    """Validate entries and return them in canonical severity/path/kind order."""

    return sorted(_validated_entries(entries), key=_entry_sort_key)


def validate_attention(entries: Sequence[Mapping[str, Any]]) -> None:
    """Validate exact entry shape, evidence, identity, and canonical order."""

    validated = _validated_entries(entries)
    expected = sorted(validated, key=_entry_sort_key)
    if validated != expected:
        _fail("entries", "entries are not in canonical severity/path/kind order")


def serialize_attention(entries: Sequence[Mapping[str, Any]]) -> str:
    """Return canonical UTF-8 JSON text for a structured attention array."""

    ordered = sort_attention_entries(entries)
    try:
        encoded = json.dumps(
            ordered,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AttentionContractError(f"entries: cannot serialize JSON: {exc}") from exc
    return encoded + "\n"


def canonical_attention_bytes(entries: Sequence[Mapping[str, Any]]) -> bytes:
    """Return canonical UTF-8 JSON bytes for structured attention entries."""

    return serialize_attention(entries).encode("utf-8")


# Short alias useful to callers that already use the machine-index terminology.
canonical_json_bytes = canonical_attention_bytes


__all__ = [
    "ATTENTION_ENTRY_FIELDS",
    "ATTENTION_KINDS",
    "ATTENTION_REASONS",
    "AttentionContractError",
    "AttentionDiagnostic",
    "AttentionEntry",
    "AttentionKind",
    "AttentionSignalSnapshot",
    "DOC_STATUSES",
    "DocStatus",
    "FAN_IN_THRESHOLD",
    "FAN_OUT_THRESHOLD",
    "JSONPrimitive",
    "LINE_COUNT_THRESHOLD",
    "SEVERITY_ORDER",
    "SEVERITY_RANK",
    "Severity",
    "canonical_attention_bytes",
    "canonical_json_bytes",
    "classify_attention",
    "serialize_attention",
    "sort_attention_entries",
    "validate_attention",
    "validate_repository_relative_path",
]
