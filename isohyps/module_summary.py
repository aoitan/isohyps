"""Deterministic, evidence-carrying summaries for source modules.

The module deliberately owns the summary contract and its pure generation
helpers.  Parsing and source-fact extraction belong to the machine-analysis
pipeline; keeping I/O separate lets readers validate or project a summary
without importing the analyzer.
"""

from __future__ import annotations

import copy
import html
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict


SummaryParser = Literal["python_ast", "tree_sitter", "regex", "none"]
SummaryOutcome = Literal[
    "ok",
    "read_error",
    "decode_error",
    "parse_error",
    "unsupported",
    "binary_skipped",
    "source_changed",
]
SummaryMethod = Literal["module_docstring", "structural_facts", "unknown"]
SummaryEvidenceKind = Literal[
    "module_docstring", "definition", "entrypoint_candidate"
]
SummaryEvidenceOrigin = Literal[
    "python_ast",
    "tree_sitter",
    "regex",
    "attention_entrypoint_resolver_v1",
]
SummaryUnknownReason = Literal[
    "read_error",
    "decode_error",
    "parse_error",
    "unsupported",
    "binary_skipped",
    "source_changed",
    "insufficient_evidence",
]


MODULE_SUMMARY_FIELDS = (
    "text",
    "method",
    "reason",
    "parser",
    "evidence",
    "omitted_evidence_count",
    "text_truncated",
)
MODULE_SUMMARY_EVIDENCE_FIELDS = (
    "kind",
    "origin",
    "line",
    "value",
    "value_truncated",
    "paragraphs_omitted",
)

SUMMARY_METHODS: tuple[SummaryMethod, ...] = (
    "module_docstring",
    "structural_facts",
    "unknown",
)
SUMMARY_PARSERS: tuple[SummaryParser, ...] = (
    "python_ast",
    "tree_sitter",
    "regex",
    "none",
)
SUMMARY_OUTCOMES: tuple[SummaryOutcome, ...] = (
    "ok",
    "read_error",
    "decode_error",
    "parse_error",
    "unsupported",
    "binary_skipped",
    "source_changed",
)
SUMMARY_FAILURE_OUTCOMES: tuple[SummaryOutcome, ...] = (
    "read_error",
    "decode_error",
    "parse_error",
    "unsupported",
    "binary_skipped",
    "source_changed",
)
SUMMARY_UNKNOWN_REASONS: tuple[SummaryUnknownReason, ...] = (
    *SUMMARY_FAILURE_OUTCOMES,
    "insufficient_evidence",
)
SUMMARY_EVIDENCE_KINDS: tuple[SummaryEvidenceKind, ...] = (
    "module_docstring",
    "definition",
    "entrypoint_candidate",
)
SUMMARY_EVIDENCE_ORIGINS: tuple[SummaryEvidenceOrigin, ...] = (
    "python_ast",
    "tree_sitter",
    "regex",
    "attention_entrypoint_resolver_v1",
)

MODULE_SUMMARY_MAX_TEXT_LENGTH = 240
MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH = 160
MODULE_SUMMARY_MAX_EVIDENCE = 6
MODULE_SUMMARY_MAX_BYTES = 16 * 1024
MODULE_SUMMARY_MAX_INTEGER = 2**63 - 1
MODULE_SUMMARY_MAX_DEFINITIONS = 5
MODULE_SUMMARY_MAX_DEFINITION_NAME_LENGTH = 24
MODULE_SUMMARY_ELLIPSIS = "…"
ENTRYPOINT_EVIDENCE_VALUE = "Entrypoint candidate detected."
LEGACY_SUMMARY_NOT_AVAILABLE = "not available (legacy input)"

_MODULE_SUMMARY_METHOD_LABELS = {
    "module_docstring": "Module docstring excerpt (unverified)",
    "structural_facts": "Structural facts",
    "unknown": "Unknown",
}
_HTML_ESCAPED_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|#x27);")
_HTML_ESCAPED_ENTITY_REFERENCES = {
    "&amp;": "&#38;",
    "&lt;": "&#60;",
    "&gt;": "&#62;",
    "&quot;": "&#34;",
    "&#x27;": "&#39;",
}
# These characters can change the meaning of a value when the generated
# fragment is subsequently embedded in Markdown.  Numeric references still
# render as the original characters in the browser, while keeping the value
# out of Markdown syntax and HTML markup.
_MARKDOWN_TEXT_REFERENCES = {
    "\\": "&#92;",
    "`": "&#96;",
    "*": "&#42;",
    "_": "&#95;",
    "{": "&#123;",
    "}": "&#125;",
    "[": "&#91;",
    "]": "&#93;",
    "(": "&#40;",
    ")": "&#41;",
    "#": "&#35;",
    "+": "&#43;",
    "!": "&#33;",
    "|": "&#124;",
    ">": "&#62;",
    "~": "&#126;",
}

# Short aliases make the limits convenient to use from the generator and
# focused tests without duplicating the contract's numbers.
SUMMARY_TEXT_MAX_LENGTH = MODULE_SUMMARY_MAX_TEXT_LENGTH
SUMMARY_EVIDENCE_VALUE_MAX_LENGTH = MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH
SUMMARY_EVIDENCE_MAX_COUNT = MODULE_SUMMARY_MAX_EVIDENCE
SUMMARY_MAX_BYTES = MODULE_SUMMARY_MAX_BYTES
SUMMARY_MAX_INTEGER = MODULE_SUMMARY_MAX_INTEGER
SUMMARY_DEFINITION_MAX_COUNT = MODULE_SUMMARY_MAX_DEFINITIONS
SUMMARY_DEFINITION_NAME_MAX_LENGTH = MODULE_SUMMARY_MAX_DEFINITION_NAME_LENGTH


class SummaryDocstring(TypedDict):
    """A module-level docstring observation retained by the analyzer."""

    text: str
    line: int


class SummaryDefinition(TypedDict):
    """A definition candidate retained by the analyzer."""

    name: str
    kind: str
    line: int | None


class ModuleSummaryEvidence(TypedDict):
    """The exact public shape of one summary evidence item."""

    kind: SummaryEvidenceKind
    origin: SummaryEvidenceOrigin
    line: int | None
    value: str
    value_truncated: bool
    paragraphs_omitted: bool


class ModuleSummary(TypedDict):
    """The exact public shape of a source module summary."""

    text: str
    method: SummaryMethod
    reason: SummaryUnknownReason | None
    parser: SummaryParser
    evidence: list[ModuleSummaryEvidence]
    omitted_evidence_count: int
    text_truncated: bool


SummaryEvidence = ModuleSummaryEvidence


@dataclass(frozen=True, slots=True)
class SummaryFacts:
    """Internal facts passed from extraction to summary generation.

    The dataclass is intentionally not a public JSON shape.  ``definitions``
    is a sequence rather than a list so callers can provide either parser
    output or an immutable fixture without changing the summary contract.
    """

    parser: SummaryParser = "none"
    outcome: SummaryOutcome = "ok"
    docstring: SummaryDocstring | None = None
    definitions: Sequence[SummaryDefinition] = ()


class ModuleSummaryContractError(ValueError):
    """Raised when a module summary violates its public contract."""


_MISSING = object()


def _fail(location: str, reason: str) -> None:
    raise ModuleSummaryContractError(f"{location}: {reason}")


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(location, "expected an object")
    for key in value:
        if not isinstance(key, str):
            _fail(location, "object keys must be strings")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(location, "expected an array")
    return value


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    value = mapping.get(key, _MISSING)
    if value is _MISSING:
        _fail(f"{location}.{key}", "required field is missing")
    return value


def _require_string(
    value: Any, location: str, *, non_empty: bool = False
) -> str:
    if not isinstance(value, str):
        _fail(location, "expected a string")
    if non_empty and not value:
        _fail(location, "expected a non-empty string")
    return value


def _require_boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        _fail(location, "expected a boolean")
    return value


def _require_bounded_integer(
    value: Any, location: str, *, minimum: int = 0
) -> int:
    # ``bool`` is an ``int`` subclass, but is not valid numeric evidence.
    if type(value) is not int or value < minimum or value > MODULE_SUMMARY_MAX_INTEGER:
        _fail(
            location,
            f"expected an integer from {minimum} to {MODULE_SUMMARY_MAX_INTEGER}",
        )
    return value


def _require_enum(value: Any, allowed: Sequence[str], location: str) -> str:
    value = _require_string(value, location)
    if value not in allowed:
        _fail(location, f"unsupported value: {value!r}")
    return value


def normalize_summary_text(value: Any, *, location: str = "text") -> str:
    """Normalize source-derived text without assigning it new meaning.

    Unicode whitespace is folded to one ASCII space, runs of spaces are
    collapsed, and remaining Unicode control/format/surrogate characters are
    replaced with U+FFFD.  Unicode normalization is intentionally not applied;
    code-point spelling remains source-observable.
    """

    text = _require_string(value, location)
    normalized: list[str] = []
    for character in text:
        if character.isspace():
            normalized.append(" ")
            continue
        if unicodedata.category(character) in {"Cc", "Cf", "Cs"}:
            normalized.append("\ufffd")
            continue
        normalized.append(character)
    return " ".join("".join(normalized).split())


def truncate_summary_text(
    value: Any, limit: int, *, location: str = "text"
) -> tuple[str, bool]:
    """Return normalized text bounded by ``limit`` code points."""

    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    text = normalize_summary_text(value, location=location)
    if len(text) <= limit:
        return text, False
    return text[: limit - 1] + MODULE_SUMMARY_ELLIPSIS, True


def _normalized_bounded_string(
    value: Any, location: str, *, maximum: int
) -> str:
    text = normalize_summary_text(value, location=location)
    if not text:
        _fail(location, "expected a non-empty normalized string")
    if len(text) > maximum:
        _fail(location, f"must be at most {maximum} Unicode code points")
    return text


def _copy_known_fields(value: Any, *, normalize: bool) -> dict[str, Any]:
    source = _require_mapping(value, "module_summary")
    projected: dict[str, Any] = {}
    for field in MODULE_SUMMARY_FIELDS:
        projected[field] = copy.deepcopy(_required(source, field, "module_summary"))

    evidence = _require_list(projected["evidence"], "module_summary.evidence")
    projected_evidence: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        location = f"module_summary.evidence[{index}]"
        item_mapping = _require_mapping(item, location)
        projected_evidence.append(
            {
                field: copy.deepcopy(_required(item_mapping, field, location))
                for field in MODULE_SUMMARY_EVIDENCE_FIELDS
            }
        )
    projected["evidence"] = projected_evidence

    if normalize:
        projected["text"] = normalize_summary_text(
            projected["text"], location="module_summary.text"
        )
        for index, item in enumerate(projected_evidence):
            item["value"] = normalize_summary_text(
                item["value"],
                location=f"module_summary.evidence[{index}].value",
            )
    return projected


def _validate_evidence(
    value: Any, index: int, *, parser: str
) -> dict[str, Any]:
    location = f"module_summary.evidence[{index}]"
    evidence = _require_mapping(value, location)
    kind = _require_enum(
        _required(evidence, "kind", location), SUMMARY_EVIDENCE_KINDS, f"{location}.kind"
    )
    origin = _require_enum(
        _required(evidence, "origin", location),
        SUMMARY_EVIDENCE_ORIGINS,
        f"{location}.origin",
    )
    line = _required(evidence, "line", location)
    if line is not None:
        line = _require_bounded_integer(line, f"{location}.line", minimum=1)
    value_text = _normalized_bounded_string(
        _required(evidence, "value", location),
        f"{location}.value",
        maximum=MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH,
    )
    value_truncated = _require_boolean(
        _required(evidence, "value_truncated", location),
        f"{location}.value_truncated",
    )
    paragraphs_omitted = _require_boolean(
        _required(evidence, "paragraphs_omitted", location),
        f"{location}.paragraphs_omitted",
    )

    if kind == "module_docstring":
        if origin != "python_ast":
            _fail(
                f"{location}.origin",
                "module_docstring evidence must originate from python_ast",
            )
        if line is None:
            _fail(f"{location}.line", "module_docstring evidence requires a line")
        if parser != "python_ast":
            _fail(
                "module_summary.parser",
                "module_docstring evidence requires parser 'python_ast'",
            )
    elif kind == "definition":
        expected_origin = {
            "python_ast": "python_ast",
            "tree_sitter": "tree_sitter",
            "regex": "regex",
        }.get(parser)
        if expected_origin is None or origin != expected_origin:
            _fail(
                f"{location}.origin",
                "definition evidence origin must match the summary parser",
            )
    else:
        if origin != "attention_entrypoint_resolver_v1":
            _fail(
                f"{location}.origin",
                "entrypoint_candidate evidence must use the resolver origin",
            )
        if line is not None:
            _fail(
                f"{location}.line",
                "entrypoint_candidate evidence does not have a source line",
            )
        if value_text != ENTRYPOINT_EVIDENCE_VALUE:
            _fail(
                f"{location}.value",
                f"expected {ENTRYPOINT_EVIDENCE_VALUE!r} for entrypoint evidence",
            )

    if kind != "module_docstring" and paragraphs_omitted:
        _fail(
            f"{location}.paragraphs_omitted",
            "paragraph omission only applies to module_docstring evidence",
        )

    return {
        "kind": kind,
        "origin": origin,
        "line": line,
        "value": value_text,
        "value_truncated": value_truncated,
        "paragraphs_omitted": paragraphs_omitted,
    }


def validate_module_summary(value: Mapping[str, Any]) -> None:
    """Validate a module summary and raise on contract violations.

    Unknown fields are tolerated for reader-side additive compatibility.  The
    public required fields, evidence shape, enum combinations, and
    method-specific invariants remain strict.
    """

    summary = _require_mapping(value, "module_summary")
    text = _normalized_bounded_string(
        _required(summary, "text", "module_summary"),
        "module_summary.text",
        maximum=MODULE_SUMMARY_MAX_TEXT_LENGTH,
    )
    method = _require_enum(
        _required(summary, "method", "module_summary"),
        SUMMARY_METHODS,
        "module_summary.method",
    )
    reason = _required(summary, "reason", "module_summary")
    parser = _require_enum(
        _required(summary, "parser", "module_summary"),
        SUMMARY_PARSERS,
        "module_summary.parser",
    )
    evidence_values = _require_list(
        _required(summary, "evidence", "module_summary"),
        "module_summary.evidence",
    )
    if len(evidence_values) > MODULE_SUMMARY_MAX_EVIDENCE:
        _fail(
            "module_summary.evidence",
            f"must contain at most {MODULE_SUMMARY_MAX_EVIDENCE} items",
        )
    omitted_evidence_count = _require_bounded_integer(
        _required(summary, "omitted_evidence_count", "module_summary"),
        "module_summary.omitted_evidence_count",
    )
    text_truncated = _require_boolean(
        _required(summary, "text_truncated", "module_summary"),
        "module_summary.text_truncated",
    )

    validated_evidence = [
        _validate_evidence(item, index, parser=parser)
        for index, item in enumerate(evidence_values)
    ]

    if method in ("module_docstring", "structural_facts"):
        if reason is not None:
            _fail(
                "module_summary.reason",
                f"{method} summaries must have reason=null",
            )
    else:
        if reason not in SUMMARY_UNKNOWN_REASONS:
            _fail(
                "module_summary.reason",
                "unknown summaries require a recognized missing-evidence reason",
            )
        if text != "unknown":
            _fail("module_summary.text", "unknown summaries must use text 'unknown'")
        if validated_evidence:
            _fail("module_summary.evidence", "unknown summaries cannot contain evidence")
        if omitted_evidence_count != 0:
            _fail(
                "module_summary.omitted_evidence_count",
                "unknown summaries cannot omit evidence",
            )
        if text_truncated:
            _fail(
                "module_summary.text_truncated",
                "unknown summaries cannot be truncated",
            )
        return

    docstring_evidence = [
        item for item in validated_evidence if item["kind"] == "module_docstring"
    ]
    if method == "module_docstring":
        if len(validated_evidence) != 1 or len(docstring_evidence) != 1:
            _fail(
                "module_summary.evidence",
                "module_docstring summaries require exactly one docstring evidence item",
            )
        item = docstring_evidence[0]
        if text != item["value"]:
            _fail(
                "module_summary.text",
                "must equal the module_docstring evidence value",
            )
        if text_truncated != item["value_truncated"]:
            _fail(
                "module_summary.text_truncated",
                "must match module_docstring evidence value_truncated",
            )
        if omitted_evidence_count != 0:
            _fail(
                "module_summary.omitted_evidence_count",
                "docstring summaries do not count rejected structural evidence",
            )
    else:
        if docstring_evidence:
            _fail(
                "module_summary.evidence",
                "structural summaries cannot contain module_docstring evidence",
            )
        if not validated_evidence:
            _fail(
                "module_summary.evidence",
                "structural summaries require at least one evidence item",
            )
        if not any(
            item["kind"] in {"definition", "entrypoint_candidate"}
            for item in validated_evidence
        ):
            _fail(
                "module_summary.evidence",
                "structural summaries require definition or entrypoint evidence",
            )

    # Validate the canonical public projection as well.  This removes
    # additive reader-only fields from the byte-size calculation and prevents
    # a valid-looking summary from exceeding the bounded JSON contract.
    projected = _copy_known_fields(summary, normalize=False)
    try:
        encoded = json.dumps(
            projected,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        encoded_bytes = (encoded + "\n").encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ModuleSummaryContractError(
            f"module_summary: cannot serialize canonical JSON: {exc}"
        ) from exc
    if len(encoded_bytes) > MODULE_SUMMARY_MAX_BYTES:
        _fail(
            "module_summary",
            f"canonical UTF-8 JSON must be at most {MODULE_SUMMARY_MAX_BYTES} bytes",
        )


def _fact_value(fact: Any, key: str, default: Any = _MISSING) -> Any:
    """Read a fact field from the typed mapping used by parser adapters."""

    if isinstance(fact, Mapping):
        value = fact.get(key, default)
    else:
        value = getattr(fact, key, default)
    return value


def _valid_fact_parser(value: Any) -> SummaryParser:
    if isinstance(value, str) and value in SUMMARY_PARSERS:
        return value  # type: ignore[return-value]
    # Invalid internal facts cannot safely identify a parser.  Treating them
    # as parser-less keeps the public result valid without inventing evidence.
    return "none"


def _valid_fact_outcome(value: Any) -> SummaryOutcome | None:
    if isinstance(value, str) and value in SUMMARY_OUTCOMES:
        return value  # type: ignore[return-value]
    return None


def _positive_fact_line(value: Any) -> int | None:
    if type(value) is int and 1 <= value <= MODULE_SUMMARY_MAX_INTEGER:
        return value
    return None


def _fact_paragraphs(value: Any) -> list[str]:
    """Split a raw module docstring into non-empty logical paragraphs."""

    if not isinstance(value, str):
        return []

    paragraphs: list[str] = []
    current: list[str] = []
    for line in value.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


def _module_docstring_excerpt(
    value: Any,
) -> tuple[str, bool] | None:
    """Return the first non-empty docstring paragraph and omission flag."""

    paragraphs = _fact_paragraphs(value)
    normalized = [normalize_summary_text(paragraph) for paragraph in paragraphs]
    non_empty = [paragraph for paragraph in normalized if paragraph]
    if not non_empty:
        return None
    return non_empty[0], len(non_empty) > 1


def _definition_fact_key(fact: Any) -> tuple[str, str, int] | None:
    """Return a normalized, sortable definition fact.

    The three values are the complete de-duplication key required by the
    summary rules.  ``0`` is the internal sort sentinel for an unknown line.
    """

    raw_name = _fact_value(fact, "name")
    if not isinstance(raw_name, str):
        return None
    name = normalize_summary_text(raw_name)
    if not name or name.startswith("_"):
        return None

    raw_kind = _fact_value(fact, "kind", "")
    kind = normalize_summary_text(raw_kind) if isinstance(raw_kind, str) else ""
    line = _positive_fact_line(_fact_value(fact, "line"))
    line_sort_key = line or 0
    return name, kind, line_sort_key


def _sorted_definition_facts(facts: Any) -> list[tuple[str, str, int | None]]:
    """Normalize, de-duplicate, and stably sort definition observations."""

    definitions = _fact_value(facts, "definitions", ())
    if isinstance(definitions, (str, bytes)):
        return []
    try:
        candidates = iter(definitions)
    except TypeError:
        return []

    unique: set[tuple[str, str, int]] = set()
    for fact in candidates:
        key = _definition_fact_key(fact)
        if key is None:
            continue
        name, kind, line_sort_key = key
        unique.add((name, kind, line_sort_key))

    return [
        (name, kind, line or None)
        for name, kind, line in sorted(unique, key=lambda item: item)
    ]


def _unknown_module_summary(
    parser: SummaryParser, reason: SummaryUnknownReason
) -> ModuleSummary:
    return {
        "text": "unknown",
        "method": "unknown",
        "reason": reason,
        "parser": parser,
        "evidence": [],
        "omitted_evidence_count": 0,
        "text_truncated": False,
    }


def _definition_evidence(
    name: str, line: int | None, parser: SummaryParser
) -> ModuleSummaryEvidence:
    value, value_truncated = truncate_summary_text(
        name,
        MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH,
        location="definition.name",
    )
    origin = parser
    # The caller only creates this evidence for one of the three parser-backed
    # modes.  Keeping this branch explicit makes the origin/parser invariant
    # obvious to readers and protects the contract if parser literals expand.
    if origin not in {"python_ast", "tree_sitter", "regex"}:
        raise ModuleSummaryContractError(
            "definition evidence requires a parser-backed summary"
        )
    return {
        "kind": "definition",
        "origin": origin,
        "line": line,
        "value": value,
        "value_truncated": value_truncated,
        "paragraphs_omitted": False,
    }


def _structural_summary(
    *,
    parser: SummaryParser,
    definitions: Sequence[tuple[str, str, int | None]],
    entrypoint_candidate: bool,
    omitted_definition_count: int = 0,
) -> ModuleSummary:
    """Build a bounded structural summary for one selected definition set."""

    selected = list(definitions[:MODULE_SUMMARY_MAX_DEFINITIONS])
    omitted = max(0, len(definitions) - len(selected)) + omitted_definition_count
    evidence: list[ModuleSummaryEvidence] = []
    text_parts: list[str] = []
    text_truncated = False

    if entrypoint_candidate:
        evidence.append(
            {
                "kind": "entrypoint_candidate",
                "origin": "attention_entrypoint_resolver_v1",
                "line": None,
                "value": ENTRYPOINT_EVIDENCE_VALUE,
                "value_truncated": False,
                "paragraphs_omitted": False,
            }
        )
        text_parts.append(ENTRYPOINT_EVIDENCE_VALUE)

    if selected:
        if parser == "python_ast":
            prefix = "Non-underscore top-level definitions:"
        else:
            prefix = f"Definition candidates ({parser}):"
        names: list[str] = []
        for name, _kind, line in selected:
            display_name, display_truncated = truncate_summary_text(
                name,
                MODULE_SUMMARY_MAX_DEFINITION_NAME_LENGTH,
                location="definition.name",
            )
            names.append(display_name)
            text_truncated = text_truncated or display_truncated
            evidence.append(_definition_evidence(name, line, parser))
        name_list = ", ".join(names)
        if omitted:
            name_list += f" (+{omitted} more)"
        text_parts.append(f"{prefix} {name_list}.")

    text, whole_text_truncated = truncate_summary_text(
        " ".join(text_parts),
        MODULE_SUMMARY_MAX_TEXT_LENGTH,
        location="module_summary.text",
    )
    text_truncated = text_truncated or whole_text_truncated
    return {
        "text": text,
        "method": "structural_facts",
        "reason": None,
        "parser": parser,
        "evidence": evidence,
        "omitted_evidence_count": omitted,
        "text_truncated": text_truncated,
    }


def build_module_summary(
    facts: SummaryFacts, *, entrypoint_candidate: bool = False
) -> ModuleSummary:
    """Build a deterministic, bounded summary from extracted observations.

    This function only quotes an observed module docstring or describes
    structural observations.  It never infers a business responsibility from
    imports or identifier vocabulary.  Invalid internal fact values are
    treated conservatively as missing evidence so the public result remains a
    valid summary object.
    """

    parser = _valid_fact_parser(_fact_value(facts, "parser", "none"))
    outcome = _valid_fact_outcome(_fact_value(facts, "outcome", "ok"))

    if outcome is None:
        return _unknown_module_summary(parser, "insufficient_evidence")
    if outcome in SUMMARY_FAILURE_OUTCOMES and outcome != "unsupported":
        return _unknown_module_summary(parser, outcome)  # type: ignore[arg-type]

    is_entrypoint = entrypoint_candidate is True
    if outcome == "unsupported":
        if not is_entrypoint:
            return _unknown_module_summary(parser, "unsupported")
        # An unsupported scan cannot provide a module docstring, but any
        # parser-backed definition facts already retained by an adapter remain
        # direct observations.  The resolver candidate is the minimum
        # condition that permits this limited structural fallback.
        definitions = _sorted_definition_facts(facts)
        if parser not in {"python_ast", "tree_sitter", "regex"}:
            definitions = []
    else:
        docstring = _fact_value(facts, "docstring")
        docstring_line = _positive_fact_line(_fact_value(docstring, "line"))
        docstring_text = _fact_value(docstring, "text")
        if parser == "python_ast" and docstring_line is not None:
            excerpt = _module_docstring_excerpt(docstring_text)
            if excerpt is not None:
                text, paragraphs_omitted = excerpt
                text, text_truncated = truncate_summary_text(
                    text,
                    MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH,
                    location="module_summary.text",
                )
                summary: ModuleSummary = {
                    "text": text,
                    "method": "module_docstring",
                    "reason": None,
                    "parser": "python_ast",
                    "evidence": [
                        {
                            "kind": "module_docstring",
                            "origin": "python_ast",
                            "line": docstring_line,
                            "value": text,
                            "value_truncated": text_truncated,
                            "paragraphs_omitted": paragraphs_omitted,
                        }
                    ],
                    "omitted_evidence_count": 0,
                    "text_truncated": text_truncated,
                }
                validate_module_summary(summary)
                return summary

        definitions = _sorted_definition_facts(facts)
        if parser not in {"python_ast", "tree_sitter", "regex"}:
            definitions = []

    if not is_entrypoint and not definitions:
        reason: SummaryUnknownReason = (
            "unsupported" if outcome == "unsupported" else "insufficient_evidence"
        )
        return _unknown_module_summary(parser, reason)

    # The fixed field/evidence limits make this normally a single pass.  Keep
    # the reduction loop so the byte bound remains true if Unicode-heavy input
    # or future contract constants make the candidate shape larger.
    candidate_definitions = definitions
    omitted_definition_count = 0
    while True:
        summary = _structural_summary(
            parser=parser,
            definitions=candidate_definitions,
            entrypoint_candidate=is_entrypoint,
            omitted_definition_count=omitted_definition_count,
        )
        try:
            canonical_module_summary_bytes(summary)
        except ModuleSummaryContractError as exc:
            if not candidate_definitions or "at most" not in str(exc):
                raise
            candidate_definitions = candidate_definitions[:-1]
            omitted_definition_count += 1
            continue
        return summary


def normalize_module_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return an immutable-input-safe normalized public summary projection."""

    normalized = _copy_known_fields(value, normalize=True)
    validate_module_summary(normalized)
    return normalized


def project_module_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project and normalize only the seven public summary fields.

    Unknown keys are ignored at both summary and evidence levels.  The input
    mapping is never modified; nested known values are copied before the
    source-derived text normalization is applied.
    """

    return normalize_module_summary(value)


def _canonical_bytes_for_projected(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        encoded_bytes = (encoded + "\n").encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ModuleSummaryContractError(
            f"module_summary: cannot serialize canonical JSON: {exc}"
        ) from exc
    if len(encoded_bytes) > MODULE_SUMMARY_MAX_BYTES:
        _fail(
            "module_summary",
            f"canonical UTF-8 JSON must be at most {MODULE_SUMMARY_MAX_BYTES} bytes",
        )
    return encoded_bytes


def canonical_module_summary_bytes(value: Mapping[str, Any]) -> bytes:
    """Return canonical UTF-8 JSON bytes for a valid projected summary."""

    projected = project_module_summary(value)
    return _canonical_bytes_for_projected(projected)


def serialize_module_summary(value: Mapping[str, Any]) -> str:
    """Return canonical JSON text for a valid projected summary."""

    return canonical_module_summary_bytes(value).decode("utf-8")


def _escape_summary_text_node(value: Any, *, markdown_safe: bool = False) -> str:
    """Escape one value for insertion into a fixed HTML text node.

    ``html.escape`` is deliberately applied before the optional Markdown pass.
    HTML entities emitted by it are protected while the remaining Markdown
    punctuation is converted to numeric character references.  This avoids
    turning a source value such as ``</span>`` or ``|`` into markup/table
    syntax, while the rendered text remains the same to a human reader.
    """

    raw = value if isinstance(value, str) else str(value)
    # Paths are also untrusted display data.  The same normalization used by
    # the summary contract keeps newlines and control characters from
    # creating additional visual structure in the generated fragment.
    normalized = normalize_summary_text(raw, location="module_summary.display")
    escaped = html.escape(normalized, quote=True)
    if not markdown_safe:
        return escaped

    entities = [match.group(0) for match in _HTML_ESCAPED_ENTITY_RE.finditer(escaped)]
    parts = _HTML_ESCAPED_ENTITY_RE.split(escaped)
    rendered: list[str] = []
    for index, part in enumerate(parts):
        rendered.append(
            "".join(_MARKDOWN_TEXT_REFERENCES.get(character, character) for character in part)
        )
        if index < len(entities):
            rendered.append(_HTML_ESCAPED_ENTITY_REFERENCES[entities[index]])
    return "".join(rendered)


def render_module_summary(
    value: Mapping[str, Any] | None, *, path: str
) -> str:
    """Render a summary as a fixed, safe HTML fragment.

    The caller passes ``None`` when an older machine-index entry has no
    ``module_summary`` field.  Such absence is kept distinct from an explicit
    ``method == "unknown"`` summary.  For a present value, the public
    projection is validated before any value is rendered so malformed data
    cannot bypass the summary contract.

    All caller-controlled values are text nodes.  In particular, no value is
    used in an element name, attribute, URL, or Markdown link.
    """

    legacy_input = value is None
    if legacy_input:
        method_label = LEGACY_SUMMARY_NOT_AVAILABLE
        method_id = None
        summary_text = LEGACY_SUMMARY_NOT_AVAILABLE
        reason = LEGACY_SUMMARY_NOT_AVAILABLE
        parser = LEGACY_SUMMARY_NOT_AVAILABLE
        evidence: list[Mapping[str, Any]] = []
        omitted_evidence_count = 0
        text_truncated = False
    else:
        summary = project_module_summary(value)
        method = summary["method"]
        method_label = _MODULE_SUMMARY_METHOD_LABELS[method]
        method_id = method
        summary_text = summary["text"]
        reason = summary["reason"] or "none"
        parser = summary["parser"]
        evidence = summary["evidence"]
        omitted_evidence_count = summary["omitted_evidence_count"]
        text_truncated = summary["text_truncated"]

    lines = [
        '<section class="module-summary">',
        "  <h4>Module summary</h4>",
        '  <dl class="module-summary-fields">',
        f"    <dt>Path</dt><dd>{_escape_summary_text_node(path, markdown_safe=True)}</dd>",
        "    <dt>Method</dt>",
        "    <dd>",
        f"      <span class=\"module-summary-method-label\">{_escape_summary_text_node(method_label)}</span>",
    ]
    if method_id is not None:
        lines.append(
            f"      <span class=\"module-summary-method-id\">({_escape_summary_text_node(method_id)})</span>"
        )
    lines.extend(
        [
            "    </dd>",
            f"    <dt>Summary</dt><dd>{_escape_summary_text_node(summary_text, markdown_safe=True)}</dd>",
            f"    <dt>Reason</dt><dd>{_escape_summary_text_node(reason)}</dd>",
            f"    <dt>Parser</dt><dd>{_escape_summary_text_node(parser)}</dd>",
            f"    <dt>Omitted evidence</dt><dd>{_escape_summary_text_node(omitted_evidence_count)}</dd>",
            f"    <dt>Text truncated</dt><dd>{_escape_summary_text_node(str(text_truncated).lower())}</dd>",
            "  </dl>",
            "  <h5>Evidence</h5>",
            '  <ul class="module-summary-evidence">',
        ]
    )

    if legacy_input:
        lines.append(
            f'    <li class="module-summary-no-evidence">{_escape_summary_text_node(LEGACY_SUMMARY_NOT_AVAILABLE)}</li>'
        )
    elif not evidence:
        lines.append('    <li class="module-summary-no-evidence">none</li>')
    else:
        for item in evidence:
            line = "none" if item["line"] is None else item["line"]
            lines.extend(
                [
                    "    <li>",
                    f'      <span class="module-summary-evidence-kind">kind: {_escape_summary_text_node(item["kind"])}</span>',
                    f'      <span class="module-summary-evidence-origin">origin: {_escape_summary_text_node(item["origin"])}</span>',
                    f'      <span class="module-summary-evidence-line">line: {_escape_summary_text_node(line)}</span>',
                    f'      <span class="module-summary-evidence-value">value: {_escape_summary_text_node(item["value"], markdown_safe=True)}</span>',
                    f'      <span class="module-summary-evidence-truncated">value truncated: {_escape_summary_text_node(str(item["value_truncated"]).lower())}</span>',
                    f'      <span class="module-summary-evidence-paragraphs">paragraphs omitted: {_escape_summary_text_node(str(item["paragraphs_omitted"]).lower())}</span>',
                    "    </li>",
                ]
            )

    lines.extend(["  </ul>", "</section>", ""])
    return "\n".join(lines)


__all__ = [
    "ENTRYPOINT_EVIDENCE_VALUE",
    "MODULE_SUMMARY_ELLIPSIS",
    "MODULE_SUMMARY_EVIDENCE_FIELDS",
    "MODULE_SUMMARY_FIELDS",
    "MODULE_SUMMARY_MAX_BYTES",
    "MODULE_SUMMARY_MAX_DEFINITIONS",
    "MODULE_SUMMARY_MAX_DEFINITION_NAME_LENGTH",
    "MODULE_SUMMARY_MAX_EVIDENCE",
    "MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH",
    "MODULE_SUMMARY_MAX_INTEGER",
    "MODULE_SUMMARY_MAX_TEXT_LENGTH",
    "SUMMARY_EVIDENCE_KINDS",
    "SUMMARY_EVIDENCE_MAX_COUNT",
    "SUMMARY_EVIDENCE_ORIGINS",
    "SUMMARY_DEFINITION_MAX_COUNT",
    "SUMMARY_DEFINITION_NAME_MAX_LENGTH",
    "SUMMARY_EVIDENCE_VALUE_MAX_LENGTH",
    "SUMMARY_FAILURE_OUTCOMES",
    "SUMMARY_MAX_BYTES",
    "SUMMARY_MAX_INTEGER",
    "SUMMARY_METHODS",
    "SUMMARY_OUTCOMES",
    "SUMMARY_PARSERS",
    "SUMMARY_TEXT_MAX_LENGTH",
    "SUMMARY_UNKNOWN_REASONS",
    "ModuleSummary",
    "ModuleSummaryContractError",
    "ModuleSummaryEvidence",
    "SummaryDefinition",
    "SummaryDocstring",
    "SummaryEvidence",
    "SummaryEvidenceKind",
    "SummaryEvidenceOrigin",
    "SummaryFacts",
    "SummaryMethod",
    "SummaryOutcome",
    "SummaryParser",
    "SummaryUnknownReason",
    "build_module_summary",
    "canonical_module_summary_bytes",
    "normalize_module_summary",
    "normalize_summary_text",
    "project_module_summary",
    "render_module_summary",
    "serialize_module_summary",
    "truncate_summary_text",
    "validate_module_summary",
]
