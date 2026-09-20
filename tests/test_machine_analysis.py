import ast
from html.parser import HTMLParser
import unittest
import tempfile
import shutil
import json
import hashlib
import os
import re
import sys
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# 機械解析（Level 0）モジュールの各機能を検証します
import isohyps.machine_analysis as machine_analysis
from isohyps.machine_analysis import (
    _extract_python_symbols_and_imports,
    analyze_machine_level,
    build_attention_snapshots,
    build_doc_status_snapshot,
    extract_file_metadata,
    extract_file_symbols,
    build_repo_map_summary,
    detect_attention_points,
    resolve_attention_entrypoints,
)
from isohyps.machine_index import (
    MACHINE_INDEX_FILE_FIELDS,
    MACHINE_INDEX_SCHEMA_VERSION,
    MACHINE_INDEX_TOP_LEVEL_FIELDS,
    MACHINE_INDEX_V2_LEGACY_SCHEMA_VERSION,
    MACHINE_INDEX_V2_SCHEMA_VERSION,
    MachineIndexContractError,
    build_machine_index_v1,
    build_machine_index_v2,
    load_machine_index,
    resolve_machine_index_freshness,
    serialize_machine_index,
    validate_machine_index,
)
from isohyps.module_summary import (
    ENTRYPOINT_EVIDENCE_VALUE,
    MODULE_SUMMARY_EVIDENCE_FIELDS,
    MODULE_SUMMARY_FIELDS,
    MODULE_SUMMARY_MAX_BYTES,
    MODULE_SUMMARY_MAX_DEFINITIONS,
    MODULE_SUMMARY_MAX_EVIDENCE,
    MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH,
    MODULE_SUMMARY_MAX_INTEGER,
    MODULE_SUMMARY_MAX_TEXT_LENGTH,
    ModuleSummaryContractError,
    SummaryFacts,
    build_module_summary,
    canonical_module_summary_bytes,
    normalize_summary_text,
    project_module_summary,
    render_module_summary,
    serialize_module_summary,
    validate_module_summary,
)
from isohyps.attention import (
    AttentionContractError,
    AttentionSignalSnapshot,
    ATTENTION_ENTRY_FIELDS,
    ATTENTION_KINDS,
    canonical_attention_bytes,
    classify_attention,
    serialize_attention,
    validate_attention,
)
from isohyps.doc_freshness import (
    DOC_FRESHNESS_MAPPING_RULE,
    DOC_FRESHNESS_SCHEMA_VERSION,
    expected_doc_paths,
    load_doc_freshness,
    write_doc_provenance_atomic,
    write_doc_freshness_atomic,
)


class TestModuleSummaryContract(unittest.TestCase):
    @staticmethod
    def _docstring_summary() -> dict[str, object]:
        evidence = {
            "kind": "module_docstring",
            "origin": "python_ast",
            "line": 1,
            "value": "Read contour records.",
            "value_truncated": False,
            "paragraphs_omitted": False,
        }
        return {
            "text": evidence["value"],
            "method": "module_docstring",
            "reason": None,
            "parser": "python_ast",
            "evidence": [evidence],
            "omitted_evidence_count": 0,
            "text_truncated": False,
        }

    @staticmethod
    def _structural_summary() -> dict[str, object]:
        return {
            "text": "Non-underscore top-level definitions: Reader.",
            "method": "structural_facts",
            "reason": None,
            "parser": "python_ast",
            "evidence": [
                {
                    "kind": "definition",
                    "origin": "python_ast",
                    "line": None,
                    "value": "Reader",
                    "value_truncated": False,
                    "paragraphs_omitted": False,
                }
            ],
            "omitted_evidence_count": 0,
            "text_truncated": False,
        }

    def test_summary_contract_shape_and_projection_are_bounded_and_non_mutating(self):
        summary = self._docstring_summary()
        summary["future_summary_field"] = {"kept_by": "reader only"}
        summary["evidence"][0]["future_evidence_field"] = ["ignored"]
        original = deepcopy(summary)

        projected = project_module_summary(summary)

        self.assertEqual(summary, original)
        self.assertEqual(set(projected), set(MODULE_SUMMARY_FIELDS))
        self.assertEqual(
            set(projected["evidence"][0]), set(MODULE_SUMMARY_EVIDENCE_FIELDS)
        )
        self.assertNotIn("future_summary_field", projected)
        self.assertNotIn("future_evidence_field", projected["evidence"][0])
        validate_module_summary(projected)

        canonical = canonical_module_summary_bytes(summary)
        serialized = serialize_module_summary(summary)
        self.assertEqual(serialized.encode("utf-8"), canonical)
        self.assertTrue(serialized.endswith("\n"))
        self.assertLessEqual(len(canonical), MODULE_SUMMARY_MAX_BYTES)
        self.assertEqual(json.loads(serialized), projected)

    def test_normalization_folds_whitespace_and_replaces_remaining_controls(self):
        raw = "\t  Read\u00a0contours.\n\u200b\x00\ud800  "
        self.assertEqual(normalize_summary_text(raw), "Read contours. ���")

        summary = self._docstring_summary()
        summary["text"] = "\tRead\u00a0contour records.\n"
        summary["evidence"][0]["value"] = "Read contour records."
        projected = project_module_summary(summary)
        self.assertEqual(projected["text"], "Read contour records.")
        self.assertEqual(
            projected["evidence"][0]["value"], "Read contour records."
        )
        validate_module_summary(projected)

    def test_summary_facts_is_internal_fact_shape(self):
        facts = SummaryFacts(
            parser="python_ast",
            outcome="ok",
            docstring={"text": "Read contours.", "line": 1},
            definitions=[],
        )
        self.assertEqual(facts.parser, "python_ast")
        self.assertEqual(facts.outcome, "ok")
        self.assertEqual(facts.docstring["line"], 1)
        self.assertEqual(facts.definitions, [])

    def test_valid_structural_and_unknown_shapes(self):
        structural = self._structural_summary()
        structural["evidence"][0]["line"] = MODULE_SUMMARY_MAX_INTEGER
        structural["omitted_evidence_count"] = MODULE_SUMMARY_MAX_INTEGER
        validate_module_summary(structural)

        unknown = {
            "text": "unknown",
            "method": "unknown",
            "reason": "insufficient_evidence",
            "parser": "python_ast",
            "evidence": [],
            "omitted_evidence_count": 0,
            "text_truncated": False,
        }
        validate_module_summary(unknown)

    def test_invalid_summary_shapes_are_rejected(self):
        cases: list[tuple[str, dict[str, object]]] = []

        missing_field = self._docstring_summary()
        del missing_field["text"]
        cases.append(("missing field", missing_field))

        wrong_text_type = self._docstring_summary()
        wrong_text_type["text"] = {"not": "text"}
        cases.append(("text type", wrong_text_type))

        too_long_text = self._docstring_summary()
        too_long_text["text"] = "x" * (MODULE_SUMMARY_MAX_TEXT_LENGTH + 1)
        cases.append(("text length", too_long_text))

        invalid_method = self._docstring_summary()
        invalid_method["method"] = "llm"
        cases.append(("method enum", invalid_method))

        invalid_parser = self._docstring_summary()
        invalid_parser["parser"] = "guess"
        cases.append(("parser enum", invalid_parser))

        too_many_evidence = self._structural_summary()
        too_many_evidence["evidence"] = [
            deepcopy(too_many_evidence["evidence"][0])
            for _ in range(MODULE_SUMMARY_MAX_EVIDENCE + 1)
        ]
        cases.append(("evidence count", too_many_evidence))

        too_long_value = self._structural_summary()
        too_long_value["evidence"][0]["value"] = "x" * (
            MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH + 1
        )
        cases.append(("evidence value length", too_long_value))

        invalid_line = self._structural_summary()
        invalid_line["evidence"][0]["line"] = 0
        cases.append(("line lower bound", invalid_line))

        boolean_line = self._structural_summary()
        boolean_line["evidence"][0]["line"] = True
        cases.append(("boolean line", boolean_line))

        boolean_count = self._structural_summary()
        boolean_count["omitted_evidence_count"] = True
        cases.append(("boolean count", boolean_count))

        count_overflow = self._structural_summary()
        count_overflow["omitted_evidence_count"] = MODULE_SUMMARY_MAX_INTEGER + 1
        cases.append(("count upper bound", count_overflow))

        wrong_docstring_origin = self._docstring_summary()
        wrong_docstring_origin["evidence"][0]["origin"] = "regex"
        cases.append(("docstring origin", wrong_docstring_origin))

        wrong_docstring_line = self._docstring_summary()
        wrong_docstring_line["evidence"][0]["line"] = None
        cases.append(("docstring line", wrong_docstring_line))

        mismatched_docstring_text = self._docstring_summary()
        mismatched_docstring_text["text"] = "Different text."
        cases.append(("docstring text evidence", mismatched_docstring_text))

        structural_without_evidence = self._structural_summary()
        structural_without_evidence["evidence"] = []
        cases.append(("structural evidence", structural_without_evidence))

        structural_with_reason = self._structural_summary()
        structural_with_reason["reason"] = "insufficient_evidence"
        cases.append(("known reason", structural_with_reason))

        unknown_with_success_reason = {
            "text": "unknown",
            "method": "unknown",
            "reason": "ok",
            "parser": "none",
            "evidence": [],
            "omitted_evidence_count": 0,
            "text_truncated": False,
        }
        cases.append(("unknown success reason", unknown_with_success_reason))

        unknown_with_evidence = {
            "text": "unknown",
            "method": "unknown",
            "reason": "parse_error",
            "parser": "python_ast",
            "evidence": [
                {
                    "kind": "definition",
                    "origin": "python_ast",
                    "line": 1,
                    "value": "ignored",
                    "value_truncated": False,
                    "paragraphs_omitted": False,
                }
            ],
            "omitted_evidence_count": 0,
            "text_truncated": False,
        }
        cases.append(("unknown evidence", unknown_with_evidence))

        for label, value in cases:
            with self.subTest(case=label):
                with self.assertRaises(ModuleSummaryContractError):
                    validate_module_summary(value)

    def test_structural_evidence_requires_parser_origin_and_entrypoint_shape(self):
        wrong_definition_origin = self._structural_summary()
        wrong_definition_origin["parser"] = "regex"
        with self.assertRaises(ModuleSummaryContractError):
            validate_module_summary(wrong_definition_origin)

        entrypoint = {
            "text": ENTRYPOINT_EVIDENCE_VALUE,
            "method": "structural_facts",
            "reason": None,
            "parser": "none",
            "evidence": [
                {
                    "kind": "entrypoint_candidate",
                    "origin": "attention_entrypoint_resolver_v1",
                    "line": None,
                    "value": ENTRYPOINT_EVIDENCE_VALUE,
                    "value_truncated": False,
                    "paragraphs_omitted": False,
                }
            ],
            "omitted_evidence_count": 0,
            "text_truncated": False,
        }
        validate_module_summary(entrypoint)

        invalid_entrypoint = deepcopy(entrypoint)
        invalid_entrypoint["evidence"][0]["line"] = 1
        with self.assertRaises(ModuleSummaryContractError):
            validate_module_summary(invalid_entrypoint)


class TestModuleSummaryGeneration(unittest.TestCase):
    @staticmethod
    def _definition(name: str, kind: str = "function", line: int | None = 1):
        return {"name": name, "kind": kind, "line": line}

    def test_module_docstring_has_priority_and_records_omitted_paragraphs(self):
        summary = build_module_summary(
            SummaryFacts(
                parser="python_ast",
                docstring={
                    "text": (
                        "  Process payments.  \n\n"
                        "Deprecated; this module only draws contours."
                    ),
                    "line": 4,
                },
                definitions=[self._definition("draw_contours", line=8)],
            ),
            entrypoint_candidate=True,
        )

        self.assertEqual(summary["method"], "module_docstring")
        self.assertEqual(summary["text"], "Process payments.")
        self.assertFalse(summary["text_truncated"])
        self.assertEqual(summary["omitted_evidence_count"], 0)
        self.assertEqual(len(summary["evidence"]), 1)
        self.assertEqual(summary["evidence"][0]["kind"], "module_docstring")
        self.assertEqual(summary["evidence"][0]["line"], 4)
        self.assertTrue(summary["evidence"][0]["paragraphs_omitted"])
        self.assertFalse(summary["evidence"][0]["value_truncated"])
        validate_module_summary(summary)


    def test_blank_docstring_falls_back_to_limited_structural_facts(self):
        summary = build_module_summary(
            SummaryFacts(
                parser="python_ast",
                docstring={"text": " \n\t ", "line": 1},
                definitions=[self._definition("run", line=3)],
            )
        )

        self.assertEqual(
            summary["text"], "Non-underscore top-level definitions: run."
        )
        self.assertEqual(summary["method"], "structural_facts")
        self.assertEqual(summary["evidence"][0]["kind"], "definition")
        self.assertEqual(summary["evidence"][0]["origin"], "python_ast")

    def test_structural_facts_are_sorted_deduplicated_and_deterministic(self):
        definitions = [
            self._definition("zebra", "function", 7),
            self._definition("run", "function", 4),
            self._definition("_private", "function", 2),
            self._definition("Alpha", "class", 3),
            self._definition("run", "function", 4),
            self._definition("run", "function", 8),
        ]
        first = build_module_summary(
            SummaryFacts(parser="python_ast", definitions=definitions)
        )
        second = build_module_summary(
            SummaryFacts(parser="python_ast", definitions=list(reversed(definitions)))
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first["text"],
            "Non-underscore top-level definitions: Alpha, run, run, zebra.",
        )
        self.assertEqual(
            [(item["value"], item["line"]) for item in first["evidence"]],
            [("Alpha", 3), ("run", 4), ("run", 8), ("zebra", 7)],
        )
        self.assertLessEqual(len(canonical_module_summary_bytes(first)), MODULE_SUMMARY_MAX_BYTES)

    def test_structural_summary_is_stable_across_hash_seeds_in_subprocesses(self):
        script = """
from isohyps.module_summary import (
    SummaryFacts,
    build_module_summary,
    canonical_module_summary_bytes,
)

line_by_name = {"zebra": 7, "Alpha": 3, "run": 4, "load": 9}
definitions = [
    {"name": name, "kind": "function", "line": line_by_name[name]}
    for name in set(line_by_name)
]
summary = build_module_summary(
    SummaryFacts(parser="python_ast", definitions=definitions)
)
print(canonical_module_summary_bytes(summary).decode("utf-8"), end="")
"""
        outputs = []
        for seed in ("1", "42"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    text=True,
                )
            )

        self.assertEqual(outputs[0], outputs[1])
        summary = json.loads(outputs[0])
        self.assertEqual(
            [item["value"] for item in summary["evidence"]],
            ["Alpha", "load", "run", "zebra"],
        )

    def test_entrypoint_is_limited_observation_and_unsupported_only_allows_it(self):
        with_definitions = build_module_summary(
            SummaryFacts(
                parser="python_ast",
                definitions=[self._definition("main", line=9)],
            ),
            entrypoint_candidate=True,
        )
        self.assertEqual(
            with_definitions["text"],
            "Entrypoint candidate detected. Non-underscore top-level definitions: main.",
        )
        self.assertEqual(
            [item["kind"] for item in with_definitions["evidence"]],
            ["entrypoint_candidate", "definition"],
        )
        self.assertEqual(
            with_definitions["evidence"][0]["origin"],
            "attention_entrypoint_resolver_v1",
        )

        unsupported = SummaryFacts(
            parser="none",
            outcome="unsupported",
            definitions=[self._definition("main")],
        )
        self.assertEqual(
            build_module_summary(unsupported)["reason"], "unsupported"
        )
        entrypoint_only = build_module_summary(
            unsupported, entrypoint_candidate=True
        )
        self.assertEqual(entrypoint_only["method"], "structural_facts")
        self.assertEqual(entrypoint_only["parser"], "none")
        self.assertEqual(entrypoint_only["text"], ENTRYPOINT_EVIDENCE_VALUE)
        validate_module_summary(entrypoint_only)

    def test_failures_and_import_only_facts_never_get_a_structural_summary(self):
        for outcome in (
            "read_error",
            "decode_error",
            "parse_error",
            "binary_skipped",
            "source_changed",
        ):
            with self.subTest(outcome=outcome):
                summary = build_module_summary(
                    SummaryFacts(
                        parser="python_ast",
                        outcome=outcome,
                        docstring={"text": "Observed text.", "line": 1},
                        definitions=[self._definition("main")],
                    ),
                    entrypoint_candidate=True,
                )
                self.assertEqual(summary["method"], "unknown")
                self.assertEqual(summary["reason"], outcome)
                self.assertEqual(summary["text"], "unknown")
                self.assertEqual(summary["evidence"], [])

        import_only = build_module_summary(SummaryFacts(parser="python_ast"))
        self.assertEqual(import_only["text"], "unknown")
        self.assertEqual(import_only["reason"], "insufficient_evidence")

    def test_definition_and_summary_text_truncation_are_independent_and_bounded(self):
        long_name = "x" * (MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH + 40)
        summary = build_module_summary(
            SummaryFacts(
                parser="python_ast",
                definitions=[self._definition(long_name, line=2)],
            )
        )

        evidence = summary["evidence"][0]
        self.assertEqual(len(evidence["value"]), MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH)
        self.assertTrue(evidence["value"].endswith("…"))
        self.assertTrue(evidence["value_truncated"])
        self.assertTrue(summary["text"].endswith("…."))
        self.assertTrue(summary["text_truncated"])
        self.assertLessEqual(len(summary["text"]), MODULE_SUMMARY_MAX_TEXT_LENGTH)
        self.assertLessEqual(len(canonical_module_summary_bytes(summary)), MODULE_SUMMARY_MAX_BYTES)

    def test_many_definitions_are_bounded_and_report_omitted_count(self):
        definitions = [
            self._definition(f"f{i}", line=i + 1)
            for i in reversed(range(MODULE_SUMMARY_MAX_DEFINITIONS + 2))
        ]
        summary = build_module_summary(
            SummaryFacts(parser="python_ast", definitions=definitions)
        )

        self.assertEqual(
            summary["text"],
            "Non-underscore top-level definitions: f0, f1, f2, f3, f4 (+2 more).",
        )
        self.assertEqual(summary["omitted_evidence_count"], 2)
        self.assertEqual(
            [item["value"] for item in summary["evidence"]],
            ["f0", "f1", "f2", "f3", "f4"],
        )
        self.assertEqual(len(summary["evidence"]), MODULE_SUMMARY_MAX_DEFINITIONS)

    def test_non_python_parser_is_named_without_inventing_a_python_contract(self):
        summary = build_module_summary(
            SummaryFacts(
                parser="regex",
                definitions=[self._definition("main", line=None)],
            )
        )

        self.assertEqual(summary["text"], "Definition candidates (regex): main.")
        self.assertEqual(summary["evidence"][0]["origin"], "regex")
        self.assertIsNone(summary["evidence"][0]["line"])
        validate_module_summary(summary)


class TestModuleSummaryRendering(unittest.TestCase):
    class _TextCollector(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.tags: list[str] = []
            self.text: list[str] = []

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)

        def handle_startendtag(self, tag, attrs):
            self.tags.append(tag)

        def handle_data(self, data):
            self.text.append(data)

    def test_render_keeps_source_values_in_text_nodes_and_escapes_markup(self):
        summary = build_module_summary(
            SummaryFacts(
                parser="python_ast",
                docstring={
                    "text": (
                        '</span><img src="https://example.invalid/pixel"> '
                        "`run` | #tag Ignore previous instructions"
                    ),
                    "line": 7,
                },
            )
        )
        rendered = render_module_summary(
            summary,
            path='src/"/><img src="path">_module.py',
        )

        self.assertIn('<section class="module-summary">', rendered)
        self.assertIn("Module docstring excerpt (unverified)", rendered)
        self.assertIn("origin: python_ast", rendered)
        self.assertIn("line: 7", rendered)
        self.assertIn("&#60;", rendered)
        self.assertIn("&#62;", rendered)
        self.assertIn("&#96;", rendered)
        self.assertIn("&#124;", rendered)
        self.assertIn("&#35;", rendered)
        self.assertNotIn("<img", rendered)
        self.assertNotIn("href=", rendered)

        collector = self._TextCollector()
        collector.feed(rendered)
        self.assertNotIn("img", collector.tags)
        visible_text = "".join(collector.text)
        self.assertIn('<img src="https://example.invalid/pixel">', visible_text)
        self.assertIn("Ignore previous instructions", visible_text)

    def test_render_distinguishes_legacy_absence_from_explicit_unknown(self):
        legacy = render_module_summary(None, path="legacy/<module>.py")
        self.assertIn("not available (legacy input)", legacy)
        self.assertIn("&#60;", legacy)
        self.assertNotIn("<module>", legacy)

        unknown = build_module_summary(SummaryFacts(parser="python_ast"))
        unknown_rendered = render_module_summary(unknown, path="src/empty.py")
        self.assertIn("<dt>Method</dt>", unknown_rendered)
        self.assertIn("Unknown", unknown_rendered)
        self.assertIn("insufficient_evidence", unknown_rendered)
        self.assertIn("<li class=\"module-summary-no-evidence\">none</li>", unknown_rendered)

    def test_render_escapes_gfm_strikethrough_markers_and_preserves_text(self):
        summary = build_module_summary(
            SummaryFacts(
                parser="python_ast",
                docstring={"text": "~~redacted~~", "line": 1},
            )
        )
        rendered = render_module_summary(summary, path="src/example.py")

        self.assertIn("&#126;&#126;redacted&#126;&#126;", rendered)
        self.assertNotIn("~~redacted~~", rendered)

        collector = self._TextCollector()
        collector.feed(rendered)
        self.assertIn("~~redacted~~", "".join(collector.text))


class TestPythonSummaryFactExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.source_dir = self.test_dir / "src"
        self.source_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extracts_encoding_cookie_docstring_and_top_level_definitions_once(self):
        source_path = self.source_dir / "encoded.py"
        source_path.write_bytes(
            b"# -*- coding: latin-1 -*-\n"
            b'"""caf\xe9 module."""\n'
            b"import os\n"
            b"class Public:\n"
            b"    def method(self):\n"
            b"        pass\n"
            b"async def fetch():\n"
            b"    pass\n"
            b"def outer():\n"
            b"    class Nested:\n"
            b"        pass\n"
            b"    def nested():\n"
            b"        pass\n"
        )

        with patch(
            "isohyps.machine_analysis.ast.parse",
            wraps=ast.parse,
        ) as parse:
            result = extract_file_symbols(source_path, self.test_dir)

        self.assertEqual(parse.call_count, 1)
        self.assertEqual(
            [symbol["name"] for symbol in result["symbols"]],
            ["Public", "Public.method", "fetch", "outer"],
        )
        self.assertEqual([item["module"] for item in result["imports"]], ["os"])
        self.assertEqual(result["exports"], ["Public", "fetch", "outer"])

        facts = result["summary_facts"]
        self.assertIsInstance(facts, SummaryFacts)
        self.assertEqual(facts.parser, "python_ast")
        self.assertEqual(facts.outcome, "ok")
        self.assertEqual(facts.docstring, {"text": "café module.", "line": 2})
        self.assertEqual(
            [
                (item["name"], item["kind"], item["line"])
                for item in facts.definitions
            ],
            [
                ("Public", "class", 4),
                ("fetch", "function", 7),
                ("outer", "function", 9),
            ],
        )
        self.assertNotIn("Nested", [item["name"] for item in facts.definitions])
        self.assertNotIn("nested", [item["name"] for item in facts.definitions])
        self.assertFalse(hasattr(facts, "source"))
        self.assertFalse(hasattr(facts, "tree"))

    def test_existing_python_tuple_wrapper_uses_the_shared_extractor(self):
        symbols, imports, exports = _extract_python_symbols_and_imports(
            '"""module"""\n'
            "class Reader:\n"
            "    pass\n"
            "async def load():\n"
            "    pass\n"
        )

        self.assertEqual([item["name"] for item in symbols], ["Reader", "load"])
        self.assertEqual(imports, [])
        self.assertEqual(exports, ["Reader", "load"])

    def test_python_extraction_distinguishes_read_decode_parse_and_empty_outcomes(self):
        cases = {
            "empty.py": (b"", "ok"),
            "bad_syntax.py": (b"def broken(\n", "parse_error"),
            "bad_decode.py": (b'"""\xff"""\n', "decode_error"),
            "bad_cookie.py": (b"# coding: no_such_codec\n", "decode_error"),
        }
        for name, (source, outcome) in cases.items():
            with self.subTest(file=name):
                path = self.source_dir / name
                path.write_bytes(source)
                result = extract_file_symbols(path, self.test_dir)
                facts = result["summary_facts"]
                self.assertIsInstance(facts, SummaryFacts)
                self.assertEqual(facts.outcome, outcome)
                self.assertEqual(result["symbols"], [])
                self.assertEqual(result["imports"], [])
                self.assertEqual(result["exports"], [])

        missing = extract_file_symbols(
            self.source_dir / "missing.py", self.test_dir
        )
        self.assertEqual(missing["summary_facts"].outcome, "read_error")
        self.assertEqual(missing["summary_facts"].docstring, None)


class TestNonPythonAndMachineSummaryIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.source_dir = self.test_dir / "src"
        self.source_dir.mkdir()
        self.output_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.output_dir)

    def test_non_python_tree_sitter_facts_use_the_read_bytes_snapshot(self):
        source_path = self.source_dir / "widget.js"
        source = b"// header\nfunction render() {}\n"
        source_path.write_bytes(source)
        node = SimpleNamespace(
            start_byte=10,
            start_point=(1, 0),
            type="function_declaration",
        )
        tree = SimpleNamespace(root_node=object())
        seen_sources: list[bytes] = []

        class FakeParser:
            def parse(self, payload):
                seen_sources.append(payload)
                return tree

        class FakeQuery:
            def captures(self, _root):
                return [(node, "symbol")]

        class FakeLanguage:
            def query(self, _query):
                return FakeQuery()

        fake_module = SimpleNamespace(
            get_parser=lambda _language: FakeParser(),
            get_language=lambda _language: FakeLanguage(),
        )
        with patch.dict(sys.modules, {"tree_sitter_languages": fake_module}):
            result = extract_file_symbols(source_path, self.test_dir)

        facts = result["summary_facts"]
        self.assertEqual(seen_sources, [source])
        self.assertEqual(facts.parser, "tree_sitter")
        self.assertEqual(facts.outcome, "ok")
        self.assertEqual(
            [(item["name"], item["kind"], item["line"]) for item in facts.definitions],
            [("render", "function", 2)],
        )
        self.assertEqual(
            result["summary_source_hash"], hashlib.sha256(source).hexdigest()
        )

    def test_non_python_tree_sitter_error_tree_returns_parse_error(self):
        source_path = self.source_dir / "broken.js"
        source_path.write_bytes(b"function broken( {\n")
        tree = SimpleNamespace(root_node=SimpleNamespace(has_error=True))

        class FakeParser:
            def parse(self, payload):
                if not isinstance(payload, bytes):
                    raise AssertionError("parser payload must be bytes")
                return tree

        class FakeLanguage:
            def query(self, _query):
                raise AssertionError("query should not run for parse-error trees")

        fake_module = SimpleNamespace(
            get_parser=lambda _language: FakeParser(),
            get_language=lambda _language: FakeLanguage(),
        )
        with patch.dict(sys.modules, {"tree_sitter_languages": fake_module}):
            result = extract_file_symbols(source_path, self.test_dir)

        facts = result["summary_facts"]
        self.assertEqual(facts.parser, "tree_sitter")
        self.assertEqual(facts.outcome, "parse_error")
        self.assertEqual(list(facts.definitions), [])
        self.assertEqual(result["symbols"], [])

    def test_non_python_regex_unknown_binary_and_decode_outcomes_are_distinct(self):
        regex_path = self.source_dir / "fallback.js"
        regex_path.write_text("function render() {}\n", encoding="utf-8")
        with patch.dict(sys.modules, {"tree_sitter_languages": None}):
            regex_result = extract_file_symbols(regex_path, self.test_dir)
        self.assertEqual(regex_result["summary_facts"].parser, "regex")
        self.assertEqual(regex_result["summary_facts"].outcome, "ok")
        self.assertEqual(regex_result["summary_facts"].definitions[0]["line"], 1)

        unknown_path = self.source_dir / "opaque.xyz"
        unknown_path.write_text(
            "function render() {}\nimport billing\n", encoding="utf-8"
        )
        unknown_result = extract_file_symbols(unknown_path, self.test_dir)
        self.assertEqual(unknown_result["summary_facts"].parser, "none")
        self.assertEqual(unknown_result["summary_facts"].outcome, "unsupported")
        self.assertIn(
            {"module": "billing", "internal": False},
            unknown_result["imports"],
        )

        binary_path = self.source_dir / "asset.png"
        binary_path.write_bytes(b"\x89PNG\r\n")
        binary_result = extract_file_symbols(binary_path, self.test_dir)
        self.assertEqual(binary_result["summary_facts"].parser, "none")
        self.assertEqual(binary_result["summary_facts"].outcome, "binary_skipped")

        invalid_path = self.source_dir / "invalid.js"
        invalid_path.write_bytes(b"function render() {}\n\xff")
        invalid_result = extract_file_symbols(invalid_path, self.test_dir)
        self.assertEqual(invalid_result["summary_facts"].parser, "none")
        self.assertEqual(invalid_result["summary_facts"].outcome, "decode_error")

    def test_missing_metadata_is_fail_closed_without_raising(self):
        missing = self.source_dir / "gone.py"
        metadata = extract_file_metadata(missing, self.test_dir)
        self.assertEqual(metadata["hash"], "error")
        self.assertEqual(metadata["mtime"], 0)
        self.assertEqual(metadata["size"], 0)
        self.assertFalse(metadata["readable"])

    def test_scan_adds_summary_to_all_source_entries_and_hides_internal_facts(self):
        (self.source_dir / "good.py").write_text(
            '"""Read contour records."""\n\n'
            "def read_records():\n    pass\n",
            encoding="utf-8",
        )
        (self.source_dir / "app.py").write_text(
            "def main():\n    pass\n", encoding="utf-8"
        )
        (self.source_dir / "widget.js").write_text(
            "function render() {}\n", encoding="utf-8"
        )
        (self.source_dir / "opaque.xyz").write_text(
            "import billing\n", encoding="utf-8"
        )
        (self.source_dir / "__init__.py").write_text("", encoding="utf-8")
        (self.test_dir / "tests").mkdir()
        (self.test_dir / "tests" / "test_widget.py").write_text(
            "def test_widget():\n    pass\n", encoding="utf-8"
        )
        (self.test_dir / "settings.toml").write_text("enabled = true\n", encoding="utf-8")

        with patch.dict(sys.modules, {"tree_sitter_languages": None}):
            result = analyze_machine_level(self.test_dir, self.output_dir)

        entries = {entry["path"]: entry for entry in result["files"]}
        source_entries = {
            path: entry for path, entry in entries.items() if entry["kind"] == "source"
        }
        self.assertEqual(
            set(source_entries),
            {"src/app.py", "src/good.py", "src/opaque.xyz", "src/widget.js"},
        )
        for entry in source_entries.values():
            self.assertIn("module_summary", entry)

        good_summary = source_entries["src/good.py"]["module_summary"]
        self.assertEqual(good_summary["method"], "module_docstring")
        self.assertEqual(good_summary["text"], "Read contour records.")
        self.assertEqual(good_summary["evidence"][0]["line"], 1)

        app_summary = source_entries["src/app.py"]["module_summary"]
        self.assertEqual(app_summary["method"], "structural_facts")
        self.assertEqual(
            [item["kind"] for item in app_summary["evidence"]],
            ["entrypoint_candidate", "definition"],
        )
        self.assertEqual(
            app_summary["evidence"][0]["origin"],
            "attention_entrypoint_resolver_v1",
        )

        unknown_summary = source_entries["src/opaque.xyz"]["module_summary"]
        self.assertEqual(unknown_summary["text"], "unknown")
        self.assertEqual(unknown_summary["reason"], "unsupported")
        self.assertEqual(unknown_summary["evidence"], [])

        self.assertNotIn("module_summary", entries["src/__init__.py"])
        self.assertNotIn("module_summary", entries["tests/test_widget.py"])
        self.assertNotIn("module_summary", entries["settings.toml"])
        self.assertTrue(all("summary_facts" not in item for item in result["symbols"]))
        self.assertTrue(all("summary_source_hash" not in item for item in result["symbols"]))
        self.assertNotIn("SummaryFacts", (self.output_dir / "machine_analysis.json").read_text())
        self.assertNotIn("summary_facts", (self.output_dir / "machine_analysis.yaml").read_text())

    def test_markdown_indexes_render_all_public_source_summaries_in_path_order(self):
        (self.source_dir / "zeta.py").write_text(
            '"""Zeta module.\n\nAdditional context is omitted from the excerpt."""\n\n'
            "def run():\n    pass\n",
            encoding="utf-8",
        )
        (self.source_dir / "alpha.py").write_text(
            "def load_records():\n    pass\n", encoding="utf-8"
        )
        # Unknown-language files remain source entries and must not disappear
        # from the human-readable summary inventory.
        (self.source_dir / "opaque.xyz").write_text(
            "import billing\n", encoding="utf-8"
        )

        with patch.dict(sys.modules, {"tree_sitter_languages": None}):
            first_result = analyze_machine_level(self.test_dir, self.output_dir)
            first_index_bytes = (self.output_dir / "machine_index.json").read_bytes()
            second_result = analyze_machine_level(self.test_dir, self.output_dir)

        public_index = json.loads(
            (self.output_dir / "machine_index.json").read_text(encoding="utf-8")
        )
        public_sources = sorted(
            (
                entry
                for entry in public_index["files"]
                if entry["kind"] == "source"
            ),
            key=lambda entry: entry["path"],
        )
        expected_paths = [
            "src/alpha.py",
            "src/opaque.xyz",
            "src/zeta.py",
        ]
        self.assertEqual(
            [entry["path"] for entry in public_sources], expected_paths
        )

        index_content = (self.output_dir / "index.md").read_text(encoding="utf-8")
        report_content = (
            self.output_dir / "machine_report.md"
        ).read_text(encoding="utf-8")
        for document, content in (
            ("index", index_content),
            ("report", report_content),
        ):
            with self.subTest(document=document):
                summary_section = content.split("## Module Summaries", 1)[1]
                self.assertLess(
                    summary_section.index("src/alpha.py"),
                    summary_section.index("src/opaque.xyz"),
                )
                self.assertLess(
                    summary_section.index("src/opaque.xyz"),
                    summary_section.index("src/zeta.py"),
                )
                self.assertIn(
                    "Deterministic summaries for every discovered source file.",
                    content,
                )
                self.assertIn("Zeta module.", content)
                self.assertIn("paragraphs omitted: true", content)
                self.assertIn(
                    "Non-underscore top-level definitions: load&#95;records.",
                    content,
                )
                self.assertIn("unknown", content)
                self.assertIn("unsupported", content)
                self.assertIn("Omitted evidence</dt><dd>0</dd>", content)

        public_by_path = {entry["path"]: entry for entry in public_sources}
        self.assertEqual(
            public_by_path["src/zeta.py"]["module_summary"]["text"],
            "Zeta module.",
        )
        self.assertEqual(
            public_by_path["src/zeta.py"]["module_summary"]["method"],
            "module_docstring",
        )
        self.assertTrue(
            public_by_path["src/zeta.py"]["module_summary"]["evidence"][0][
                "paragraphs_omitted"
            ]
        )
        self.assertEqual(
            public_by_path["src/opaque.xyz"]["module_summary"]["reason"],
            "unsupported",
        )

        # The second scan marks the files unchanged, but the all-source
        # summary inventory remains present and uses the same public values.
        second_paths = [
            entry["path"]
            for entry in second_result["files"]
            if entry["kind"] == "source"
        ]
        self.assertEqual(sorted(second_paths), expected_paths)
        self.assertTrue(
            all(
                entry["status"] == "unchanged"
                for entry in second_result["files"]
                if entry["path"] in expected_paths
            )
        )
        first_summaries = {
            entry["path"]: entry["module_summary"]
            for entry in first_result["files"]
            if entry["path"] in expected_paths
        }
        second_summaries = {
            entry["path"]: entry["module_summary"]
            for entry in second_result["files"]
            if entry["path"] in expected_paths
        }
        self.assertEqual(first_summaries, second_summaries)
        self.assertEqual(
            first_index_bytes,
            (self.output_dir / "machine_index.json").read_bytes(),
        )
        self.assertIn("## High Priority Files to Inspect", index_content)
        self.assertIn("Refer to [machine_report.md](./machine_report.md)", index_content)

    def test_hash_mismatch_discards_entrypoint_and_definition_evidence(self):
        app = self.source_dir / "app.py"
        app.write_text("def main():\n    pass\n", encoding="utf-8")
        real_extract_metadata = machine_analysis.extract_file_metadata

        def mismatching_metadata(path, root, previous_meta=None):
            metadata = real_extract_metadata(path, root, previous_meta)
            if metadata["path"] == "src/app.py":
                metadata["hash"] = "0" * 64
            return metadata

        with (
            patch.object(
                machine_analysis,
                "extract_file_metadata",
                side_effect=mismatching_metadata,
            ),
            patch.dict(sys.modules, {"tree_sitter_languages": None}),
        ):
            result = analyze_machine_level(self.test_dir, self.output_dir)

        summary = next(
            item["module_summary"]
            for item in result["files"]
            if item["path"] == "src/app.py"
        )
        self.assertEqual(summary["method"], "unknown")
        self.assertEqual(summary["reason"], "source_changed")
        self.assertEqual(summary["evidence"], [])

    def test_scan_survives_file_removed_between_metadata_and_extraction(self):
        transient = self.source_dir / "transient.py"
        transient.write_text("def run():\n    pass\n", encoding="utf-8")
        real_extract_symbols = machine_analysis.extract_file_symbols

        def remove_before_extract(path, root):
            if path == transient:
                path.unlink()
            return real_extract_symbols(path, root)

        with patch.object(
            machine_analysis,
            "extract_file_symbols",
            side_effect=remove_before_extract,
        ):
            result = analyze_machine_level(self.test_dir, self.output_dir)

        summary = next(
            item["module_summary"]
            for item in result["files"]
            if item["path"] == "src/transient.py"
        )
        self.assertEqual(summary["method"], "unknown")
        self.assertEqual(summary["reason"], "read_error")

class TestMachineAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())

        # テスト用のファイル群を作成
        self.src_dir = self.test_dir / "src"
        self.src_dir.mkdir()
        self.test_files_dir = self.test_dir / "tests"
        self.test_files_dir.mkdir()

        # 1. 普通のPythonソースファイル
        self.runner_file = self.src_dir / "runner.py"
        self.runner_file.write_text(
            "import os\n"
            "import sys\n"
            "from src.config import Config\n"
            "\n"
            "class Runner:\n"
            "    def __init__(self):\n"
            "        self.config = Config()\n"
            "\n"
            "    def run(self):\n"
            "        # TODO: Implement run details\n"
            "        print('running')\n"
            "\n"
            "def main():\n"
            "    r = Runner()\n"
            "    r.run()\n",
            encoding="utf-8"
        )

        # 2. 設定ファイル (config.py) - 複数のファイルからインポートされる想定
        self.config_file = self.src_dir / "config.py"
        self.config_file.write_text(
            "class Config:\n"
            "    def __init__(self):\n"
            "        self.debug = True\n",
            encoding="utf-8"
        )

        # 3. テストファイル
        self.runner_test_file = self.test_files_dir / "test_runner.py"
        self.runner_test_file.write_text(
            "import unittest\n"
            "from src.runner import Runner\n"
            "\n"
            "class TestRunner(unittest.TestCase):\n"
            "    def test_run(self):\n"
            "        r = Runner()\n"
            "        self.assertIsNotNone(r)\n",
            encoding="utf-8"
        )

        # 4. 大きなレガシーファイル (300行以上にして large file 警告をトリガーする)
        self.legacy_file = self.src_dir / "legacy.py"
        large_content = "\n".join([f"line_{i} = {i}" for i in range(350)])
        self.legacy_file.write_text(large_content, encoding="utf-8")

        # 4b. 空の __init__.py (no tests 警告から除外されるべきファイル)
        self.init_file = self.src_dir / "__init__.py"
        self.init_file.write_text("", encoding="utf-8")

        # 5. pyproject.toml (config/entrypoint)
        self.toml_file = self.test_dir / "pyproject.toml"
        self.toml_file.write_text(
            "[project]\n"
            "name = 'test-project'\n"
            "[project.scripts]\n"
            "test-cli = 'src.runner:main'\n",
            encoding="utf-8"
        )

        # 6. 境界値・設定ファイル群 (ノイズ削減テスト用)
        self.conftest_file = self.src_dir / "conftest.py"
        self.conftest_file.write_text("# Test fixtures", encoding="utf-8")

        self.dockerfile_file = self.test_dir / "Dockerfile"
        self.dockerfile_file.write_text("FROM python:3.9", encoding="utf-8")

        self.github_dir = self.test_dir / ".github/workflows"
        self.github_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_file = self.github_dir / "deploy.yml"
        self.workflow_file.write_text("name: deploy", encoding="utf-8")

        self.yaml_config_file = self.src_dir / "settings.yaml"
        self.yaml_config_file.write_text("debug: false", encoding="utf-8")

    @staticmethod
    def _valid_machine_index_fixture():
        """Return a small v1 fixture with a dependency edge and non-source file."""

        return {
            "schema_version": MACHINE_INDEX_SCHEMA_VERSION,
            "files": [
                {
                    "path": "README.md",
                    "hash": "b" * 64,
                    "size": 12,
                    "language": "unknown",
                    "kind": "doc",
                    "public_symbols": [],
                    "internal_symbols": [],
                    "fan_in": 0,
                    "fan_out": 0,
                },
                {
                    "path": "src/app.py",
                    "hash": "a" * 64,
                    "size": 128,
                    "language": "python",
                    "kind": "source",
                    "public_symbols": ["App", "run"],
                    "internal_symbols": ["_helper"],
                    "fan_in": 0,
                    "fan_out": 1,
                },
                {
                    "path": "src/config.py",
                    "hash": "c" * 64,
                    "size": 64,
                    "language": "python",
                    "kind": "source",
                    "public_symbols": ["Config"],
                    "internal_symbols": [],
                    "fan_in": 1,
                    "fan_out": 0,
                },
            ],
            "dependency_graph": {
                "src/app.py": ["src/config.py"],
                "src/config.py": [],
            },
            "dependency_order": ["src/config.py", "src/app.py"],
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.output_dir)

    def test_extract_file_metadata(self):
        # 1ファイルのメタデータ抽出テスト
        meta = extract_file_metadata(self.runner_file, self.test_dir)
        self.assertEqual(meta["path"], "src/runner.py")
        self.assertEqual(meta["language"], "python")
        self.assertEqual(meta["kind"], "source")
        self.assertTrue(len(meta["hash"]) > 0)
        self.assertEqual(meta["size"], self.runner_file.stat().st_size)
        self.assertEqual(meta["line_count"], len(self.runner_file.read_text(encoding="utf-8").splitlines()))
        self.assertTrue(meta["readable"])

        meta_test = extract_file_metadata(self.runner_test_file, self.test_dir)
        self.assertEqual(meta_test["kind"], "test")

        meta_toml = extract_file_metadata(self.toml_file, self.test_dir)
        self.assertEqual(meta_toml["kind"], "config")

        # 通常ソースだが「test」を含むファイル名の境界値テスト
        helpers_file = self.src_dir / "test_helpers.py"
        helpers_file.write_text("def helper(): pass", encoding="utf-8")
        meta_helpers = extract_file_metadata(helpers_file, self.test_dir)
        self.assertEqual(meta_helpers["kind"], "source")

        tester_file = self.src_dir / "auth_tester.py"
        tester_file.write_text("class AuthTester: pass", encoding="utf-8")
        meta_tester = extract_file_metadata(tester_file, self.test_dir)
        self.assertEqual(meta_tester["kind"], "source")

        # 独自パッケージフォルダ配下のテストヘルパーの境界値テスト
        kuroko_helpers = self.test_dir / "kuroko/test_utils.py"
        kuroko_helpers.parent.mkdir(parents=True, exist_ok=True)
        kuroko_helpers.write_text("def helper(): pass", encoding="utf-8")
        meta_kuroko_helpers = extract_file_metadata(kuroko_helpers, self.test_dir)
        self.assertEqual(meta_kuroko_helpers["kind"], "source")

        # 新しい設定・境界ファイルの判定テスト
        meta_conftest = extract_file_metadata(self.conftest_file, self.test_dir)
        self.assertEqual(meta_conftest["kind"], "config")

        meta_docker = extract_file_metadata(self.dockerfile_file, self.test_dir)
        self.assertEqual(meta_docker["kind"], "config")

        meta_workflow = extract_file_metadata(self.workflow_file, self.test_dir)
        self.assertEqual(meta_workflow["kind"], "config")

        meta_yaml = extract_file_metadata(self.yaml_config_file, self.test_dir)
        self.assertEqual(meta_yaml["kind"], "config")

    def test_extract_file_symbols(self):
        # シンボル抽出テスト
        symbols_info = extract_file_symbols(self.runner_file, self.test_dir)
        self.assertEqual(symbols_info["path"], "src/runner.py")

        # class Runner, def run, def main の抽出を確認
        symbol_names = [sym["name"] for sym in symbols_info["symbols"]]
        self.assertIn("Runner", symbol_names)
        self.assertIn("Runner.run", symbol_names)
        self.assertIn("main", symbol_names)

        # imports 抽出の確認
        imports = [imp["module"] for imp in symbols_info["imports"]]
        self.assertIn("os", imports)
        self.assertIn("sys", imports)
        self.assertIn("src.config", imports)

        # exports 抽出の確認 (Python のデフォルトは all 以外の public シンボル等)
        self.assertIn("Runner", symbols_info["exports"])
        self.assertIn("main", symbols_info["exports"])

    def test_build_repo_map_summary(self):
        # repo_map サマリー作成のテスト
        files_meta = [
            extract_file_metadata(self.runner_file, self.test_dir),
            extract_file_metadata(self.config_file, self.test_dir),
            extract_file_metadata(self.runner_test_file, self.test_dir),
            extract_file_metadata(self.toml_file, self.test_dir),
        ]
        
        summary = build_repo_map_summary(self.test_dir, files_meta)
        
        # ディレクトリサマリーの確認
        self.assertIn("src", summary["directories"])
        self.assertEqual(summary["directories"]["src"]["files"], 2)
        self.assertIn("python", summary["directories"]["src"]["languages"])

        # エントリポイントの確認
        self.assertIn("pyproject.toml: test-cli -> src.runner:main", summary["entrypoints"])

        # テストファイルの確認
        self.assertIn("tests/test_runner.py", summary["tests"])

    def test_detect_attention_points(self):
        # アテンションポイント（リスクや警告）の検出テスト
        files_meta = [
            extract_file_metadata(self.runner_file, self.test_dir),
            extract_file_metadata(self.config_file, self.test_dir),
            extract_file_metadata(self.legacy_file, self.test_dir),
            extract_file_metadata(self.init_file, self.test_dir),
            extract_file_metadata(self.toml_file, self.test_dir),
        ]
        
        symbols_list = [
            extract_file_symbols(self.runner_file, self.test_dir),
            extract_file_symbols(self.config_file, self.test_dir),
            extract_file_symbols(self.legacy_file, self.test_dir),
            extract_file_symbols(self.init_file, self.test_dir),
            extract_file_symbols(self.toml_file, self.test_dir),
        ]

        attention = detect_attention_points(self.test_dir, files_meta, symbols_list)
        
        # 注意項目の検出を確認
        attention_texts = [att for att in attention]
        
        # 1. legacy.py は 300 行以上のため large file であること
        self.assertTrue(any("large" in text and "legacy.py" in text for text in attention_texts))
        # 2. config.py にはテストがない
        self.assertTrue(any("no tests" in text and "config.py" in text for text in attention_texts))
        # 3. runner.py に TODO が含まれる
        self.assertTrue(any("TODO/FIXME" in text and "runner.py" in text for text in attention_texts))
        # 4. __init__.py はテスト不足警告から除外されていること
        self.assertFalse(any("no tests" in text and "__init__.py" in text for text in attention_texts))

    def test_analyze_machine_level(self):
        # level 0 全体プロセスのテスト
        analyze_machine_level(self.test_dir, self.output_dir)
        
        # 出力ファイルの存在確認
        json_path = self.output_dir / "machine_analysis.json"
        yaml_path = self.output_dir / "machine_analysis.yaml"
        report_path = self.output_dir / "machine_report.md"

        self.assertTrue(json_path.exists())
        self.assertTrue(yaml_path.exists())
        self.assertTrue(report_path.exists())

        # JSON の中身の簡易的な検証
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("files", data)
            self.assertIn("repo_map", data)
            self.assertIn("attention", data)

        # YAML の中身の簡易的な検証
        yaml_content = yaml_path.read_text(encoding="utf-8")
        self.assertIn('"files":', yaml_content)
        self.assertIn('"repo_map":', yaml_content)

        # Markdown レポートの検証
        report_content = report_path.read_text(encoding="utf-8")
        self.assertIn("# Project Machine Analysis Report", report_content)
        self.assertIn("## Repo Map Summary", report_content)
        self.assertIn("## Attention Points", report_content)

    def test_machine_analysis_yaml_quotes_source_summary_scalars(self):
        special_values = {
            "colon_comment.py": "key: value # not a comment",
            "yaml_special.py": "- [special]: true",
        }
        for filename, summary_text in special_values.items():
            (self.src_dir / filename).write_text(
                f'"""{summary_text}"""\n', encoding="utf-8"
            )

        analyze_machine_level(self.test_dir, self.output_dir)
        yaml_content = (self.output_dir / "machine_analysis.yaml").read_text(
            encoding="utf-8"
        )

        for summary_text in special_values.values():
            with self.subTest(summary_text=summary_text):
                self.assertIn(
                    f'"value": {json.dumps(summary_text, ensure_ascii=False)}',
                    yaml_content,
                )
                self.assertNotIn(f'"value": {summary_text}', yaml_content)

    def test_machine_analysis_yaml_quotes_path_keys(self):
        edge_dir = self.src_dir / "special: [dir]"
        edge_dir.mkdir()
        edge_file = edge_dir / "edge.py"
        edge_file.write_text("import billing\n", encoding="utf-8")

        analyze_machine_level(self.test_dir, self.output_dir)
        yaml_content = (self.output_dir / "machine_analysis.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn(f'{json.dumps("src/special: [dir]")}:', yaml_content)
        self.assertIn(f'{json.dumps("src/special: [dir]/edge.py")}:', yaml_content)

    def test_readme_describes_current_machine_index_contract(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
            encoding="utf-8"
        )
        contract_intro = next(
            line
            for line in readme.splitlines()
            if line.startswith("`machine_index.json` は")
        )

        self.assertIn("schema version `2.2`", contract_intro)
        self.assertIn("`module_summary`", contract_intro)
        self.assertNotIn("schema version `2.0`", contract_intro)

    def test_gitignore_filtering(self):
        # .gitignore ファイルの作成
        gitignore_file = self.test_dir / ".gitignore"
        gitignore_file.write_text(
            "# ignore python caches\n"
            "__pycache__/\n"
            "*.pyc\n"
            "# ignore build artifacts\n"
            "dist/\n"
            "build/\n"
            "# ignore test temp directory\n"
            "temp_to_ignore/\n"
            "# neglect complex negation pattern for safety\n"
            "!src/keep_me.py\n",
            encoding="utf-8"
        )
        
        # 除外されるべきフォルダとファイルの作成
        ignored_dir = self.test_dir / "temp_to_ignore"
        ignored_dir.mkdir()
        ignored_file = ignored_dir / "should_be_ignored.py"
        ignored_file.write_text("print('ignored')", encoding="utf-8")
        
        # 除外されるべきキャッシュフォルダの作成
        pytest_cache_dir = self.test_dir / ".pytest_cache"
        pytest_cache_dir.mkdir()
        pytest_cache_file = pytest_cache_dir / "nodeids"
        pytest_cache_file.write_text("nodeid_data", encoding="utf-8")
        
        serena_dir = self.test_dir / ".serena"
        serena_dir.mkdir()
        serena_file = serena_dir / "document_symbols.pkl"
        serena_file.write_text("pickle_data", encoding="utf-8")

        egg_info_dir = self.test_dir / "test_project.egg-info"
        egg_info_dir.mkdir()
        egg_info_file = egg_info_dir / "PKG-INFO"
        egg_info_file.write_text("pkg_info_data", encoding="utf-8")
        
        # 無視されない通常ファイル
        kept_file = self.src_dir / "keep_me.py"
        kept_file.write_text("print('keep')", encoding="utf-8")
        
        result = analyze_machine_level(self.test_dir, self.output_dir)
        files_paths = [f["path"] for f in result["files"]]
        
        # 無視されるべきファイルが含まれていないことを確認
        self.assertNotIn("temp_to_ignore/should_be_ignored.py", files_paths)
        self.assertNotIn(".pytest_cache/nodeids", files_paths)
        self.assertNotIn(".serena/document_symbols.pkl", files_paths)
        self.assertNotIn("test_project.egg-info/PKG-INFO", files_paths)
        
        # 通常ファイルが維持されていることを確認
        self.assertIn("src/keep_me.py", files_paths)

    def test_machine_synthesized_reports(self):
        # 1回目：ドキュメントなし（カバレッジ0%）の検証
        analyze_machine_level(self.test_dir, self.output_dir)
        
        index_path = self.output_dir / "index.md"
        report_path = self.output_dir / "analysis_report.md"
        
        self.assertTrue(index_path.exists())
        self.assertTrue(report_path.exists())
        
        report_content = report_path.read_text(encoding="utf-8")
        # コントローラー互換フォーマットの検証
        self.assertIn("Status:** finished  \n", report_content)
        self.assertIn("Steps Used:** 0  \n", report_content)
        self.assertIn("Approx Tokens:** 0  \n", report_content)
        self.assertIn("## Source Coverage", report_content)
        self.assertIn("Source files discovered:", report_content)
        self.assertIn("Source files missing matching docs:", report_content)
        
        # 詳細な行末スペースとセクションの完全同期テスト
        self.assertIn("Root Directory:** `" + str(self.test_dir.resolve()) + "`  \n", report_content)
        self.assertIn("### Weak Or Failed Docs\n\n- (none)", report_content)
        self.assertIn("### Extra Docs Without Matching Source\n\n- (none)", report_content)
        
        index_content = index_path.read_text(encoding="utf-8")
        self.assertIn("Directory: " + self.test_dir.name, index_content)
        self.assertIn("Stale or Newly Added Files", index_content)
        # 通常のソースファイルは要説明として検出されること
        self.assertIn("src/runner.py", index_content)
        # テストファイルや設定ファイルは要説明に入っていないこと
        self.assertNotIn("tests/test_runner.py", index_content)
        self.assertNotIn("pyproject.toml", index_content)
        
        # 境界・ノイズ設定ファイルが除外されていることの検証
        self.assertNotIn("conftest.py", index_content)
        self.assertNotIn("Dockerfile", index_content)
        self.assertNotIn("deploy.yml", index_content)
        self.assertNotIn("settings.yaml", index_content)

    def test_coverage_and_hash_stale_detection(self):
        # 明示 provenance を基準に、mtime ではなく source bytes の変更を stale と判定する。
        
        # validなドキュメントを作成 (runner.py.md)
        doc_dir = self.output_dir / "src"
        doc_dir.mkdir(parents=True, exist_ok=True)
        runner_doc = doc_dir / "runner.py.md"
        runner_doc.write_text("# Runner Doc", encoding="utf-8")
        import os
        # ソースコードより新しく更新時刻を設定
        runner_mtime = self.runner_file.stat().st_mtime
        os.utime(runner_doc, (runner_mtime + 10.0, runner_mtime + 10.0))

        # staleなドキュメントを作成 (config.py.md)
        config_doc = doc_dir / "config.py.md"
        config_doc.write_text("# Config Doc", encoding="utf-8")
        config_mtime = self.config_file.stat().st_mtime
        # mtime の前後は freshness 判定に影響しない。
        os.utime(config_doc, (config_mtime - 10.0, config_mtime - 10.0))

        write_doc_provenance_atomic(
            self.output_dir / "doc_provenance.json",
            {
                "schema_version": "1.0",
                "hash_algorithm": "sha256",
                "assertions": [
                    {
                        "source_path": "src/config.py",
                        "doc_path": "src/config.py.md",
                        "recorded_source_hash": hashlib.sha256(
                            self.config_file.read_bytes()
                        ).hexdigest(),
                        "recorded_doc_hash": hashlib.sha256(
                            config_doc.read_bytes()
                        ).hexdigest(),
                        "producer": "test.producer",
                        "producer_version": "1",
                    },
                    {
                        "source_path": "src/runner.py",
                        "doc_path": "src/runner.py.md",
                        "recorded_source_hash": hashlib.sha256(
                            self.runner_file.read_bytes()
                        ).hexdigest(),
                        "recorded_doc_hash": hashlib.sha256(
                            runner_doc.read_bytes()
                        ).hexdigest(),
                        "producer": "test.producer",
                        "producer_version": "1",
                    },
                ],
            },
        )

        # Source bytes changed after the producer assertion; only this target
        # should become stale.
        self.config_file.write_text(
            self.config_file.read_text(encoding="utf-8") + "\n# changed\n",
            encoding="utf-8",
        )

        # 再度分析を実行
        analyze_machine_level(self.test_dir, self.output_dir)

        index_path = self.output_dir / "index.md"
        report_path = self.output_dir / "analysis_report.md"
        
        report_content = report_path.read_text(encoding="utf-8")
        # runner.py と config.py にドキュメントが存在するため、
        # カバレッジが 0% より大きくなること
        self.assertNotIn("Coverage: 0.0%", report_content)

        freshness = load_doc_freshness(self.output_dir / "doc_freshness.json")
        statuses = {
            entry["source_path"]: (entry["status"], entry["reason"])
            for entry in freshness["entries"]
        }
        self.assertEqual(statuses["src/runner.py"], ("fresh", "source_hash_match"))
        self.assertEqual(
            statuses["src/config.py"], ("stale", "source_hash_mismatch")
        )
        
        index_content = index_path.read_text(encoding="utf-8")
        # config.py は source bytes の変更により stale としてリストされること
        self.assertTrue(any("config.py" in line and "stale" in line for line in index_content.splitlines()))
        # runner.py は valid なので、git modified などの別の理由がない限り index.md の「要説明」から除外されること
        self.assertFalse(any("src/runner.py" in line and "missing" in line for line in index_content.splitlines()))
        # テストファイルは missing doc と判定されないため、リストされないこと
        self.assertNotIn("tests/test_runner.py", index_content)

    def test_freshness_is_mtime_independent_and_tracks_content_changes(self):
        root = self.test_dir / "freshness-root"
        source = root / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"def app():\n    return 1\n")

        output = self.output_dir / "freshness-output"
        doc = output / "src" / "app.py.md"
        doc.parent.mkdir(parents=True)
        doc.write_bytes(b"# App\n")
        provenance_path = output / "doc_provenance.json"
        write_doc_provenance_atomic(
            provenance_path,
            {
                "schema_version": "1.0",
                "hash_algorithm": "sha256",
                "assertions": [
                    {
                        "source_path": "src/app.py",
                        "doc_path": "src/app.py.md",
                        "recorded_source_hash": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        "recorded_doc_hash": hashlib.sha256(doc.read_bytes()).hexdigest(),
                        "producer": "test.producer",
                        "producer_version": "1",
                    }
                ],
            },
        )

        analyze_machine_level(root, output)
        first_freshness = (output / "doc_freshness.json").read_bytes()
        first_provenance = provenance_path.read_bytes()
        first_entry = load_doc_freshness(output / "doc_freshness.json")["entries"][0]
        self.assertEqual((first_entry["status"], first_entry["reason"]), ("fresh", "source_hash_match"))

        source_mtime = source.stat().st_mtime
        os.utime(source, (source_mtime + 100.0, source_mtime + 100.0))
        os.utime(doc, (source_mtime - 100.0, source_mtime - 100.0))
        analyze_machine_level(root, output)

        self.assertEqual(first_freshness, (output / "doc_freshness.json").read_bytes())
        self.assertEqual(first_provenance, provenance_path.read_bytes())
        mtime_entry = load_doc_freshness(output / "doc_freshness.json")["entries"][0]
        self.assertEqual((mtime_entry["status"], mtime_entry["reason"]), ("fresh", "source_hash_match"))

        source.write_bytes(b"def app():\n    return 2\n")
        analyze_machine_level(root, output)
        changed_freshness = load_doc_freshness(output / "doc_freshness.json")
        changed_entry = changed_freshness["entries"][0]
        self.assertEqual(
            (changed_entry["status"], changed_entry["reason"]),
            ("stale", "source_hash_mismatch"),
        )
        self.assertNotEqual(first_freshness, (output / "doc_freshness.json").read_bytes())

    def test_machine_analysis_preserves_freshness_reason_diagnostics(self):
        root = self.test_dir / "freshness-diagnostics-root"
        source = root / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"def app():\n    return 1\n")

        output = self.output_dir / "freshness-diagnostics-output"
        doc = output / "src" / "app.py.md"
        doc.parent.mkdir(parents=True)
        doc.write_bytes(b"# App\n")

        result = analyze_machine_level(root, output)
        expected_missing = ("doc_status", "recorded_source_hash_missing", "src/app.py")
        self.assertIn(
            expected_missing,
            {
                (item["detector"], item["code"], item["path"])
                for item in result["attention_diagnostics"]
            },
        )
        machine_analysis = json.loads(
            (output / "machine_analysis.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            expected_missing,
            {
                (item["detector"], item["code"], item["path"])
                for item in machine_analysis["attention_diagnostics"]
            },
        )

        (output / "doc_provenance.json").write_text("{invalid", encoding="utf-8")
        result = analyze_machine_level(root, output)
        expected_invalid = ("doc_status", "provenance_artifact_invalid", "src/app.py")
        self.assertIn(
            expected_invalid,
            {
                (item["detector"], item["code"], item["path"])
                for item in result["attention_diagnostics"]
            },
        )
        machine_analysis = json.loads(
            (output / "machine_analysis.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            expected_invalid,
            {
                (item["detector"], item["code"], item["path"])
                for item in machine_analysis["attention_diagnostics"]
            },
        )

    def test_machine_scan_rejects_unsafe_provenance_artifacts(self):
        root = self.test_dir / "provenance-boundary-root"
        output = self.output_dir / "provenance-boundary-output"
        source = root / "src" / "app.py"
        document = output / "src" / "app.py.md"
        source.parent.mkdir(parents=True)
        document.parent.mkdir(parents=True)
        source_bytes = b"def app():\n    return 1\n"
        doc_bytes = b"# App\n"
        source.write_bytes(source_bytes)
        document.write_bytes(doc_bytes)

        external_provenance = self.test_dir / "external-provenance" / "doc.json"
        write_doc_provenance_atomic(
            external_provenance,
            {
                "schema_version": "1.0",
                "hash_algorithm": "sha256",
                "assertions": [
                    {
                        "source_path": "src/app.py",
                        "doc_path": "src/app.py.md",
                        "recorded_source_hash": hashlib.sha256(source_bytes).hexdigest(),
                        "recorded_doc_hash": hashlib.sha256(doc_bytes).hexdigest(),
                        "producer": "test.producer",
                        "producer_version": "1",
                    }
                ],
            },
        )
        provenance_path = output / "doc_provenance.json"
        try:
            provenance_path.symlink_to(external_provenance)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        external_bytes = external_provenance.read_bytes()
        result = analyze_machine_level(root, output)
        entry = load_doc_freshness(output / "doc_freshness.json")["entries"][0]
        self.assertEqual(
            (entry["status"], entry["reason"]),
            ("unknown", "provenance_artifact_invalid"),
        )
        self.assertEqual(result["coverage_summary"]["unknown_docs"], 1)
        self.assertEqual(external_provenance.read_bytes(), external_bytes)

        provenance_path.unlink()
        if hasattr(os, "mkfifo"):
            try:
                os.mkfifo(provenance_path)
            except (NotImplementedError, OSError):
                pass
            else:
                result = analyze_machine_level(root, output)
                entry = load_doc_freshness(output / "doc_freshness.json")["entries"][0]
                self.assertEqual(
                    (entry["status"], entry["reason"]),
                    ("unknown", "provenance_artifact_invalid"),
                )
                self.assertEqual(result["coverage_summary"]["unknown_docs"], 1)

    def test_machine_scan_distinguishes_fresh_stale_missing_and_unknown(self):
        root = self.test_dir / "freshness-boundary-root"
        output = self.output_dir / "freshness-boundary-output"
        source_bytes = {
            "src/fresh.py": b"def fresh():\n    return 1\n",
            "src/stale.py": b"def stale():\n    return 1\n",
            "src/missing.py": b"def missing():\n    return 1\n",
            "src/unknown.py": b"def unknown():\n    return 1\n",
        }
        doc_bytes = {
            "src/fresh.py.md": b"# Fresh\n",
            "src/stale.py.md": b"# Stale\n",
            "src/unknown.py.md": b"# Unknown\n",
        }
        for relative_path, payload in source_bytes.items():
            source_path = root / relative_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(payload)
        for relative_path, payload in doc_bytes.items():
            doc_path = output / relative_path
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_bytes(payload)

        digest = lambda payload: hashlib.sha256(payload).hexdigest()
        write_doc_provenance_atomic(
            output / "doc_provenance.json",
            {
                "schema_version": "1.0",
                "hash_algorithm": "sha256",
                "assertions": [
                    {
                        "source_path": "src/fresh.py",
                        "doc_path": "src/fresh.py.md",
                        "recorded_source_hash": digest(source_bytes["src/fresh.py"]),
                        "recorded_doc_hash": digest(doc_bytes["src/fresh.py.md"]),
                        "producer": "test.producer",
                        "producer_version": "1",
                    },
                    {
                        "source_path": "src/stale.py",
                        "doc_path": "src/stale.py.md",
                        "recorded_source_hash": digest(source_bytes["src/stale.py"]),
                        "recorded_doc_hash": digest(doc_bytes["src/stale.py.md"]),
                        "producer": "test.producer",
                        "producer_version": "1",
                    },
                ],
            },
        )
        (root / "src/stale.py").write_bytes(b"def stale():\n    return 2\n")

        result = analyze_machine_level(root, output)
        freshness_path = output / "doc_freshness.json"
        freshness = load_doc_freshness(freshness_path)
        entries = {entry["source_path"]: entry for entry in freshness["entries"]}
        expected = {
            "src/fresh.py": ("fresh", "source_hash_match", "src/fresh.py.md"),
            "src/stale.py": ("stale", "source_hash_mismatch", "src/stale.py.md"),
            "src/missing.py": ("missing", "doc_missing", None),
            "src/unknown.py": (
                "unknown",
                "recorded_source_hash_missing",
                "src/unknown.py.md",
            ),
        }
        required_fields = {
            "source_path",
            "doc_path",
            "expected_doc_paths",
            "current_source_hash",
            "recorded_source_hash",
            "doc_hash",
            "status",
            "reason",
            "diagnostics",
        }
        self.assertEqual(set(entries), set(expected))
        for source_path, (status, reason, doc_path) in expected.items():
            with self.subTest(source_path=source_path):
                entry = entries[source_path]
                self.assertEqual(set(entry), required_fields)
                self.assertEqual(
                    entry["expected_doc_paths"], list(expected_doc_paths(source_path))
                )
                self.assertEqual(
                    (entry["status"], entry["reason"], entry["doc_path"]),
                    (status, reason, doc_path),
                )
                self.assertIsNotNone(entry["current_source_hash"])
                if doc_path is None:
                    self.assertIsNone(entry["doc_hash"])
                    self.assertIsNone(entry["recorded_source_hash"])
                else:
                    self.assertIsNotNone(entry["doc_hash"])
                if status in ("fresh", "stale"):
                    self.assertIsNotNone(entry["recorded_source_hash"])
                else:
                    self.assertIsNone(entry["recorded_source_hash"])

        self.assertEqual(
            freshness["counts"], {"missing": 1, "fresh": 1, "stale": 1, "unknown": 1}
        )
        self.assertEqual(
            result["coverage_summary"]["unknown_docs"], 1
        )
        self.assertEqual(
            result["coverage_summary"]["missing_docs"], 1
        )
        statuses, _diagnostics = build_doc_status_snapshot(
            root, result["files"], output
        )
        self.assertEqual(
            {
                path: statuses[path]
                for path in expected
            },
            {
                "src/fresh.py": "current",
                "src/stale.py": "stale",
                "src/missing.py": "missing",
                "src/unknown.py": "unavailable",
            },
        )
        self.assertIn(
            "unknown doc (recorded_source_hash_missing)",
            (output / "index.md").read_text(encoding="utf-8"),
        )

    def test_machine_scan_rename_does_not_inherit_old_provenance(self):
        root = self.test_dir / "rename-boundary-root"
        output = self.output_dir / "rename-boundary-output"
        old_source = root / "src" / "old.py"
        old_doc = output / "src" / "old.py.md"
        new_source = root / "src" / "new.py"
        new_doc = output / "src" / "new.py.md"
        source_payload = b"def value():\n    return 1\n"
        doc_payload = b"# Value\n"
        old_source.parent.mkdir(parents=True)
        old_source.write_bytes(source_payload)
        old_doc.parent.mkdir(parents=True)
        old_doc.write_bytes(doc_payload)
        write_doc_provenance_atomic(
            output / "doc_provenance.json",
            {
                "schema_version": "1.0",
                "hash_algorithm": "sha256",
                "assertions": [
                    {
                        "source_path": "src/old.py",
                        "doc_path": "src/old.py.md",
                        "recorded_source_hash": hashlib.sha256(source_payload).hexdigest(),
                        "recorded_doc_hash": hashlib.sha256(doc_payload).hexdigest(),
                        "producer": "test.producer",
                        "producer_version": "1",
                    }
                ],
            },
        )

        old_source.rename(new_source)
        new_doc.write_bytes(doc_payload)
        result = analyze_machine_level(root, output)
        freshness = load_doc_freshness(output / "doc_freshness.json")
        self.assertEqual(
            [entry["source_path"] for entry in freshness["entries"]],
            ["src/new.py"],
        )
        entry = freshness["entries"][0]
        self.assertEqual(
            (entry["status"], entry["reason"], entry["doc_path"]),
            ("unknown", "recorded_source_hash_missing", "src/new.py.md"),
        )
        self.assertIsNone(entry["recorded_source_hash"])
        self.assertIn("orphan_provenance_assertion", freshness["diagnostics"])
        self.assertEqual(result["coverage_summary"]["fresh_docs"], 0)
        self.assertEqual(result["coverage_summary"]["unknown_docs"], 1)

    def test_machine_scan_mapping_collisions_and_unsafe_docs_never_become_current(self):
        root = self.test_dir / "mapping-boundary-root"
        output = self.output_dir / "mapping-boundary-output"
        source_payloads = {
            "src/ambiguous.py": b"def ambiguous():\n    return 1\n",
            "src/shared.py": b"def shared_py():\n    return 1\n",
            "src/shared.js": b"function sharedJs() { return 1; }\n",
            "src/unsafe.py": b"def unsafe():\n    return 1\n",
        }
        for relative_path, payload in source_payloads.items():
            source_path = root / relative_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(payload)

        docs = {
            "src/ambiguous.py.md": b"# first candidate\n",
            "src/ambiguous.md": b"# second candidate\n",
            "src/shared.md": b"# shared candidate\n",
        }
        for relative_path, payload in docs.items():
            doc_path = output / relative_path
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_bytes(payload)

        outside = self.test_dir / "outside-doc.md"
        outside.write_bytes(b"outside\n")
        unsafe_doc = output / "src" / "unsafe.py.md"
        try:
            unsafe_doc.symlink_to(outside)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        result = analyze_machine_level(root, output)
        freshness = load_doc_freshness(output / "doc_freshness.json")
        entries = {entry["source_path"]: entry for entry in freshness["entries"]}
        expected_reasons = {
            "src/ambiguous.py": "ambiguous_doc_mapping",
            "src/shared.py": "doc_identity_collision",
            "src/shared.js": "doc_identity_collision",
            "src/unsafe.py": "unsafe_doc_path",
        }
        self.assertEqual(set(entries), set(expected_reasons))
        for source_path, reason in expected_reasons.items():
            with self.subTest(source_path=source_path):
                entry = entries[source_path]
                self.assertEqual((entry["status"], entry["reason"]), ("unknown", reason))
                self.assertIsNone(entry["doc_path"])
                self.assertIsNone(entry["doc_hash"])
                self.assertIsNone(entry["recorded_source_hash"])

        self.assertEqual(freshness["counts"], {"missing": 0, "fresh": 0, "stale": 0, "unknown": 4})
        statuses, _diagnostics = build_doc_status_snapshot(
            root, result["files"], output
        )
        self.assertEqual(
            {path: statuses[path] for path in expected_reasons},
            {path: "unavailable" for path in expected_reasons},
        )
        self.assertNotIn("source_hash_match", freshness["diagnostics"])

    def test_machine_scan_freshness_bytes_are_stable_for_creation_order(self):
        logical_files = [
            ("src/root.py", b"from src.alpha import Alpha\n"),
            ("src/alpha.py", b"class Alpha: pass\n"),
            ("src/zed.py", b"class Zed: pass\n"),
            ("src/root.py.md", b"# Root\n"),
            ("src/alpha.py.md", b"# Alpha\n"),
            ("src/zed.py.md", b"# Zed\n"),
        ]
        freshness_bytes = []
        for suffix, creation_order in (
            ("forward", logical_files),
            ("reverse", list(reversed(logical_files))),
        ):
            root = self.test_dir / f"freshness-order-{suffix}-root"
            output = self.output_dir / f"freshness-order-{suffix}-output"
            for relative_path, payload in creation_order:
                destination_root = root if not relative_path.endswith(".md") else output
                path = destination_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            analyze_machine_level(root, output)
            freshness_bytes.append((output / "doc_freshness.json").read_bytes())

        self.assertEqual(freshness_bytes[0], freshness_bytes[1])

    def test_dependency_graph_building(self):
        # 依存関係抽出と Mermaid グラフ生成、トポロジカルソートの統合検証テスト
        # setUp にて runner.py が src.config に依存している
        result = analyze_machine_level(self.test_dir, self.output_dir)
        
        # 1. JSON の出力内容に symbols と imports があり、逆引き解決されていることの検証
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("dependency_graph", data)
            dep_graph = data["dependency_graph"]
            # runner.py -> config.py の依存関係
            self.assertIn("src/config.py", dep_graph.get("src/runner.py", []))

        # 2. machine_report.md に Mermaid グラフが出力されていることの検証
        report_path = self.output_dir / "machine_report.md"
        report_content = report_path.read_text(encoding="utf-8")
        self.assertIn("## Dependency Graph Map", report_content)
        self.assertIn("graph TD", report_content)
        # エスケープされたノード名が定義されていること
        self.assertIn("src/runner.py", report_content)
        self.assertIn("src/config.py", report_content)

        # 3. index.md の Stale or Newly Added Files がボトムアップ推奨読解順（config.py が runner.py より先）に並んでいることの検証
        index_path = self.output_dir / "index.md"
        index_content = index_path.read_text(encoding="utf-8")
        lines = index_content.splitlines()
        
        runner_idx = -1
        config_idx = -1
        for i, line in enumerate(lines):
            if "src/runner.py" in line:
                runner_idx = i
            elif "src/config.py" in line:
                config_idx = i
                
        self.assertTrue(config_idx != -1 and runner_idx != -1)
        # config.py は runner.py の依存先なので、先に読むべき（インデックスの上位）
        self.assertTrue(config_idx < runner_idx, f"Expected config.py (idx {config_idx}) to be before runner.py (idx {runner_idx})")

    def test_dependency_cycles_and_determinism(self):
        # 循環参照が存在する場合のハング防止、および決定論的なアルファベット順ソートの検証テスト
        
        # 相互参照するダミーファイルを作成
        cycle_a = self.src_dir / "cycle_a.py"
        cycle_a.write_text("from src.cycle_b import B\nclass A: pass", encoding="utf-8")
        cycle_b = self.src_dir / "cycle_b.py"
        cycle_b.write_text("from src.cycle_a import A\nclass B: pass", encoding="utf-8")
        
        # 無限ループせずに正常終了すること
        result = analyze_machine_level(self.test_dir, self.output_dir)
        
        # JSON内の dependency_graph に循環が記録、あるいは安全に処理されていること
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("dependency_graph", data)
            
        # index.md が一意に出力されていることの確認
        index_path = self.output_dir / "index.md"
        index_content = index_path.read_text(encoding="utf-8")
        self.assertIn("src/cycle_a.py", index_content)
        self.assertIn("src/cycle_b.py", index_content)

    def test_mermaid_limit_guard(self):
        # 大規模プロジェクト時の Mermaid ノード数制限ガードの検証テスト
        # 制限（50件）を超えるように55個のダミーソースファイルを作成
        for i in range(55):
            dummy_file = self.src_dir / f"dummy_{i:02d}.py"
            # 決定論的な依存関係を少し作っておく
            if i > 0:
                dummy_file.write_text(f"from src.dummy_{i-1:02d} import X", encoding="utf-8")
            else:
                dummy_file.write_text("pass", encoding="utf-8")
                
        analyze_machine_level(self.test_dir, self.output_dir)
        
        report_path = self.output_dir / "machine_report.md"
        report_content = report_path.read_text(encoding="utf-8")
        
        # Mermaid グラフがスキップされ、注意警告文になっていること
        self.assertIn("Dependency graph is too large to display as a Mermaid diagram", report_content)
        self.assertNotIn("graph TD", report_content)

    def test_git_status_extraction(self):
        # Gitステータス抽出と例外安全、JSON出力の検証
        # ダミーのGit変更ファイルを模擬するため、一時ディレクトリ上にGitリポジトリを初期化してテストすることも可能だが、
        # ここでは get_git_modified_files の戻り値が analyze_machine_level の処理を通じて
        # JSON内の files_meta 各項目の git_status キーに格納されるかを検証。
        result = analyze_machine_level(self.test_dir, self.output_dir)
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 各ファイルのメタデータに git_status がデフォルト（"?" や "unchanged" や Noneなど）で含まれていること
            for file_info in data["files"]:
                if file_info["kind"] == "source":
                    self.assertIn("git_status", file_info)

    def test_ast_outline_extraction(self):
        # ASTを用いたクラス・主要関数の抽出と例外安全の検証
        # テスト対象ファイルの1つにダミーのクラスと関数を定義しておく
        dummy_code = (
            "class MyDummyClass:\n"
            "    def method(self):\n"
            "        pass\n"
            "def my_dummy_function():\n"
            "    pass\n"
        )
        dummy_py = self.src_dir / "dummy_ast.py"
        dummy_py.write_text(dummy_code, encoding="utf-8")

        # 構文エラーのファイルも作成して例外安全（クラッシュしないこと）を検証
        bad_py = self.src_dir / "bad_syntax.py"
        bad_py.write_text("class Unfinished:", encoding="utf-8")

        result = analyze_machine_level(self.test_dir, self.output_dir)
        
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # files_meta 内の各ファイルに classes / functions フィールドが存在すること
            found_dummy = False
            found_bad = False
            for file_info in data["files"]:
                if file_info["path"] == "src/dummy_ast.py":
                    found_dummy = True
                    self.assertIn("classes", file_info)
                    self.assertIn("functions", file_info)
                    self.assertEqual(file_info["classes"], ["MyDummyClass"])
                    self.assertEqual(file_info["functions"], ["my_dummy_function"])
                elif file_info["path"] == "src/bad_syntax.py":
                    found_bad = True
                    self.assertIn("classes", file_info)
                    self.assertIn("functions", file_info)
                    self.assertEqual(file_info["classes"], [])
                    self.assertEqual(file_info["functions"], [])
            self.assertTrue(found_dummy)
            self.assertTrue(found_bad)

    def test_fan_in_fan_out_metrics(self):
        # ファンイン・ファンアウトメトリクスの算出、JSON保存、およびAttention Pointsへの追加検証
        # setUpにより runner.py が src/config.py をインポートしているため、
        # config.py のファンインは >= 1, runner.py のファンアウトは >= 1 になるはず。
        result = analyze_machine_level(self.test_dir, self.output_dir)
        
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # files 各項目の fan_in / fan_out キーの存在と値の検証
            for file_info in data["files"]:
                if file_info["path"] == "src/config.py":
                    self.assertIn("fan_in", file_info)
                    self.assertGreaterEqual(file_info["fan_in"], 1)
                elif file_info["path"] == "src/runner.py":
                    self.assertIn("fan_out", file_info)
                    self.assertGreaterEqual(file_info["fan_out"], 1)

    def test_markdown_noise_reduction_details(self):
        # クラス・主要関数数が閾値以上、または常に index.md 等で折りたたみ（details）構造になっているかの検証
        result = analyze_machine_level(self.test_dir, self.output_dir)
        index_path = self.output_dir / "index.md"
        index_content = index_path.read_text(encoding="utf-8")
        
        # HTMLの details タグによる折りたたみがマークダウンに含まれていること
        self.assertIn("<details>", index_content)
        self.assertIn("</details>", index_content)

    def test_machine_index_json_generation_and_symbol_classification(self):
        # machine_index.json の生成と、public/internal シンボル分類の検証
        # テスト対象ファイルの1つに、公開および内部（_始まり）のクラスと関数を定義しておく
        dummy_code = (
            "class PublicClass:\n"
            "    pass\n"
            "class _InternalClass:\n"
            "    pass\n"
            "def public_function():\n"
            "    pass\n"
            "def _internal_function():\n"
            "    pass\n"
        )
        dummy_py = self.src_dir / "dummy_symbols.py"
        dummy_py.write_text(dummy_code, encoding="utf-8")

        result = analyze_machine_level(self.test_dir, self.output_dir)

        index_json_path = self.output_dir / "machine_index.json"
        self.assertTrue(index_json_path.exists())

        with open(index_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            found_dummy = False
            for file_info in data.get("files", []):
                if file_info["path"] == "src/dummy_symbols.py":
                    found_dummy = True
                    self.assertIn("public_symbols", file_info)
                    self.assertIn("internal_symbols", file_info)
                    
                    # 公開シンボル（先頭が _ でないもの）の検証
                    self.assertIn("PublicClass", file_info["public_symbols"])
                    self.assertIn("public_function", file_info["public_symbols"])
                    self.assertNotIn("_InternalClass", file_info["public_symbols"])
                    self.assertNotIn("_internal_function", file_info["public_symbols"])
                    
                    # 内部シンボル（先頭が _ で始まるもの）の検証
                    self.assertIn("_InternalClass", file_info["internal_symbols"])
                    self.assertIn("_internal_function", file_info["internal_symbols"])
                    self.assertNotIn("PublicClass", file_info["internal_symbols"])
                    self.assertNotIn("public_function", file_info["internal_symbols"])

            self.assertTrue(found_dummy)

    def test_machine_index_preserves_exact_symbol_order(self):
        # classes -> top-level functions の既存抽出順と public/internal 分類を固定する。
        root = self.test_dir / "symbol-order-root"
        source = root / "src" / "symbols.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "class PublicClass:\n"
            "    pass\n"
            "class _InternalClass:\n"
            "    pass\n"
            "def public_first():\n"
            "    pass\n"
            "def _internal_helper():\n"
            "    pass\n"
            "class PublicSecond:\n"
            "    pass\n"
            "def public_second():\n"
            "    pass\n",
            encoding="utf-8",
        )

        analyze_machine_level(root, self.output_dir)
        data = json.loads(
            (self.output_dir / "machine_index.json").read_text(encoding="utf-8")
        )
        entry = next(file for file in data["files"] if file["path"] == "src/symbols.py")

        self.assertEqual(
            entry["public_symbols"],
            ["PublicClass", "PublicSecond", "public_first", "public_second"],
        )
        self.assertEqual(
            entry["internal_symbols"], ["_InternalClass", "_internal_helper"]
        )

    def test_machine_index_preserves_dependency_semantics_for_branch_and_cycle(self):
        # 分岐、同一 source からの複数 edge、cycle を含む isolated fixture を使う。
        root = self.test_dir / "dependency-contract-root"
        files = {
            "src/app.py": (
                "from src.beta import Beta\n"
                "from src.alpha import Alpha\n"
                "class App: pass\n"
            ),
            "src/worker.py": (
                "from src.beta import Beta\n"
                "from src.alpha import Alpha\n"
                "def work(): pass\n"
            ),
            "src/alpha.py": (
                "from src.shared import Shared\n"
                "class Alpha: pass\n"
            ),
            "src/beta.py": (
                "from src.shared import Shared\n"
                "class Beta: pass\n"
            ),
            "src/shared.py": "class Shared: pass\n",
            "src/cycle_a.py": (
                "from src.cycle_b import B\n"
                "class A: pass\n"
            ),
            "src/cycle_b.py": (
                "from src.cycle_a import A\n"
                "class B: pass\n"
            ),
        }
        for relative_path, content in files.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        output = self.output_dir / "dependency-contract"
        analyze_machine_level(root, output)
        index_path = output / "machine_index.json"
        data = json.loads(index_path.read_text(encoding="utf-8"))
        first_bytes = index_path.read_bytes()

        self.assertEqual(
            data["dependency_graph"],
            {
                "src/alpha.py": ["src/shared.py"],
                "src/app.py": ["src/alpha.py", "src/beta.py"],
                "src/beta.py": ["src/shared.py"],
                "src/cycle_a.py": ["src/cycle_b.py"],
                "src/cycle_b.py": ["src/cycle_a.py"],
                "src/shared.py": [],
                "src/worker.py": ["src/alpha.py", "src/beta.py"],
            },
        )
        self.assertEqual(
            data["dependency_order"],
            [
                "src/shared.py",
                "src/alpha.py",
                "src/beta.py",
                "src/app.py",
                "src/worker.py",
                "src/cycle_a.py",
                "src/cycle_b.py",
            ],
        )
        self.assertEqual(
            {
                file["path"]: (file["fan_in"], file["fan_out"])
                for file in data["files"]
            },
            {
                "src/alpha.py": (2, 1),
                "src/app.py": (0, 2),
                "src/beta.py": (2, 1),
                "src/cycle_a.py": (1, 1),
                "src/cycle_b.py": (1, 1),
                "src/shared.py": (2, 0),
                "src/worker.py": (0, 2),
            },
        )

        analyze_machine_level(root, output)
        self.assertEqual(first_bytes, index_path.read_bytes())

    def test_machine_index_is_byte_stable_across_prior_analysis_and_unknown_docs(self):
        # provenance のない既存 doc は scan 回数や mtime により current/stale に
        # 遷移せず、公開 index の bytes も変わらないことを確認する。
        root = self.test_dir / "repeat-scan-root"
        source = root / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_text("def app():\n    return 'stable'\n", encoding="utf-8")

        doc = self.output_dir / "src" / "app.py.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("# App", encoding="utf-8")
        source_mtime = source.stat().st_mtime
        os.utime(doc, (source_mtime - 10.0, source_mtime - 10.0))

        first_result = analyze_machine_level(root, self.output_dir)
        index_path = self.output_dir / "machine_index.json"
        first_bytes = index_path.read_bytes()
        first_file = next(
            file for file in first_result["files"] if file["path"] == "src/app.py"
        )
        self.assertNotEqual(first_file["status"], "unchanged")
        self.assertEqual(first_result["coverage_summary"]["unknown_docs"], 1)

        second_result = analyze_machine_level(root, self.output_dir)
        second_bytes = index_path.read_bytes()
        second_file = next(
            file for file in second_result["files"] if file["path"] == "src/app.py"
        )

        self.assertEqual(second_file["status"], "unchanged")
        self.assertNotEqual(first_file["status"], second_file["status"])
        self.assertEqual(first_bytes, second_bytes)
        first_index = json.loads(first_bytes)
        second_index = json.loads(second_bytes)
        self.assertEqual(first_index["schema_version"], MACHINE_INDEX_V2_SCHEMA_VERSION)
        self.assertFalse(any(entry["kind"] == "doc_stale" for entry in first_index["attention"]))
        self.assertFalse(any(entry["kind"] == "doc_stale" for entry in second_index["attention"]))

    def test_machine_index_v21_binds_and_resolves_freshness_artifact(self):
        root = self.test_dir / "index-freshness-root"
        source = root / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_text("def app():\n    return 1\n", encoding="utf-8")

        output = self.output_dir / "index-freshness-output"
        analyze_machine_level(root, output)

        index_path = output / "machine_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        freshness_path = output / "doc_freshness.json"
        freshness = load_doc_freshness(freshness_path)

        self.assertEqual(index["schema_version"], MACHINE_INDEX_V2_SCHEMA_VERSION)
        self.assertEqual(
            index["doc_freshness"],
            {
                "path": "doc_freshness.json",
                "schema_version": freshness["schema_version"],
                "sha256": hashlib.sha256(freshness_path.read_bytes()).hexdigest(),
                "counts": freshness["counts"],
            },
        )
        self.assertEqual(resolve_machine_index_freshness(index_path), freshness)

    def test_machine_index_v20_remains_readable_without_freshness_reference(self):
        legacy = build_machine_index_v1(self._valid_machine_index_fixture())
        legacy["schema_version"] = MACHINE_INDEX_V2_LEGACY_SCHEMA_VERSION
        legacy["attention"] = []

        index_path = self.output_dir / "machine-index-v20.json"
        index_path.write_text(
            serialize_machine_index(legacy, supported_major=2),
            encoding="utf-8",
        )

        loaded = load_machine_index(index_path, supported_major=2)
        self.assertEqual(
            loaded["schema_version"], MACHINE_INDEX_V2_LEGACY_SCHEMA_VERSION
        )
        self.assertIsNone(resolve_machine_index_freshness(index_path))

    def test_machine_index_freshness_reference_fails_closed_on_mismatch(self):
        root = self.test_dir / "index-reference-root"
        source = root / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_text("def app():\n    return 1\n", encoding="utf-8")
        output = self.output_dir / "index-reference-output"
        analyze_machine_level(root, output)

        index_path = output / "machine_index.json"
        original_index_bytes = index_path.read_bytes()
        original_index = json.loads(original_index_bytes)

        unsafe_path = deepcopy(original_index)
        unsafe_path["doc_freshness"]["path"] = "../doc_freshness.json"
        with self.assertRaises(MachineIndexContractError):
            validate_machine_index(unsafe_path, supported_major=2)

        for field, value in (
            ("sha256", "0" * 64),
            ("schema_version", "1.1"),
            (
                "counts",
                {
                    **original_index["doc_freshness"]["counts"],
                    "unknown": original_index["doc_freshness"]["counts"]["unknown"] + 1,
                },
            ),
        ):
            with self.subTest(reference_field=field):
                mismatched = deepcopy(original_index)
                mismatched["doc_freshness"][field] = value
                index_path.write_text(
                    serialize_machine_index(mismatched, supported_major=2),
                    encoding="utf-8",
                )
                with self.assertRaises(MachineIndexContractError):
                    resolve_machine_index_freshness(index_path)

        # Simulate the crash window in which a new freshness artifact is
        # published but the old index is still visible to a reader.
        index_path.write_bytes(original_index_bytes)
        source.write_text("def app():\n    return 2\n", encoding="utf-8")
        analyze_machine_level(root, output)
        new_freshness_bytes = (output / "doc_freshness.json").read_bytes()
        self.assertNotEqual(
            hashlib.sha256(new_freshness_bytes).hexdigest(),
            original_index["doc_freshness"]["sha256"],
        )
        index_path.write_bytes(original_index_bytes)
        with self.assertRaises(MachineIndexContractError):
            resolve_machine_index_freshness(index_path)

    def test_machine_index_freshness_reader_rejects_replaced_symlink(self):
        root = self.test_dir / "index-reference-race-root"
        source = root / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_text("def app():\n    return 1\n", encoding="utf-8")
        output = self.output_dir / "index-reference-race-output"
        analyze_machine_level(root, output)

        index_path = output / "machine_index.json"
        freshness_path = output / "doc_freshness.json"
        replacement = output / "replaced-freshness.json"
        try:
            replacement.symlink_to(freshness_path)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        # Bypass the pre-read path inspection to model a replacement between
        # that check and the old path-based read_bytes() call.
        with patch(
            "isohyps.machine_index._resolve_freshness_reference_path",
            return_value=replacement,
        ):
            with self.assertRaises(MachineIndexContractError):
                resolve_machine_index_freshness(index_path)

    def test_machine_index_schema_and_runtime_reject_noncanonical_versions(self):
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "machine-index.schema.json"
        )
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)

        schema_pattern = re.compile(schema["properties"]["schema_version"]["pattern"])
        for version in ("2.01", "02.1"):
            with self.subTest(version=version):
                self.assertIsNone(schema_pattern.fullmatch(version))
                fixture = self._valid_machine_index_fixture()
                fixture["schema_version"] = version
                with self.assertRaises(MachineIndexContractError):
                    validate_machine_index(fixture)

    def test_machine_index_is_byte_stable_when_files_are_created_in_different_orders(self):
        logical_files = [
            (
                "src/root.py",
                "from src.zed import Zed\nfrom src.alpha import Alpha\n",
            ),
            ("src/zed.py", "class Zed: pass\n"),
            ("src/alpha.py", "class Alpha: pass\n"),
            ("README.md", "# Fixture\n"),
        ]
        outputs = []

        for suffix, creation_order in (
            ("forward", logical_files),
            ("reverse", list(reversed(logical_files))),
        ):
            root = self.test_dir / f"creation-order-{suffix}"
            for relative_path, content in creation_order:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            output = self.output_dir / f"creation-order-{suffix}"
            analyze_machine_level(root, output)
            outputs.append((output / "machine_index.json").read_bytes())

        self.assertEqual(outputs[0], outputs[1])

    def test_machine_scan_excludes_root_contained_output_and_rejects_same_root(self):
        root = self.test_dir / "contained-output-root"
        source = root / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_text("def app():\n    return 1\n", encoding="utf-8")

        contained_output = root / "analysis"
        (contained_output / "nested").mkdir(parents=True)
        (contained_output / "ghost.py").write_text("def ghost(): pass\n", encoding="utf-8")
        (contained_output / "nested" / "ghost.py").write_text(
            "def nested_ghost(): pass\n", encoding="utf-8"
        )

        first_result = analyze_machine_level(root, contained_output)
        second_result = analyze_machine_level(root, contained_output)
        for result in (first_result, second_result):
            paths = [file["path"] for file in result["files"]]
            self.assertEqual(paths, ["src/app.py"])
            self.assertNotIn("analysis/ghost.py", paths)
            self.assertNotIn("analysis/nested/ghost.py", paths)

            public_data = json.loads(
                (contained_output / "machine_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [file["path"] for file in public_data["files"]], ["src/app.py"]
            )

        same_root = self.test_dir / "same-root-output"
        same_root.mkdir()
        (same_root / "source.py").write_text("value = 1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            analyze_machine_level(same_root, same_root)
        self.assertFalse((same_root / "machine_analysis.json").exists())

    def test_machine_index_schema_matches_python_contract(self):
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "machine-index.schema.json"
        )
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)

        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertEqual(
            schema["required"], list(MACHINE_INDEX_TOP_LEVEL_FIELDS)
        )
        self.assertTrue(schema["additionalProperties"])

        top_level_types = {
            "schema_version": "string",
            "files": "array",
            "dependency_graph": "object",
            "dependency_order": "array",
        }
        for field, expected_type in top_level_types.items():
            with self.subTest(top_level_field=field):
                self.assertEqual(schema["properties"][field]["type"], expected_type)

        self.assertEqual(
            schema["properties"]["doc_freshness"]["$ref"],
            "#/$defs/docFreshnessReference",
        )
        freshness_reference = schema["$defs"]["docFreshnessReference"]
        self.assertEqual(
            freshness_reference["required"],
            ["path", "schema_version", "sha256", "counts"],
        )
        self.assertEqual(
            freshness_reference["properties"]["counts"]["required"],
            ["missing", "fresh", "stale", "unknown"],
        )

        file_schema = schema["$defs"]["fileEntry"]
        self.assertEqual(file_schema["required"], list(MACHINE_INDEX_FILE_FIELDS))
        self.assertTrue(file_schema["additionalProperties"])

        file_types = {
            "hash": "string",
            "size": "integer",
            "language": "string",
            "kind": "string",
            "public_symbols": "array",
            "internal_symbols": "array",
            "fan_in": "integer",
            "fan_out": "integer",
        }
        for field, expected_type in file_types.items():
            with self.subTest(file_field=field):
                self.assertEqual(
                    file_schema["properties"][field]["type"], expected_type
                )
        self.assertEqual(
            file_schema["properties"]["path"]["$ref"],
            "#/$defs/repositoryRelativePath",
        )

    def test_machine_index_schema_gates_module_summary_at_v22(self):
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "machine-index.schema.json"
        )
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)

        file_schema = schema["$defs"]["fileEntry"]
        self.assertIn("module_summary", file_schema["properties"])
        self.assertNotIn("module_summary", file_schema["required"])
        # The property remains an unconstrained additive field on legacy
        # versions; the root-level conditional below applies its definition
        # only to 2.2 and newer.
        self.assertNotIn("$ref", file_schema["properties"]["module_summary"])

        summary_pattern = r"^2\.(?:[2-9]|[1-9][0-9]+)$"
        summary_conditions = [
            condition
            for condition in schema["allOf"]
            if condition.get("if", {})
            .get("properties", {})
            .get("schema_version", {})
            .get("pattern")
            == summary_pattern
        ]
        self.assertEqual(len(summary_conditions), 1)
        summary_condition = summary_conditions[0]
        summary_items = (
            summary_condition["then"]["properties"]["files"]["items"]
        )
        self.assertEqual(summary_items["if"], {"required": ["module_summary"]})
        self.assertEqual(summary_items["then"]["properties"]["kind"]["const"], "source")
        self.assertEqual(
            summary_items["then"]["properties"]["module_summary"]["$ref"],
            "#/$defs/moduleSummary",
        )

        applies_to = ("2.2", "2.9", "2.10", "2.100")
        does_not_apply_to = ("1.0", "1.7", "2.0", "2.1")
        for version in applies_to:
            with self.subTest(version=version, applies=True):
                self.assertIsNotNone(re.fullmatch(summary_pattern, version))
        for version in does_not_apply_to:
            with self.subTest(version=version, applies=False):
                self.assertIsNone(re.fullmatch(summary_pattern, version))

        summary_schema = schema["$defs"]["moduleSummary"]
        self.assertEqual(summary_schema["required"], list(MODULE_SUMMARY_FIELDS))
        self.assertTrue(summary_schema["additionalProperties"])
        self.assertEqual(
            summary_schema["properties"]["text"]["maxLength"],
            MODULE_SUMMARY_MAX_TEXT_LENGTH,
        )
        self.assertEqual(
            summary_schema["properties"]["evidence"]["maxItems"],
            MODULE_SUMMARY_MAX_EVIDENCE,
        )
        self.assertEqual(
            summary_schema["properties"]["omitted_evidence_count"]["maximum"],
            MODULE_SUMMARY_MAX_INTEGER,
        )

        evidence_schema = schema["$defs"]["moduleSummaryEvidence"]
        self.assertEqual(
            evidence_schema["required"], list(MODULE_SUMMARY_EVIDENCE_FIELDS)
        )
        self.assertTrue(evidence_schema["additionalProperties"])
        self.assertEqual(
            evidence_schema["properties"]["value"]["maxLength"],
            MODULE_SUMMARY_MAX_EVIDENCE_VALUE_LENGTH,
        )
        self.assertEqual(
            evidence_schema["properties"]["line"]["maximum"],
            MODULE_SUMMARY_MAX_INTEGER,
        )

        # Cross-field relations which JSON Schema cannot express without
        # duplicating the runtime contract remain covered by
        # validate_machine_index / validate_module_summary tests.
        self.assertIn("allOf", summary_schema)

    def test_machine_index_minor_versions_and_unknown_fields_are_compatible(self):
        fixture = self._valid_machine_index_fixture()
        fixture["schema_version"] = "1.7"
        fixture["future_top_level"] = {"owner": "downstream"}
        fixture["files"][1]["future_file_field"] = ["kept for a newer reader"]

        index_path = self.output_dir / "machine-index-v1-minor.json"
        index_path.write_text(serialize_machine_index(fixture), encoding="utf-8")
        loaded = load_machine_index(index_path)

        self.assertEqual(loaded["schema_version"], "1.7")
        self.assertEqual(loaded["future_top_level"], {"owner": "downstream"})
        self.assertEqual(
            loaded["files"][1]["future_file_field"], ["kept for a newer reader"]
        )
        self.assertEqual(
            {field: loaded[field] for field in MACHINE_INDEX_TOP_LEVEL_FIELDS},
            {field: fixture[field] for field in MACHINE_INDEX_TOP_LEVEL_FIELDS},
        )
        for loaded_entry, expected_entry in zip(loaded["files"], fixture["files"]):
            self.assertEqual(
                {field: loaded_entry[field] for field in MACHINE_INDEX_FILE_FIELDS},
                {field: expected_entry[field] for field in MACHINE_INDEX_FILE_FIELDS},
            )

    def test_machine_index_rejects_invalid_versions_required_fields_and_types(self):
        fixture = self._valid_machine_index_fixture()
        invalid_cases = []

        malformed_version = deepcopy(fixture)
        malformed_version["schema_version"] = "1"
        invalid_cases.append(("malformed version", malformed_version, "schema_version"))

        non_string_version = deepcopy(fixture)
        non_string_version["schema_version"] = 1.0
        invalid_cases.append(("version type", non_string_version, "schema_version"))

        unknown_major = deepcopy(fixture)
        unknown_major["schema_version"] = "2.0"
        invalid_cases.append(("unknown major", unknown_major, "schema_version"))

        missing_top_level = deepcopy(fixture)
        del missing_top_level["files"]
        invalid_cases.append(("missing top-level field", missing_top_level, "root.files"))

        missing_file_field = deepcopy(fixture)
        del missing_file_field["files"][1]["hash"]
        invalid_cases.append(("missing file field", missing_file_field, "files[1].hash"))

        wrong_file_type = deepcopy(fixture)
        wrong_file_type["files"][1]["size"] = True
        invalid_cases.append(("file field type", wrong_file_type, "files[1].size"))

        wrong_nested_type = deepcopy(fixture)
        wrong_nested_type["files"][1]["public_symbols"] = ["App", 3]
        invalid_cases.append(
            ("symbol item type", wrong_nested_type, "files[1].public_symbols[1]")
        )

        wrong_graph_type = deepcopy(fixture)
        wrong_graph_type["dependency_graph"] = []
        invalid_cases.append(("graph type", wrong_graph_type, "dependency_graph"))

        for label, data, location in invalid_cases:
            with self.subTest(case=label):
                with self.assertRaises(MachineIndexContractError) as context:
                    validate_machine_index(data)
                self.assertIn(location, str(context.exception))

    def test_machine_index_rejects_invalid_and_duplicate_paths(self):
        invalid_paths = [
            ("absolute path", "/README.md", "files[0].path"),
            ("traversal path", "../README.md", "files[0].path"),
            ("empty path segment", "src//app.py", "files[1].path"),
            ("windows path", r"src\\app.py", "files[1].path"),
        ]

        for label, path, location in invalid_paths:
            with self.subTest(case=label):
                data = self._valid_machine_index_fixture()
                target_index = 0 if location == "files[0].path" else 1
                data["files"][target_index]["path"] = path
                with self.assertRaises(MachineIndexContractError) as context:
                    validate_machine_index(data)
                self.assertIn(location, str(context.exception))

        duplicate_path = self._valid_machine_index_fixture()
        duplicate_path["files"][1]["path"] = duplicate_path["files"][0]["path"]
        with self.assertRaises(MachineIndexContractError) as context:
            validate_machine_index(duplicate_path)
        self.assertIn("files[1].path", str(context.exception))

        invalid_dependency_path = self._valid_machine_index_fixture()
        invalid_dependency_path["dependency_graph"]["src/app.py"] = [
            "../src/config.py"
        ]
        with self.assertRaises(MachineIndexContractError) as context:
            validate_machine_index(invalid_dependency_path)
        self.assertIn("dependency_graph['src/app.py'][0]", str(context.exception))

    def test_machine_index_rejects_graph_order_and_fan_invariant_violations(self):
        invalid_cases = []

        missing_graph_key = self._valid_machine_index_fixture()
        del missing_graph_key["dependency_graph"]["src/config.py"]
        invalid_cases.append(("missing graph key", missing_graph_key, "dependency_graph"))

        unknown_dependency = self._valid_machine_index_fixture()
        unknown_dependency["dependency_graph"]["src/app.py"] = ["src/missing.py"]
        invalid_cases.append(
            (
                "unknown dependency",
                unknown_dependency,
                "dependency_graph['src/app.py'][0]",
            )
        )

        wrong_order = self._valid_machine_index_fixture()
        wrong_order["dependency_order"] = ["src/app.py", "src/config.py"]
        invalid_cases.append(("dependency order", wrong_order, "dependency_order"))

        wrong_fan_out = self._valid_machine_index_fixture()
        wrong_fan_out["files"][1]["fan_out"] = 0
        invalid_cases.append(("fan out", wrong_fan_out, "files[1].fan_out"))

        wrong_fan_in = self._valid_machine_index_fixture()
        wrong_fan_in["files"][2]["fan_in"] = 0
        invalid_cases.append(("fan in", wrong_fan_in, "files[2].fan_in"))

        cycle_with_wrong_fallback = {
            "schema_version": "1.0",
            "files": [
                {
                    "path": path,
                    "hash": "a" * 64,
                    "size": 1,
                    "language": "python",
                    "kind": "source",
                    "public_symbols": [],
                    "internal_symbols": [],
                    "fan_in": 1,
                    "fan_out": 1,
                }
                for path in ("a.py", "b.py")
            ],
            "dependency_graph": {"a.py": ["b.py"], "b.py": ["a.py"]},
            "dependency_order": ["b.py", "a.py"],
        }
        invalid_cases.append(
            ("cycle fallback order", cycle_with_wrong_fallback, "dependency_order")
        )

        for label, data, location in invalid_cases:
            with self.subTest(case=label):
                with self.assertRaises(MachineIndexContractError) as context:
                    validate_machine_index(data)
                self.assertIn(location, str(context.exception))

    def test_machine_index_projection_is_allowlisted_and_does_not_mutate_analysis(self):
        analysis = self._valid_machine_index_fixture()
        analysis.pop("schema_version")
        analysis["attention"] = ["history-dependent diagnostic"]
        analysis["coverage"] = {"stale_docs": 1}
        analysis["files"][1]["status"] = "added"
        analysis["files"][1]["mtime"] = 123.0
        original_analysis = deepcopy(analysis)

        projected = build_machine_index_v1(analysis)

        self.assertEqual(analysis, original_analysis)
        self.assertEqual(set(projected), set(MACHINE_INDEX_TOP_LEVEL_FIELDS))
        for entry in projected["files"]:
            self.assertEqual(set(entry), set(MACHINE_INDEX_FILE_FIELDS))
        self.assertNotIn("attention", projected)
        self.assertNotIn("coverage", projected)
        self.assertNotIn("status", projected["files"][1])
        self.assertEqual(projected["schema_version"], MACHINE_INDEX_SCHEMA_VERSION)

    def test_machine_index_v22_projects_bounded_source_summaries_only(self):
        analysis = self._valid_machine_index_fixture()
        analysis["attention"] = []
        analysis["doc_freshness"] = {
            "path": "doc_freshness.json",
            "schema_version": "1.0",
            "sha256": "d" * 64,
            "counts": {"missing": 0, "fresh": 0, "stale": 0, "unknown": 0},
        }
        summary = TestModuleSummaryContract._docstring_summary()
        summary["future_summary_field"] = {"internal": True}
        summary["evidence"][0]["future_evidence_field"] = "ignored"
        analysis["files"][1]["module_summary"] = summary
        analysis["files"][0]["module_summary"] = {"internal": "not public"}
        original_analysis = deepcopy(analysis)

        legacy_projection = build_machine_index_v1(analysis)
        projected = build_machine_index_v2(analysis)

        self.assertEqual(analysis, original_analysis)
        self.assertEqual(projected["schema_version"], MACHINE_INDEX_V2_SCHEMA_VERSION)
        self.assertNotIn("module_summary", legacy_projection["files"][1])

        by_path = {entry["path"]: entry for entry in projected["files"]}
        app_entry = by_path["src/app.py"]
        self.assertEqual(
            set(app_entry), set(MACHINE_INDEX_FILE_FIELDS) | {"module_summary"}
        )
        self.assertEqual(
            app_entry["module_summary"]["text"], "Read contour records."
        )
        self.assertNotIn(
            "future_summary_field", app_entry["module_summary"]
        )
        self.assertNotIn(
            "future_evidence_field", app_entry["module_summary"]["evidence"][0]
        )
        self.assertNotIn("module_summary", by_path["README.md"])
        self.assertNotIn("module_summary", by_path["src/config.py"])
        validate_machine_index(projected, supported_major=2)

    def test_machine_index_summary_validation_starts_at_v22_and_loads_legacy_versions(self):
        invalid_summary = {"text": "not a module summary"}
        versions = ("1.0", "2.0", "2.1", "2.2", "2.9", "2.10", "2.100")

        for version in versions:
            with self.subTest(version=version):
                fixture = self._valid_machine_index_fixture()
                major, minor = (int(part) for part in version.split("."))
                fixture["schema_version"] = version
                if major == 2:
                    fixture["attention"] = []
                if major == 2 and minor >= 1:
                    fixture["doc_freshness"] = {
                        "path": "doc_freshness.json",
                        "schema_version": "1.0",
                        "sha256": "e" * 64,
                        "counts": {
                            "missing": 0,
                            "fresh": 0,
                            "stale": 0,
                            "unknown": 0,
                        },
                    }
                validate_machine_index(fixture, supported_major=major)

                fixture["files"][1]["module_summary"] = deepcopy(invalid_summary)
                if major == 2 and minor >= 2:
                    with self.assertRaises(MachineIndexContractError) as context:
                        validate_machine_index(fixture, supported_major=major)
                    self.assertIn("files[1].module_summary", str(context.exception))
                else:
                    validate_machine_index(fixture, supported_major=major)

        legacy = self._valid_machine_index_fixture()
        legacy["schema_version"] = "2.1"
        legacy["attention"] = []
        legacy["doc_freshness"] = {
            "path": "doc_freshness.json",
            "schema_version": "1.0",
            "sha256": "f" * 64,
            "counts": {"missing": 0, "fresh": 0, "stale": 0, "unknown": 0},
        }
        legacy["files"][1]["module_summary"] = deepcopy(invalid_summary)
        index_path = self.output_dir / "legacy-with-unknown-summary.json"
        index_path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = load_machine_index(index_path, supported_major=2)
        self.assertEqual(loaded["files"][1]["module_summary"], invalid_summary)

    def test_machine_index_v22_rejects_module_summary_on_non_source_entries(self):
        fixture = self._valid_machine_index_fixture()
        fixture["schema_version"] = "2.2"
        fixture["attention"] = []
        fixture["doc_freshness"] = {
            "path": "doc_freshness.json",
            "schema_version": "1.0",
            "sha256": "f" * 64,
            "counts": {"missing": 0, "fresh": 0, "stale": 0, "unknown": 0},
        }
        fixture["files"][0]["module_summary"] = TestModuleSummaryContract._docstring_summary()

        with self.assertRaises(MachineIndexContractError) as context:
            validate_machine_index(fixture, supported_major=2)
        self.assertIn("files[0].module_summary", str(context.exception))


class TestAttentionSnapshotBuilder(unittest.TestCase):
    @staticmethod
    def metadata(path: str, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "path": path,
            "hash": "a" * 64,
            "size": 100,
            "language": "python",
            "kind": "source",
            "status": "changed",
            "todo_count": 0,
            "line_count": 10,
            "readable": True,
            "fan_in": 0,
            "fan_out": 0,
            "mtime": 100,
        }
        values.update(overrides)
        return values

    def test_builds_one_sorted_snapshot_with_all_attention_signals(self):
        files_meta = [
            self.metadata(
                "src/app.py",
                line_count=301,
                todo_count=2,
                fan_in=5,
                fan_out=15,
            ),
            self.metadata("src/untested.py", size=100),
            self.metadata("tests/test_app.py", kind="test", language="python"),
            self.metadata("README.md", kind="doc", language="unknown"),
        ]

        snapshots, diagnostics = build_attention_snapshots(
            None,
            list(reversed(files_meta)),
            {"src/app.py": {"todo_count": 1}},
            doc_status_by_path={
                "src/app.py": "missing",
                "src/untested.py": "stale",
            },
            entrypoint_paths={"src/app.py"},
        )

        self.assertEqual(diagnostics, [])
        self.assertEqual([snapshot.path for snapshot in snapshots], [
            "README.md",
            "src/app.py",
            "src/untested.py",
            "tests/test_app.py",
        ])
        by_path = {snapshot.path: snapshot for snapshot in snapshots}

        app = by_path["src/app.py"]
        self.assertEqual(app.line_count, 301)
        self.assertEqual(app.fan_in, 5)
        self.assertEqual(app.fan_out, 15)
        self.assertFalse(app.test_missing)
        self.assertEqual(app.todo_current, 2)
        self.assertEqual(app.todo_previous, 1)
        self.assertTrue(app.todo_increased)
        self.assertTrue(app.is_entrypoint)
        self.assertEqual(app.doc_status, "missing")

        untested = by_path["src/untested.py"]
        self.assertTrue(untested.test_missing)
        self.assertEqual(untested.doc_status, "stale")

    def test_only_kind_test_paths_are_used_for_test_lookup(self):
        source_files = [
            self.metadata("src/widget.py"),
            # A source-shaped filename must not count as a test candidate.
            self.metadata("src/test_widget.py"),
        ]
        snapshots, diagnostics = build_attention_snapshots(None, source_files)
        self.assertEqual(diagnostics, [])
        self.assertTrue(next(s for s in snapshots if s.path == "src/widget.py").test_missing)

        test_files = source_files + [
            self.metadata("tests/test_widget.py", kind="test"),
        ]
        snapshots, diagnostics = build_attention_snapshots(None, test_files)
        self.assertEqual(diagnostics, [])
        self.assertFalse(next(s for s in snapshots if s.path == "src/widget.py").test_missing)

    def test_todo_previous_count_and_first_snapshot_boundaries(self):
        files_meta = [
            self.metadata("src/increased.py", todo_count=2),
            self.metadata("src/unchanged.py", todo_count=1),
            self.metadata("src/decreased.py", todo_count=1),
            self.metadata("src/first.py", todo_count=1),
            self.metadata("src/empty-first.py", todo_count=0),
        ]
        previous = {
            "src/increased.py": {"todo_count": 1},
            "src/unchanged.py": {"todo_count": 1},
            "src/decreased.py": {"todo_count": 2},
        }

        snapshots, diagnostics = build_attention_snapshots(None, files_meta, previous)
        self.assertEqual(diagnostics, [])
        by_path = {snapshot.path: snapshot for snapshot in snapshots}
        self.assertTrue(by_path["src/increased.py"].todo_increased)
        self.assertFalse(by_path["src/unchanged.py"].todo_increased)
        self.assertFalse(by_path["src/decreased.py"].todo_increased)
        self.assertTrue(by_path["src/first.py"].todo_increased)
        self.assertFalse(by_path["src/empty-first.py"].todo_increased)
        self.assertIsNone(by_path["src/first.py"].todo_previous)

    def test_unavailable_previous_todo_baselines_do_not_trigger_first_snapshot(self):
        files_meta = [
            self.metadata("src/invalid-previous.py", todo_count=1, size=40),
            self.metadata("src/missing-previous.py", todo_count=1, size=40),
        ]
        previous = {
            "src/invalid-previous.py": {"todo_count": "bad"},
            "src/missing-previous.py": {},
        }

        snapshots, diagnostics = build_attention_snapshots(None, files_meta, previous)
        by_path = {snapshot.path: snapshot for snapshot in snapshots}

        self.assertFalse(by_path["src/invalid-previous.py"].todo_increased)
        self.assertFalse(by_path["src/missing-previous.py"].todo_increased)
        self.assertEqual(
            {(diagnostic.detector, diagnostic.code, diagnostic.path) for diagnostic in diagnostics},
            {
                ("metadata", "invalid_todo_count", "src/invalid-previous.py"),
                (
                    "metadata",
                    "previous_todo_count_unavailable",
                    "src/missing-previous.py",
                ),
            },
        )
        self.assertEqual(classify_attention(snapshots), [])

    def test_missing_graph_metrics_are_diagnostics_and_valid_zero_is_preserved(self):
        missing_fan_in = self.metadata("src/missing-in.py", size=40)
        missing_fan_in.pop("fan_in")
        null_fan_out = self.metadata("src/null-out.py", size=40, fan_out=None)
        valid_zero = self.metadata("src/zero.py", size=40, fan_in=0, fan_out=0)

        snapshots, diagnostics = build_attention_snapshots(
            None, [missing_fan_in, null_fan_out, valid_zero]
        )
        by_path = {snapshot.path: snapshot for snapshot in snapshots}

        self.assertEqual(set(by_path), {"src/zero.py"})
        self.assertEqual(by_path["src/zero.py"].fan_in, 0)
        self.assertEqual(by_path["src/zero.py"].fan_out, 0)
        self.assertEqual(
            {(diagnostic.detector, diagnostic.code, diagnostic.path) for diagnostic in diagnostics},
            {
                ("metadata", "fan_in_unavailable", "src/missing-in.py"),
                ("metadata", "fan_out_unavailable", "src/null-out.py"),
            },
        )

    def test_doc_status_snapshot_uses_freshness_artifact_not_mtime(self):
        root = Path(tempfile.mkdtemp())
        output = Path(tempfile.mkdtemp())
        try:
            files_meta = [
                self.metadata("src/exact.py", mtime=100),
                self.metadata("src/stale.py", mtime=100),
                self.metadata("src/missing.py", mtime=100),
                self.metadata("src/unchanged.py", mtime=100, status="unchanged"),
            ]
            for path, mtime in (
                ("src/exact.py.md", 98.0),
                ("src/stale.py.md", 97.99),
                ("src/unchanged.py.md", 0.0),
            ):
                doc = output / path
                doc.parent.mkdir(parents=True, exist_ok=True)
                doc.write_text("# doc", encoding="utf-8")
                os.utime(doc, (mtime, mtime))

            def entry(
                path: str,
                *,
                doc_path: str | None,
                current: str,
                recorded: str | None,
                doc_hash: str | None,
                status: str,
                reason: str,
            ) -> dict[str, object]:
                return {
                    "source_path": path,
                    "doc_path": doc_path,
                    "expected_doc_paths": list(expected_doc_paths(path)),
                    "current_source_hash": current,
                    "recorded_source_hash": recorded,
                    "doc_hash": doc_hash,
                    "status": status,
                    "reason": reason,
                    "diagnostics": [],
                }

            hash_a = "a" * 64
            hash_b = "b" * 64
            hash_c = "c" * 64
            freshness_document = {
                "schema_version": DOC_FRESHNESS_SCHEMA_VERSION,
                "hash_algorithm": "sha256",
                "mapping_rule": DOC_FRESHNESS_MAPPING_RULE,
                "counts": {"missing": 1, "fresh": 2, "stale": 1, "unknown": 0},
                "entries": [
                    entry(
                        "src/exact.py",
                        doc_path="src/exact.py.md",
                        current=hash_a,
                        recorded=hash_a,
                        doc_hash=hash_c,
                        status="fresh",
                        reason="source_hash_match",
                    ),
                    entry(
                        "src/missing.py",
                        doc_path=None,
                        current=hash_a,
                        recorded=None,
                        doc_hash=None,
                        status="missing",
                        reason="doc_missing",
                    ),
                    entry(
                        "src/stale.py",
                        doc_path="src/stale.py.md",
                        current=hash_a,
                        recorded=hash_b,
                        doc_hash=hash_c,
                        status="stale",
                        reason="source_hash_mismatch",
                    ),
                    entry(
                        "src/unchanged.py",
                        doc_path="src/unchanged.py.md",
                        current=hash_a,
                        recorded=hash_a,
                        doc_hash=hash_c,
                        status="fresh",
                        reason="source_hash_match",
                    ),
                ],
                "diagnostics": [],
            }
            write_doc_freshness_atomic(output / "doc_freshness.json", freshness_document)

            statuses, diagnostics = build_doc_status_snapshot(root, files_meta, output)
            self.assertEqual(diagnostics, [])
            self.assertEqual(statuses["src/exact.py"], "current")
            self.assertEqual(statuses["src/stale.py"], "stale")
            self.assertEqual(statuses["src/missing.py"], "missing")
            self.assertEqual(statuses["src/unchanged.py"], "current")

            snapshots, diagnostics = build_attention_snapshots(
                root, files_meta, output_dir=output
            )
            self.assertEqual(diagnostics, [])
            by_path = {snapshot.path: snapshot for snapshot in snapshots}
            self.assertEqual(by_path["src/exact.py"].doc_status, "current")
            self.assertEqual(by_path["src/stale.py"].doc_status, "stale")
            self.assertEqual(by_path["src/missing.py"].doc_status, "missing")
            self.assertEqual(by_path["src/unchanged.py"].doc_status, "current")
        finally:
            shutil.rmtree(root)
            shutil.rmtree(output)

    def test_doc_status_snapshot_fails_closed_for_omitted_coverage_target(self):
        root = Path(tempfile.mkdtemp())
        output = Path(tempfile.mkdtemp())
        try:
            files_meta = [
                self.metadata("src/included.py"),
                self.metadata("src/omitted.py"),
                self.metadata("README.md", kind="doc", language="unknown"),
            ]
            freshness_document = {
                "schema_version": DOC_FRESHNESS_SCHEMA_VERSION,
                "hash_algorithm": "sha256",
                "mapping_rule": DOC_FRESHNESS_MAPPING_RULE,
                "counts": {"missing": 0, "fresh": 1, "stale": 0, "unknown": 0},
                "entries": [
                    {
                        "source_path": "src/included.py",
                        "doc_path": "src/included.py.md",
                        "expected_doc_paths": list(
                            expected_doc_paths("src/included.py")
                        ),
                        "current_source_hash": "a" * 64,
                        "recorded_source_hash": "a" * 64,
                        "doc_hash": "b" * 64,
                        "status": "fresh",
                        "reason": "source_hash_match",
                        "diagnostics": [],
                    }
                ],
                "diagnostics": [],
            }
            write_doc_freshness_atomic(
                output / "doc_freshness.json", freshness_document
            )

            statuses, diagnostics = build_doc_status_snapshot(
                root, files_meta, output
            )

            self.assertEqual(statuses["src/included.py"], "current")
            self.assertEqual(statuses["src/omitted.py"], "unavailable")
            self.assertEqual(statuses["README.md"], "current")
            self.assertIn(
                ("doc_status", "freshness_target_missing", "src/omitted.py"),
                {
                    (item.detector, item.code, item.path) for item in diagnostics
                },
            )
        finally:
            shutil.rmtree(root)
            shutil.rmtree(output)

    def test_unavailable_metadata_and_doc_status_are_diagnostics_not_findings(self):
        unavailable = self.metadata(
            "src/unavailable.py",
            hash="error",
            readable=False,
            line_count=None,
            todo_count=None,
            size=40,
            doc_status="unavailable",
        )
        snapshots, diagnostics = build_attention_snapshots(
            None,
            [unavailable],
            doc_status_by_path={"src/unavailable.py": "unavailable"},
        )

        self.assertEqual(len(snapshots), 1)
        self.assertFalse(snapshots[0].readable)
        self.assertIsNone(snapshots[0].line_count)
        self.assertEqual(
            {(diagnostic.detector, diagnostic.code, diagnostic.path) for diagnostic in diagnostics},
            {
                ("metadata", "read_unavailable", "src/unavailable.py"),
                ("doc_status", "status_unavailable", "src/unavailable.py"),
            },
        )
        self.assertEqual(classify_attention(snapshots), [])

    def test_resolves_structured_python_and_node_entrypoints(self):
        root = Path(tempfile.mkdtemp())
        try:
            (root / "pyproject.toml").write_text(
                '[project.scripts]\ncli = "pkg.cli:main"\n', encoding="utf-8"
            )
            (root / "package.json").write_text(
                json.dumps({"bin": {"tool": "bin/tool.js"}, "scripts": {"fake": "src/fake.js"}}),
                encoding="utf-8",
            )
            metadata = [
                self.metadata("src/pkg/cli.py"),
                self.metadata("bin/tool.js", language="javascript"),
                self.metadata("src/fake.js", language="javascript"),
            ]
            paths, diagnostics = resolve_attention_entrypoints(root, metadata)
            self.assertEqual(paths, {"src/pkg/cli.py", "bin/tool.js"})
            self.assertEqual(diagnostics, [])
        finally:
            shutil.rmtree(root)


class TestAttentionClassifier(unittest.TestCase):
    @staticmethod
    def snapshot(path: str, **overrides: object) -> AttentionSignalSnapshot:
        values: dict[str, object] = {
            "path": path,
            "kind": "source",
            "language": "python",
            "readable": True,
            "line_count": 10,
            "fan_in": 0,
            "fan_out": 0,
            "test_missing": False,
            "todo_current": 0,
            "todo_previous": None,
            "todo_increased": False,
            "is_entrypoint": False,
            "doc_status": "current",
        }
        values.update(overrides)
        return AttentionSignalSnapshot(**values)

    @staticmethod
    def entry_by_identity(entries: list[dict[str, object]], path: str, kind: str):
        return next(entry for entry in entries if entry["path"] == path and entry["kind"] == kind)

    def test_classifier_emits_all_kinds_and_severity_levels(self):
        snapshots = [
            self.snapshot("src/critical_hub.py", line_count=301, fan_in=5),
            self.snapshot(
                "src/critical_cli.py",
                is_entrypoint=True,
                doc_status="missing",
            ),
            self.snapshot("src/high_hub.py", fan_in=5),
            self.snapshot("src/high_cli.py", is_entrypoint=True, doc_status="stale"),
            self.snapshot("src/testless.py", test_missing=True),
            self.snapshot("src/wide.py", fan_out=15),
            self.snapshot("src/undocumented.py", doc_status="missing"),
            self.snapshot("src/stale.py", doc_status="stale"),
            self.snapshot(
                "src/todo.py",
                todo_current=1,
                todo_previous=None,
                todo_increased=True,
            ),
        ]

        entries = classify_attention(snapshots)

        self.assertEqual({entry["kind"] for entry in entries}, set(ATTENTION_KINDS))
        self.assertEqual(
            {entry["severity"] for entry in entries},
            {"critical", "high", "medium", "low"},
        )
        for entry in entries:
            self.assertEqual(set(entry), set(ATTENTION_ENTRY_FIELDS))
            self.assertTrue(entry["reason"])
            self.assertIsInstance(entry["evidence"], dict)

        self.assertEqual(
            self.entry_by_identity(entries, "src/critical_hub.py", "large_file")["severity"],
            "critical",
        )
        self.assertEqual(
            self.entry_by_identity(entries, "src/critical_hub.py", "high_fan_in")["severity"],
            "critical",
        )
        self.assertEqual(
            self.entry_by_identity(entries, "src/critical_cli.py", "doc_missing")["severity"],
            "critical",
        )
        self.assertEqual(
            self.entry_by_identity(entries, "src/high_hub.py", "high_fan_in")["severity"],
            "high",
        )
        self.assertEqual(
            self.entry_by_identity(entries, "src/high_cli.py", "doc_stale")["severity"],
            "high",
        )
        for path, kind in (
            ("src/testless.py", "test_missing"),
            ("src/wide.py", "high_fan_out"),
            ("src/undocumented.py", "doc_missing"),
            ("src/stale.py", "doc_stale"),
        ):
            with self.subTest(path=path, kind=kind):
                self.assertEqual(self.entry_by_identity(entries, path, kind)["severity"], "medium")
        self.assertEqual(
            self.entry_by_identity(entries, "src/todo.py", "todo_increase")["severity"],
            "low",
        )

    def test_classifier_enforces_threshold_boundaries_and_non_promotions(self):
        entries = classify_attention(
            [
                self.snapshot("boundary/fan-in-4.py", fan_in=4),
                self.snapshot("boundary/fan-in-5.py", fan_in=5),
                self.snapshot("boundary/fan-out-14.py", fan_out=14),
                self.snapshot("boundary/fan-out-15.py", fan_out=15),
                self.snapshot("boundary/lines-300.py", line_count=300),
                self.snapshot("boundary/lines-301.py", line_count=301),
                self.snapshot("single/missing.py", doc_status="missing"),
                self.snapshot("single/stale.py", doc_status="stale"),
                self.snapshot("single/large.py", line_count=301),
                self.snapshot("single/testless.py", test_missing=True),
            ]
        )
        found = {(entry["path"], entry["kind"]): entry for entry in entries}

        self.assertNotIn(("boundary/fan-in-4.py", "high_fan_in"), found)
        self.assertEqual(found[("boundary/fan-in-5.py", "high_fan_in")]["severity"], "high")
        self.assertNotIn(("boundary/fan-out-14.py", "high_fan_out"), found)
        self.assertEqual(found[("boundary/fan-out-15.py", "high_fan_out")]["severity"], "medium")
        self.assertNotIn(("boundary/lines-300.py", "large_file"), found)
        self.assertEqual(found[("boundary/lines-301.py", "large_file")]["severity"], "medium")
        self.assertEqual(found[("single/missing.py", "doc_missing")]["severity"], "medium")
        self.assertEqual(found[("single/stale.py", "doc_stale")]["severity"], "medium")
        self.assertEqual(found[("single/large.py", "large_file")]["severity"], "medium")
        self.assertEqual(found[("single/testless.py", "test_missing")]["severity"], "medium")

    def test_classifier_applies_compound_promotions_without_overpromoting(self):
        entries = classify_attention(
            [
                self.snapshot("compound/fan-in-doc.py", fan_in=5, doc_status="missing"),
                self.snapshot(
                    "compound/entrypoint-doc.py",
                    fan_in=5,
                    is_entrypoint=True,
                    doc_status="missing",
                ),
                self.snapshot(
                    "compound/entrypoint-stale.py",
                    fan_in=5,
                    is_entrypoint=True,
                    doc_status="stale",
                ),
            ]
        )
        found = {(entry["path"], entry["kind"]): entry for entry in entries}

        self.assertEqual(
            found[("compound/fan-in-doc.py", "high_fan_in")]["severity"], "high"
        )
        self.assertEqual(
            found[("compound/fan-in-doc.py", "doc_missing")]["severity"], "high"
        )
        self.assertEqual(
            found[("compound/entrypoint-doc.py", "high_fan_in")]["severity"], "high"
        )
        self.assertEqual(
            found[("compound/entrypoint-doc.py", "doc_missing")]["severity"], "critical"
        )
        self.assertEqual(
            found[("compound/entrypoint-stale.py", "high_fan_in")]["severity"], "high"
        )
        self.assertEqual(
            found[("compound/entrypoint-stale.py", "doc_stale")]["severity"], "high"
        )
        self.assertNotEqual(
            found[("compound/fan-in-doc.py", "doc_missing")]["severity"], "critical"
        )

    def test_classifier_is_deterministic_for_permutations_and_canonical_bytes(self):
        snapshots = [
            self.snapshot("z/low.py", todo_current=2, todo_previous=1, todo_increased=True),
            self.snapshot("a/critical.py", fan_in=5, line_count=301),
            self.snapshot("m/medium.py", fan_out=15),
            self.snapshot("b/high.py", fan_in=5),
        ]
        variants = (snapshots, list(reversed(snapshots)), snapshots[2:] + snapshots[:2])
        expected = classify_attention(variants[0])
        expected_bytes = canonical_attention_bytes(expected)

        for variant in variants:
            with self.subTest(order=[snapshot.path for snapshot in variant]):
                actual = classify_attention(variant)
                self.assertEqual(actual, expected)
                self.assertEqual(canonical_attention_bytes(actual), expected_bytes)
                self.assertEqual(serialize_attention(actual).encode("utf-8"), expected_bytes)

        self.assertEqual(
            [(entry["severity"], entry["path"], entry["kind"]) for entry in expected],
            [
                ("critical", "a/critical.py", "high_fan_in"),
                ("critical", "a/critical.py", "large_file"),
                ("high", "b/high.py", "high_fan_in"),
                ("medium", "m/medium.py", "high_fan_out"),
                ("low", "z/low.py", "todo_increase"),
            ],
        )

    def test_classifier_deduplicates_identical_snapshots_and_rejects_conflicts(self):
        snapshot = self.snapshot("src/duplicate.py", fan_in=5)
        self.assertEqual(classify_attention([snapshot, snapshot]), classify_attention([snapshot]))

        with self.assertRaises(AttentionContractError):
            classify_attention(
                [
                    self.snapshot("src/conflict.py", fan_in=5),
                    self.snapshot("src/conflict.py", fan_in=6),
                ]
            )

    def test_attention_entry_validator_rejects_noncanonical_or_invalid_shape(self):
        entries = classify_attention([self.snapshot("src/a.py", fan_in=5), self.snapshot("src/b.py", fan_out=15)])
        validate_attention(entries)

        with self.assertRaises(AttentionContractError):
            validate_attention(list(reversed(entries)))

        invalid = [dict(entry) for entry in entries]
        invalid[0]["evidence"] = dict(invalid[0]["evidence"])
        invalid[0]["evidence"]["fan_in"] = True
        with self.assertRaises(AttentionContractError):
            validate_attention(invalid)

if __name__ == "__main__":
    unittest.main()
