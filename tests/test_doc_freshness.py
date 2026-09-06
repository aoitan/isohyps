import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from isohyps.doc_freshness import (
    CoverageTarget,
    DIAGNOSTIC_CODES,
    DOC_FRESHNESS_MAPPING_RULE,
    DOC_FRESHNESS_SCHEMA_MAJOR,
    DOC_FRESHNESS_SCHEMA_VERSION,
    DOC_PROVENANCE_PRODUCER,
    DOC_PROVENANCE_PRODUCER_VERSION,
    DOC_PROVENANCE_SCHEMA_MAJOR,
    DOC_PROVENANCE_SCHEMA_VERSION,
    DocFreshnessContractError,
    DocProvenanceContractError,
    DocProvenanceRecordingError,
    ProvenanceFailure,
    HashObservation,
    canonical_doc_freshness_bytes,
    canonical_doc_provenance_bytes,
    evaluate_doc_freshness,
    expected_doc_paths,
    FRESHNESS_REASONS,
    load_doc_freshness,
    load_doc_provenance,
    project_doc_statuses,
    project_freshness,
    record_doc_provenance,
    read_doc_provenance,
    serialize_doc_freshness,
    serialize_doc_provenance,
    resolve_doc_mappings,
    safe_hash_regular_file,
    validate_doc_freshness,
    validate_doc_provenance,
    write_text_regular_file,
    write_doc_freshness_atomic,
    write_doc_provenance_atomic,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def freshness_entry(
    source_path: str,
    *,
    doc_path: str | None,
    current_source_hash: str | None,
    recorded_source_hash: str | None,
    doc_hash: str | None,
    status: str,
    reason: str,
) -> dict[str, object]:
    return {
        "source_path": source_path,
        "doc_path": doc_path,
        "expected_doc_paths": list(expected_doc_paths(source_path)),
        "current_source_hash": current_source_hash,
        "recorded_source_hash": recorded_source_hash,
        "doc_hash": doc_hash,
        "status": status,
        "reason": reason,
        "diagnostics": [],
    }


def valid_freshness_document() -> dict[str, object]:
    return {
        "schema_version": DOC_FRESHNESS_SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "mapping_rule": DOC_FRESHNESS_MAPPING_RULE,
        "counts": {"missing": 1, "fresh": 1, "stale": 0, "unknown": 0},
        "entries": [
            freshness_entry(
                "src/app.py",
                doc_path="src/app.py.md",
                current_source_hash=HASH_A,
                recorded_source_hash=HASH_A,
                doc_hash=HASH_B,
                status="fresh",
                reason="source_hash_match",
            ),
            freshness_entry(
                "src/readme_generator.py",
                doc_path=None,
                current_source_hash=HASH_C,
                recorded_source_hash=None,
                doc_hash=None,
                status="missing",
                reason="doc_missing",
            ),
        ],
        "diagnostics": [],
    }


def valid_provenance_document() -> dict[str, object]:
    return {
        "schema_version": DOC_PROVENANCE_SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "assertions": [
            {
                "source_path": "src/app.py",
                "doc_path": "src/app.py.md",
                "recorded_source_hash": HASH_A,
                "recorded_doc_hash": HASH_B,
                "producer": "isohyps.project_analysis",
                "producer_version": "1",
            },
            {
                "source_path": "src/readme_generator.py",
                "doc_path": "src/readme_generator.py.md",
                "recorded_source_hash": HASH_C,
                "recorded_doc_hash": HASH_A,
                "producer": "isohyps.project_analysis",
                "producer_version": "1",
            },
        ],
    }


class TestDocFreshnessContract(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_valid_documents_round_trip_with_canonical_bytes(self) -> None:
        freshness = valid_freshness_document()
        provenance = valid_provenance_document()

        freshness_text = serialize_doc_freshness(freshness)
        provenance_text = serialize_doc_provenance(provenance)
        self.assertTrue(freshness_text.endswith("\n"))
        self.assertTrue(provenance_text.endswith("\n"))
        self.assertEqual(
            freshness_text.encode("utf-8"), canonical_doc_freshness_bytes(freshness)
        )
        self.assertEqual(
            provenance_text.encode("utf-8"), canonical_doc_provenance_bytes(provenance)
        )

        freshness_path = self.temp_dir / "doc_freshness.json"
        provenance_path = self.temp_dir / "doc_provenance.json"
        write_doc_freshness_atomic(freshness_path, freshness)
        write_doc_provenance_atomic(provenance_path, provenance)

        self.assertEqual(freshness_path.read_bytes(), freshness_text.encode("utf-8"))
        self.assertEqual(provenance_path.read_bytes(), provenance_text.encode("utf-8"))
        self.assertEqual(load_doc_freshness(freshness_path), freshness)
        self.assertEqual(load_doc_provenance(provenance_path), provenance)
        self.assertEqual(list(self.temp_dir.iterdir()), [freshness_path, provenance_path])

    def test_additive_unknown_fields_are_preserved_by_canonical_serialization(self) -> None:
        document = valid_freshness_document()
        document["producer_extension"] = {"enabled": True}
        entries = document["entries"]
        assert isinstance(entries, list)
        entries[0]["evidence_version"] = 1

        validate_doc_freshness(document)
        loaded = json.loads(serialize_doc_freshness(document))
        self.assertEqual(loaded["producer_extension"], {"enabled": True})
        self.assertEqual(loaded["entries"][0]["evidence_version"], 1)

    def test_required_nullable_evidence_fields_remain_present(self) -> None:
        document = valid_freshness_document()
        entry = document["entries"][1]
        self.assertEqual(
            set(entry),
            {
                "source_path",
                "doc_path",
                "expected_doc_paths",
                "current_source_hash",
                "recorded_source_hash",
                "doc_hash",
                "status",
                "reason",
                "diagnostics",
            },
        )
        self.assertIsNone(entry["doc_path"])
        self.assertIsNone(entry["recorded_source_hash"])
        self.assertIsNone(entry["doc_hash"])
        validate_doc_freshness(document)

    def test_freshness_rejects_invalid_paths_hashes_enums_and_invariants(self) -> None:
        invalid_paths = [
            "../app.py",
            "/app.py",
            "src\\app.py",
            "src/./app.py",
            "src//app.py",
            "src/app.py/..",
            "src/app\x00.py",
            "C:/app.py",
        ]
        for path in invalid_paths:
            with self.subTest(path=path):
                document = valid_freshness_document()
                document["entries"][0]["source_path"] = path
                with self.assertRaises(DocFreshnessContractError):
                    validate_doc_freshness(document)

        invalid_hash = valid_freshness_document()
        invalid_hash["entries"][0]["current_source_hash"] = HASH_A.upper()
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(invalid_hash)

        invalid_status = valid_freshness_document()
        invalid_status["entries"][0]["status"] = "current"
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(invalid_status)

        invalid_reason = valid_freshness_document()
        invalid_reason["entries"][0]["reason"] = "not-a-stable-reason"
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(invalid_reason)

        invalid_mapping = valid_freshness_document()
        invalid_mapping["entries"][0]["expected_doc_paths"] = [
            "src/app.md",
            "src/app.py.md",
        ]
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(invalid_mapping)

        invalid_missing_evidence = valid_freshness_document()
        del invalid_missing_evidence["entries"][1]["doc_hash"]
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(invalid_missing_evidence)

        invalid_doc_hash = valid_freshness_document()
        invalid_doc_hash["entries"][1]["doc_hash"] = HASH_A
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(invalid_doc_hash)

    def test_freshness_rejects_order_duplicate_count_and_status_reason_mismatch(self) -> None:
        out_of_order = valid_freshness_document()
        out_of_order["entries"].reverse()
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(out_of_order)

        duplicate = valid_freshness_document()
        duplicate["entries"].append(copy.deepcopy(duplicate["entries"][0]))
        duplicate["counts"]["fresh"] = 2
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(duplicate)

        wrong_counts = valid_freshness_document()
        wrong_counts["counts"]["fresh"] = 2
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(wrong_counts)

        mismatched_reason = valid_freshness_document()
        mismatched_reason["entries"][0]["reason"] = "source_hash_mismatch"
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(mismatched_reason)

    def test_unknown_schema_major_and_diagnostics_fail_closed(self) -> None:
        unknown_major = valid_freshness_document()
        unknown_major["schema_version"] = "2.0"
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(unknown_major)

        unknown_diagnostic = valid_freshness_document()
        unknown_diagnostic["diagnostics"] = ["free-form failure text"]
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(unknown_diagnostic)

        unsorted_diagnostics = valid_freshness_document()
        unsorted_diagnostics["diagnostics"] = [
            "source_hash_unavailable",
            "ambiguous_doc_mapping",
        ]
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_freshness(unsorted_diagnostics)

    def test_json_schemas_match_runtime_vocabularies_and_supported_majors(self) -> None:
        schema_dir = Path(__file__).resolve().parents[1] / "schemas"
        with (schema_dir / "doc-freshness.schema.json").open(encoding="utf-8") as handle:
            freshness_schema = json.load(handle)
        with (schema_dir / "doc-provenance.schema.json").open(encoding="utf-8") as handle:
            provenance_schema = json.load(handle)

        self.assertEqual(
            freshness_schema["$defs"]["reason"]["enum"], list(FRESHNESS_REASONS)
        )
        self.assertEqual(
            freshness_schema["$defs"]["diagnosticCode"]["enum"],
            list(DIAGNOSTIC_CODES),
        )
        self.assertEqual(
            provenance_schema["$defs"]["diagnosticCode"]["enum"],
            list(DIAGNOSTIC_CODES),
        )

        for schema, major in (
            (freshness_schema, DOC_FRESHNESS_SCHEMA_MAJOR),
            (provenance_schema, DOC_PROVENANCE_SCHEMA_MAJOR),
        ):
            pattern = re.compile(schema["properties"]["schema_version"]["pattern"])
            self.assertIsNotNone(pattern.fullmatch(f"{major}.0"))
            self.assertIsNotNone(pattern.fullmatch(f"{major}.99"))
            self.assertIsNone(pattern.fullmatch(f"{major + 1}.0"))

    def test_provenance_rejects_invalid_hashes_paths_order_and_duplicate_identity(self) -> None:
        invalid_hash = valid_provenance_document()
        invalid_hash["assertions"][0]["recorded_doc_hash"] = "D" * 64
        with self.assertRaises(DocProvenanceContractError):
            validate_doc_provenance(invalid_hash)

        invalid_path = valid_provenance_document()
        invalid_path["assertions"][0]["doc_path"] = "../escape.md"
        with self.assertRaises(DocFreshnessContractError):
            validate_doc_provenance(invalid_path)

        out_of_order = valid_provenance_document()
        out_of_order["assertions"].reverse()
        with self.assertRaises(DocProvenanceContractError):
            validate_doc_provenance(out_of_order)

        duplicate = valid_provenance_document()
        duplicate["assertions"].append(copy.deepcopy(duplicate["assertions"][0]))
        with self.assertRaises(DocProvenanceContractError):
            validate_doc_provenance(duplicate)

        unknown_major = valid_provenance_document()
        unknown_major["schema_version"] = "2.0"
        with self.assertRaises(DocProvenanceContractError):
            validate_doc_provenance(unknown_major)

    def test_atomic_writer_does_not_replace_existing_artifact_with_invalid_data(self) -> None:
        path = self.temp_dir / "nested" / "doc_freshness.json"
        document = valid_freshness_document()
        write_doc_freshness_atomic(path, document)
        original = path.read_bytes()

        invalid = copy.deepcopy(document)
        invalid["counts"]["fresh"] = 99
        with self.assertRaises(DocFreshnessContractError):
            write_doc_freshness_atomic(path, invalid)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(
            [item.name for item in path.parent.iterdir()], ["doc_freshness.json"]
        )


class TestDocFreshnessFileObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_safe_hash_regular_file_hashes_exact_bytes(self) -> None:
        source = self.temp_dir / "src" / "app.py"
        source.parent.mkdir()
        payload = b"print('exact bytes')\r\n"
        source.write_bytes(payload)

        observation = safe_hash_regular_file(source, self.temp_dir)

        self.assertEqual(observation.digest, hashlib.sha256(payload).hexdigest())
        self.assertEqual(observation.sha256, observation.digest)
        self.assertTrue(observation.available)
        self.assertEqual(observation.attempts, 1)

    def test_safe_hash_rejects_escape_symlink_and_non_regular_file(self) -> None:
        escaped = safe_hash_regular_file(self.temp_dir / ".." / "escape.py", self.temp_dir)
        self.assertIsNone(escaped.digest)
        self.assertEqual(escaped.reason, "unsafe_path")

        outside = self.temp_dir.parent / "doc-freshness-outside.md"
        outside.write_bytes(b"outside")
        try:
            symlink = self.temp_dir / "src" / "link.md"
            symlink.parent.mkdir()
            symlink.symlink_to(outside)
            symlink_observation = safe_hash_regular_file(symlink, self.temp_dir)
            self.assertIsNone(symlink_observation.digest)
            self.assertEqual(symlink_observation.reason, "unsafe_path")
        finally:
            outside.unlink(missing_ok=True)

        directory = self.temp_dir / "directory"
        directory.mkdir()
        directory_observation = safe_hash_regular_file(directory, self.temp_dir)
        self.assertIsNone(directory_observation.digest)
        self.assertEqual(directory_observation.reason, "non_regular_file")

        if hasattr(os, "mkfifo"):
            fifo = self.temp_dir / "pipe"
            try:
                os.mkfifo(fifo)
            except (NotImplementedError, OSError):
                pass
            else:
                fifo_observation = safe_hash_regular_file(fifo, self.temp_dir)
                self.assertIsNone(fifo_observation.digest)
                self.assertEqual(fifo_observation.reason, "non_regular_file")

    def test_safe_writer_does_not_write_through_parent_swap_before_temp_open(self) -> None:
        output_root = self.temp_dir / "output"
        output_root.mkdir()
        outside_root = self.temp_dir / "outside"
        outside_root.mkdir()
        escaped_directory = outside_root / "escaped"
        destination_parent = output_root / "nested"

        real_open = os.open
        swapped = False

        def replace_parent_before_temp_open(path, *args, **kwargs):
            nonlocal swapped
            if (
                isinstance(path, str)
                and path.startswith(".")
                and kwargs.get("dir_fd") is not None
                and not swapped
            ):
                destination_parent.rename(escaped_directory)
                destination_parent.symlink_to(escaped_directory, target_is_directory=True)
                swapped = True
            return real_open(path, *args, **kwargs)

        with patch("isohyps.doc_freshness.os.open", new=replace_parent_before_temp_open):
            with self.assertRaises(DocProvenanceContractError):
                write_text_regular_file(
                    destination_parent / "app.py.md",
                    "must not escape",
                    output_root,
                )

        self.assertTrue(swapped)
        self.assertTrue(destination_parent.is_symlink())
        self.assertFalse((escaped_directory / "app.py.md").exists())

    def test_safe_writer_rejects_parent_swap_at_truncate_before_external_write(self) -> None:
        output_root = self.temp_dir / "output"
        output_root.mkdir()
        outside_root = self.temp_dir / "outside"
        outside_root.mkdir()
        escaped_directory = outside_root / "escaped"
        destination_parent = output_root / "nested"
        destination_parent.mkdir()
        destination = destination_parent / "app.py.md"
        destination.write_text("keep this document\n", encoding="utf-8")

        real_ftruncate = os.ftruncate
        swapped = False

        def replace_parent_before_truncate(file_descriptor, length):
            nonlocal swapped
            if not swapped:
                destination_parent.rename(escaped_directory)
                destination_parent.symlink_to(escaped_directory, target_is_directory=True)
                swapped = True
            return real_ftruncate(file_descriptor, length)

        with patch("isohyps.doc_freshness.os.ftruncate", new=replace_parent_before_truncate):
            with self.assertRaises(DocProvenanceContractError):
                write_text_regular_file(
                    destination,
                    "must not escape",
                    output_root,
                )

        self.assertTrue(swapped)
        self.assertTrue(destination_parent.is_symlink())
        self.assertEqual(
            (escaped_directory / "app.py.md").read_text(encoding="utf-8"),
            "keep this document\n",
        )
        self.assertEqual(
            [item.name for item in output_root.iterdir()], ["nested"]
        )

    def test_safe_writer_rejects_parent_swap_at_commit_without_external_payload(self) -> None:
        output_root = self.temp_dir / "output"
        output_root.mkdir()
        outside_root = self.temp_dir / "outside"
        outside_root.mkdir()
        escaped_directory = outside_root / "escaped"
        destination_parent = output_root / "nested"
        destination_parent.mkdir()
        destination = destination_parent / "app.py.md"
        destination.write_text("keep this document\n", encoding="utf-8")

        real_replace = os.replace
        swapped = False

        def replace_parent_before_commit(source, target, **kwargs):
            nonlocal swapped
            if not swapped:
                destination_parent.rename(escaped_directory)
                destination_parent.symlink_to(escaped_directory, target_is_directory=True)
                swapped = True
            return real_replace(source, target, **kwargs)

        with patch("isohyps.doc_freshness.os.replace", new=replace_parent_before_commit):
            with self.assertRaises(DocProvenanceContractError):
                write_text_regular_file(
                    destination,
                    "must not escape",
                    output_root,
                )

        self.assertTrue(swapped)
        self.assertTrue(destination_parent.is_symlink())
        self.assertEqual(
            (escaped_directory / "app.py.md").read_text(encoding="utf-8"),
            "keep this document\n",
        )
        self.assertEqual(
            [item.name for item in output_root.iterdir()], ["nested"]
        )

    def test_safe_writer_restores_destination_when_temp_path_is_replaced_before_commit(self) -> None:
        output_root = self.temp_dir / "output"
        output_root.mkdir()
        destination = output_root / "app.py.md"
        original = b"keep this document\n"
        destination.write_bytes(original)

        real_replace = os.replace
        replaced_source = False

        def replace_source_before_commit(source, target, **kwargs):
            nonlocal replaced_source
            if not replaced_source:
                source_dir_fd = kwargs["src_dir_fd"]
                os.unlink(source, dir_fd=source_dir_fd)
                attacker_fd = os.open(
                    source,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=source_dir_fd,
                )
                try:
                    os.write(attacker_fd, b"ATTACKER-CONTROLLED")
                finally:
                    os.close(attacker_fd)
                replaced_source = True
            return real_replace(source, target, **kwargs)

        with patch("isohyps.doc_freshness.os.replace", new=replace_source_before_commit):
            with self.assertRaises(DocProvenanceContractError):
                write_text_regular_file(destination, "new document\n", output_root)

        self.assertTrue(replaced_source)
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual([item.name for item in output_root.iterdir()], ["app.py.md"])


class TestDocProvenanceRecorder(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.source_root = self.temp_dir / "repo"
        self.output_root = self.temp_dir / "docs"
        self.source_root.mkdir()
        self.output_root.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _record(self, source_path: str = "src/app.py", doc_path: str = "src/app.py.md") -> None:
        record_doc_provenance(
            self.output_root / "doc_provenance.json",
            source_root=self.source_root,
            output_root=self.output_root,
            source_path=source_path,
            doc_path=doc_path,
            producer=DOC_PROVENANCE_PRODUCER,
            producer_version=DOC_PROVENANCE_PRODUCER_VERSION,
        )

    def test_records_current_source_and_doc_bytes_and_replaces_identity(self) -> None:
        source = self.source_root / "src/app.py"
        document = self.output_root / "src/app.py.md"
        source.parent.mkdir()
        document.parent.mkdir()
        source.write_bytes(b"source-v1")
        document.write_bytes(b"doc-v1")

        self._record()
        first = load_doc_provenance(self.output_root / "doc_provenance.json")
        self.assertEqual(len(first["assertions"]), 1)
        self.assertEqual(first["assertions"][0]["source_path"], "src/app.py")
        self.assertEqual(first["assertions"][0]["doc_path"], "src/app.py.md")
        self.assertEqual(
            first["assertions"][0]["recorded_source_hash"],
            hashlib.sha256(b"source-v1").hexdigest(),
        )
        self.assertEqual(
            first["assertions"][0]["recorded_doc_hash"],
            hashlib.sha256(b"doc-v1").hexdigest(),
        )

        source.write_bytes(b"source-v2")
        document.write_bytes(b"doc-v2")
        self._record()
        second = load_doc_provenance(self.output_root / "doc_provenance.json")
        self.assertEqual(len(second["assertions"]), 1)
        self.assertEqual(
            second["assertions"][0]["recorded_source_hash"],
            hashlib.sha256(b"source-v2").hexdigest(),
        )
        self.assertEqual(
            second["assertions"][0]["recorded_doc_hash"],
            hashlib.sha256(b"doc-v2").hexdigest(),
        )

    def test_unavailable_source_does_not_replace_existing_provenance(self) -> None:
        source = self.source_root / "src/app.py"
        document = self.output_root / "src/app.py.md"
        source.parent.mkdir()
        document.parent.mkdir()
        source.write_bytes(b"source-v1")
        document.write_bytes(b"doc-v1")
        self._record()
        provenance_path = self.output_root / "doc_provenance.json"
        original = provenance_path.read_bytes()

        source.unlink()
        with self.assertRaises(DocProvenanceRecordingError) as context:
            self._record()

        self.assertEqual(context.exception.reason, "source_hash_unavailable")
        self.assertEqual(provenance_path.read_bytes(), original)
        self.assertEqual(
            [item.name for item in self.output_root.iterdir()],
            ["src", "doc_provenance.json"],
        )

    def test_provenance_writer_uses_safe_writer_instead_of_path_based_tempfile(self) -> None:
        provenance = valid_provenance_document()

        with patch("isohyps.doc_freshness.tempfile.mkstemp") as mkstemp:
            with patch("isohyps.doc_freshness._write_atomic") as path_writer:
                write_doc_provenance_atomic(
                    self.output_root / "doc_provenance.json",
                    provenance,
                )

        mkstemp.assert_not_called()
        path_writer.assert_not_called()
        self.assertEqual(
            load_doc_provenance(self.output_root / "doc_provenance.json"),
            provenance,
        )

    def test_provenance_writer_rejects_commit_parent_swap_without_external_payload(self) -> None:
        output_root = self.output_root
        outside_root = self.temp_dir / "outside"
        outside_root.mkdir()
        escaped_directory = outside_root / "escaped"
        destination_parent = output_root / "nested"
        destination_parent.mkdir()
        destination = destination_parent / "doc_provenance.json"

        real_replace = os.replace
        swapped = False

        def replace_parent_before_commit(source, target, **kwargs):
            nonlocal swapped
            if not swapped:
                destination_parent.rename(escaped_directory)
                destination_parent.symlink_to(escaped_directory, target_is_directory=True)
                swapped = True
            return real_replace(source, target, **kwargs)

        with patch("isohyps.doc_freshness.os.replace", new=replace_parent_before_commit):
            with self.assertRaises(DocProvenanceContractError):
                write_doc_provenance_atomic(destination, valid_provenance_document())

        self.assertTrue(swapped)
        self.assertTrue(destination_parent.is_symlink())
        self.assertFalse((escaped_directory / "doc_provenance.json").exists())

    def test_safe_hash_retries_one_metadata_race_then_returns_digest(self) -> None:
        source = self.temp_dir / "src.py"
        source.write_bytes(b"source")
        digest = hashlib.sha256(b"source").hexdigest()

        with patch(
            "isohyps.doc_freshness._hash_regular_file_once",
            side_effect=[
                HashObservation(None, "file_changed_during_scan"),
                HashObservation(digest),
            ],
        ) as hash_once:
            observation = safe_hash_regular_file(source, self.temp_dir)

        self.assertEqual(hash_once.call_count, 2)
        self.assertEqual(observation.digest, digest)
        self.assertEqual(observation.attempts, 2)

    def test_safe_hash_keeps_unstable_file_unavailable_after_retry(self) -> None:
        source = self.temp_dir / "src.py"
        source.write_bytes(b"source")

        with patch(
            "isohyps.doc_freshness._hash_regular_file_once",
            side_effect=[
                HashObservation(None, "file_changed_during_scan"),
                HashObservation(None, "file_changed_during_scan"),
            ],
        ):
            observation = safe_hash_regular_file(source, self.temp_dir)

        self.assertIsNone(observation.digest)
        self.assertEqual(observation.reason, "file_changed_during_scan")
        self.assertEqual(observation.attempts, 2)


class TestDocFreshnessMapping(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_mapping_distinguishes_zero_one_and_two_candidates(self) -> None:
        one = self.temp_dir / "src" / "one.py.md"
        one.parent.mkdir()
        one.write_bytes(b"one")

        two_first = self.temp_dir / "src" / "two.py.md"
        two_second = self.temp_dir / "src" / "two.md"
        two_first.write_bytes(b"two first")
        two_second.write_bytes(b"two second")

        resolution = resolve_doc_mappings(
            [
                CoverageTarget(path="src/two.py"),
                {"path": "src/missing.py"},
                {"source_path": "src/one.py"},
            ],
            self.temp_dir,
        )

        self.assertEqual(
            [entry.source_path for entry in resolution.entries],
            ["src/missing.py", "src/one.py", "src/two.py"],
        )
        self.assertEqual(resolution["src/one.py"].doc_path, "src/one.py.md")
        self.assertIsNone(resolution["src/missing.py"].doc_path)
        self.assertEqual(resolution["src/missing.py"].reason, "doc_missing")
        self.assertIsNone(resolution["src/two.py"].doc_path)
        self.assertEqual(resolution["src/two.py"].reason, "ambiguous_doc_mapping")
        self.assertEqual(
            resolution["src/two.py"].existing_doc_paths,
            ("src/two.py.md", "src/two.md"),
        )

        reversed_resolution = resolve_doc_mappings(
            ["src/one.py", "src/two.py", "src/missing.py"], self.temp_dir
        )
        self.assertEqual(resolution, reversed_resolution)

    def test_mapping_rejects_duplicate_source_identity(self) -> None:
        with self.assertRaisesRegex(DocFreshnessContractError, "duplicate_source_identity"):
            resolve_doc_mappings(["src/app.py", "src/app.py"], self.temp_dir)

    def test_mapping_does_not_select_reverse_or_portable_collisions(self) -> None:
        reverse_doc = self.temp_dir / "src" / "app.md"
        reverse_doc.parent.mkdir()
        reverse_doc.write_bytes(b"shared")
        reverse = resolve_doc_mappings(
            ["src/app.py", "src/app.md"], self.temp_dir
        )
        for entry in reverse.entries:
            self.assertIsNone(entry.doc_path)
            self.assertEqual(entry.reason, "doc_identity_collision")

        lower_doc = self.temp_dir / "src" / "case.py.md"
        upper_doc = self.temp_dir / "src" / "CASE.py.md"
        lower_doc.write_bytes(b"lower")
        upper_doc.write_bytes(b"upper")
        portable = resolve_doc_mappings(
            ["src/case.py", "src/CASE.py"], self.temp_dir
        )
        for entry in portable.entries:
            self.assertIsNone(entry.doc_path)
            self.assertEqual(entry.reason, "doc_identity_collision")
        self.assertIn("doc_identity_collision", portable.diagnostics)

        composed = self.temp_dir / "src" / "café.py.md"
        decomposed = self.temp_dir / "src" / "café.py.md"
        composed.write_bytes(b"composed")
        decomposed.write_bytes(b"decomposed")
        unicode_resolution = resolve_doc_mappings(
            ["src/café.py", "src/café.py"], self.temp_dir
        )
        for entry in unicode_resolution.entries:
            self.assertIsNone(entry.doc_path)
            self.assertEqual(entry.reason, "doc_identity_collision")

    def test_mapping_rejects_symlink_and_non_regular_candidates(self) -> None:
        outside = self.temp_dir.parent / "mapping-outside.md"
        outside.write_bytes(b"outside")
        try:
            symlink = self.temp_dir / "src" / "app.py.md"
            symlink.parent.mkdir()
            symlink.symlink_to(outside)
            symlink_resolution = resolve_doc_mappings(["src/app.py"], self.temp_dir)
            self.assertIsNone(symlink_resolution.entries[0].doc_path)
            self.assertEqual(symlink_resolution.entries[0].reason, "unsafe_doc_path")
        finally:
            outside.unlink(missing_ok=True)

        directory = self.temp_dir / "src" / "directory.py.md"
        directory.mkdir(parents=True)
        directory_resolution = resolve_doc_mappings(
            ["src/directory.py"], self.temp_dir
        )
        self.assertIsNone(directory_resolution.entries[0].doc_path)
        self.assertEqual(directory_resolution.entries[0].reason, "unsafe_doc_path")


class TestDocFreshnessEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def provenance(
        source_path: str,
        doc_path: str,
        source_hash: str,
        doc_hash: str,
    ) -> dict[str, object]:
        return {
            "schema_version": DOC_PROVENANCE_SCHEMA_VERSION,
            "hash_algorithm": "sha256",
            "assertions": [
                {
                    "source_path": source_path,
                    "doc_path": doc_path,
                    "recorded_source_hash": source_hash,
                    "recorded_doc_hash": doc_hash,
                    "producer": "test.producer",
                    "producer_version": "1",
                }
            ],
        }

    def make_document_fixture(
        self,
        *,
        source_path: str = "src/app.py",
        source_bytes: bytes = b"source-v1",
        doc_path: str | None = "src/app.py.md",
        doc_bytes: bytes = b"doc-v1",
    ) -> tuple[str, str | None, str, str | None, object]:
        source = self.temp_dir / source_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(source_bytes)
        if doc_path is not None:
            document = self.temp_dir / doc_path
            document.parent.mkdir(parents=True, exist_ok=True)
            document.write_bytes(doc_bytes)
        resolution = resolve_doc_mappings([source_path], self.temp_dir)
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        doc_hash = hashlib.sha256(doc_bytes).hexdigest() if doc_path is not None else None
        return source_path, doc_path, source_hash, doc_hash, resolution

    def evaluate_case(
        self,
        *,
        recorded_source_hash: str | None,
        recorded_doc_hash: str | None = None,
        current_source_hash: str | None = None,
        current_doc_hash: str | None = None,
        source_path: str = "src/app.py",
        doc_path: str | None = "src/app.py.md",
        source_bytes: bytes = b"source-v1",
        doc_bytes: bytes = b"doc-v1",
        provenance: object = "use-default",
    ) -> dict[str, object]:
        source_path, resolved_doc_path, source_hash, doc_hash, resolution = self.make_document_fixture(
            source_path=source_path,
            source_bytes=source_bytes,
            doc_path=doc_path,
            doc_bytes=doc_bytes,
        )
        if current_source_hash is None:
            current_source_hash = recorded_source_hash or source_hash
        current_doc_hash = current_doc_hash if current_doc_hash is not None else doc_hash
        if provenance == "use-default":
            if resolved_doc_path is None:
                provenance = None
            else:
                provenance = self.provenance(
                    source_path,
                    resolved_doc_path,
                    recorded_source_hash or source_hash,
                    recorded_doc_hash or current_doc_hash or doc_hash or HASH_A,
                )
        document = evaluate_doc_freshness(
            [{"path": source_path, "hash": current_source_hash}],
            resolution,
            provenance,
            doc_hashes={resolved_doc_path: current_doc_hash}
            if resolved_doc_path is not None
            else {},
        )
        return document["entries"][0]

    def test_valid_provenance_is_required_for_fresh_and_stale(self) -> None:
        fresh = self.evaluate_case(recorded_source_hash=HASH_A)
        self.assertEqual((fresh["status"], fresh["reason"]), ("fresh", "source_hash_match"))
        self.assertEqual(fresh["recorded_source_hash"], HASH_A)

        stale = self.evaluate_case(
            recorded_source_hash=HASH_B,
            current_source_hash=HASH_A,
        )
        self.assertEqual((stale["status"], stale["reason"]), ("stale", "source_hash_mismatch"))
        self.assertEqual(stale["recorded_source_hash"], HASH_B)

    def test_missing_doc_and_missing_recorded_hash_are_distinct(self) -> None:
        missing = self.evaluate_case(
            recorded_source_hash=None,
            doc_path=None,
            provenance=None,
        )
        self.assertEqual((missing["status"], missing["reason"]), ("missing", "doc_missing"))
        self.assertIsNone(missing["doc_path"])
        self.assertIsNone(missing["doc_hash"])

        recorded_missing = self.evaluate_case(
            recorded_source_hash=None,
            provenance=None,
        )
        self.assertEqual(
            (recorded_missing["status"], recorded_missing["reason"]),
            ("unknown", "recorded_source_hash_missing"),
        )
        self.assertEqual(recorded_missing["doc_path"], "src/app.py.md")
        self.assertIsNotNone(recorded_missing["doc_hash"])

    def test_source_hash_unavailable_precedes_doc_missing(self) -> None:
        source_path, _doc_path, _source_hash, _doc_hash, resolution = self.make_document_fixture(
            source_path="src/unavailable.py", doc_path=None
        )
        document = evaluate_doc_freshness(
            [{"path": source_path, "hash": HashObservation(None, "file_missing")}],
            resolution,
            None,
        )
        entry = document["entries"][0]
        self.assertEqual((entry["status"], entry["reason"]), ("unknown", "source_hash_unavailable"))
        self.assertIn("doc_missing", entry["diagnostics"])

    def test_hand_edit_and_wrong_doc_binding_fail_closed(self) -> None:
        hand_edited = self.evaluate_case(
            recorded_source_hash=HASH_A,
            recorded_doc_hash=HASH_B,
            current_doc_hash=HASH_C,
        )
        self.assertEqual(
            (hand_edited["status"], hand_edited["reason"]),
            ("unknown", "provenance_doc_hash_mismatch"),
        )
        self.assertIsNone(hand_edited["recorded_source_hash"])

        source_path, doc_path, source_hash, doc_hash, resolution = self.make_document_fixture()
        assert doc_path is not None
        wrong_doc_provenance = self.provenance(
            source_path, "src/app.md", source_hash, doc_hash or HASH_A
        )
        document = evaluate_doc_freshness(
            [{"path": source_path, "hash": source_hash}],
            resolution,
            wrong_doc_provenance,
            doc_hashes={doc_path: doc_hash},
        )
        self.assertEqual(
            (document["entries"][0]["status"], document["entries"][0]["reason"]),
            ("unknown", "provenance_binding_mismatch"),
        )

    def test_rename_does_not_inherit_old_assertion(self) -> None:
        source_path, doc_path, source_hash, doc_hash, resolution = self.make_document_fixture(
            source_path="src/new.py", doc_path="src/new.py.md"
        )
        assert doc_path is not None and doc_hash is not None
        old_provenance = self.provenance(
            "src/old.py", "src/old.py.md", source_hash, doc_hash
        )
        document = evaluate_doc_freshness(
            [{"path": source_path, "hash": source_hash}],
            resolution,
            old_provenance,
            doc_hashes={doc_path: doc_hash},
        )
        entry = document["entries"][0]
        self.assertEqual((entry["status"], entry["reason"]), ("unknown", "recorded_source_hash_missing"))
        self.assertIn("orphan_provenance_assertion", document["diagnostics"])

    def test_reader_diagnoses_missing_invalid_and_unknown_major_provenance(self) -> None:
        missing = read_doc_provenance(self.temp_dir / "missing.json")
        self.assertFalse(missing.valid)
        self.assertEqual(missing.reason, "recorded_source_hash_missing")

        invalid_path = self.temp_dir / "invalid.json"
        invalid_path.write_text("not-json", encoding="utf-8")
        invalid = read_doc_provenance(invalid_path)
        self.assertEqual(invalid.reason, "provenance_artifact_invalid")

        unknown_path = self.temp_dir / "unknown.json"
        unknown_document = self.provenance("src/app.py", "src/app.py.md", HASH_A, HASH_B)
        unknown_document["schema_version"] = "2.0"
        unknown_path.write_text(json.dumps(unknown_document), encoding="utf-8")
        unknown = read_doc_provenance(unknown_path)
        self.assertEqual(unknown.reason, "provenance_artifact_invalid")

        invalid_assertion_path = self.temp_dir / "invalid-assertion.json"
        invalid_assertion = self.provenance("src/app.py", "src/app.py.md", HASH_A, HASH_B)
        invalid_assertion["assertions"][0]["recorded_source_hash"] = "not-a-hash"
        invalid_assertion_path.write_text(json.dumps(invalid_assertion), encoding="utf-8")
        assertion_result = read_doc_provenance(invalid_assertion_path)
        self.assertEqual(assertion_result.reason, "provenance_assertion_invalid")

    def test_projection_maps_unknown_to_unavailable_without_current_fallback(self) -> None:
        source_path, doc_path, source_hash, doc_hash, resolution = self.make_document_fixture()
        assert doc_path is not None and doc_hash is not None
        document = evaluate_doc_freshness(
            [{"path": source_path, "hash": source_hash}],
            resolution,
            ProvenanceFailure("recorded_source_hash_missing"),
            doc_hashes={doc_path: doc_hash},
        )
        projection = project_freshness(document)
        self.assertEqual(project_doc_statuses(document), {source_path: "unavailable"})
        self.assertEqual(projection["unknown_docs"], [source_path])
        self.assertEqual(projection["valid_docs"], [])
        self.assertEqual(projection["coverage_summary"]["unknown_docs"], 1)
        self.assertEqual(projection["attention_diagnostics"][0]["code"], "recorded_source_hash_missing")


if __name__ == "__main__":
    unittest.main()
