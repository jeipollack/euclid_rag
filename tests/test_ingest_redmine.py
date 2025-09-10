"""Tests for the JSON ingestion logic."""

import json
from unittest.mock import MagicMock

import pytest

from euclid.rag.ingestion.ingest_redmine import JSONIngestor


# --- Tests for _validate_json_dir ---
def test_validate_json_dir_missing(tmp_path):
    bad_path = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        JSONIngestor._validate_json_dir(bad_path)


def test_validate_json_dir_not_a_directory(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a dir")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        JSONIngestor._validate_json_dir(file_path)


def test_validate_json_dir_ok(tmp_path):
    dir_path = tmp_path / "valid_dir"
    dir_path.mkdir()
    # Should not raise
    JSONIngestor._validate_json_dir(dir_path)


# --- Tests for ingest_json_files ---
@pytest.fixture
def dummy_ingestor(tmp_path):
    """Fixture that returns a JSONIngestor with mocked dependencies."""
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    # Minimal data_config
    data_config = {"embedding_model_name": "intfloat/e5-small-v2"}
    cleaner = MagicMock()
    cleaner.prepare_for_ingestion.side_effect = lambda data: data

    ingestor = JSONIngestor(
        index_dir=index_dir,
        json_dir=json_dir,
        cleaner=cleaner,
        data_config=data_config,
    )
    # Patch methods to avoid heavy dependencies
    ingestor._get_existing_source_keys = lambda: set()
    ingestor._embedder = MagicMock()
    ingestor._vectorstore = MagicMock()
    ingestor._vectorstore.save_local = MagicMock()
    return ingestor


def test_ingest_json_files_empty_dir(dummy_ingestor, caplog):
    # No files created
    dummy_ingestor.ingest_json_files()
    assert any("No JSON files found" in m for m in caplog.messages)


def test_ingest_json_files_valid_file(dummy_ingestor, tmp_path):
    # Create a minimal valid JSON file (list of dicts with content + metadata)
    json_file = dummy_ingestor._json_dir / "test.json"
    data = [{"content": "Hello world", "metadata": {"title": "test"}}]
    json_file.write_text(json.dumps(data), encoding="utf-8")

    dummy_ingestor._cleaner.prepare_for_ingestion = lambda d: d
    dummy_ingestor._vectorstore.add_documents = MagicMock()
    dummy_ingestor._vectorstore.save_local = MagicMock()

    dummy_ingestor.ingest_json_files()

    dummy_ingestor._vectorstore.add_documents.assert_called_once()
    dummy_ingestor._vectorstore.save_local.assert_called_once()


def test_ingest_json_files_invalid_json(dummy_ingestor, tmp_path, caplog):
    # Write invalid JSON
    bad_file = dummy_ingestor._json_dir / "bad.json"
    bad_file.write_text("{invalid json", encoding="utf-8")

    dummy_ingestor.ingest_json_files()
    assert any("Failed to read/parse" in m for m in caplog.messages)
