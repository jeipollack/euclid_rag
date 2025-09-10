"""Tests for the Redmine JSON ingestor."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from euclid.rag.ingestion.ingest_redmine import JSONIngestor


# --- Tests for _validate_json_dir ---
def test_validate_json_dir_missing(tmp_path: Path) -> None:
    bad_path: Path = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        JSONIngestor._validate_json_dir(bad_path)


def test_validate_json_dir_not_a_directory(tmp_path: Path) -> None:
    file_path: Path = tmp_path / "file.txt"
    file_path.write_text("not a dir")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        JSONIngestor._validate_json_dir(file_path)


def test_validate_json_dir_ok(tmp_path: Path) -> None:
    dir_path: Path = tmp_path / "valid_dir"
    dir_path.mkdir()
    # Should not raise
    JSONIngestor._validate_json_dir(dir_path)


@pytest.fixture
def dummy_ingestor(tmp_path: Path) -> JSONIngestor:
    """Fixture that returns a JSONIngestor with mocked dependencies."""
    json_dir: Path = tmp_path / "json"
    json_dir.mkdir()
    index_dir: Path = tmp_path / "index"
    index_dir.mkdir()

    # Minimal data_config
    data_config: dict = {"embedding_model_name": "intfloat/e5-small-v2"}

    cleaner = MagicMock()
    cleaner.prepare_for_ingestion.side_effect = lambda data: data

    ingestor: JSONIngestor = JSONIngestor(
        index_dir=index_dir,
        json_dir=json_dir,
        cleaner=cleaner,
        data_config=data_config,
    )

    # Patch methods safely
    patcher = patch.object(ingestor, "_get_existing_source_keys", return_value=set())
    patcher.start()
    ingestor._embedder = MagicMock()

    ingestor._vectorstore = MagicMock()
    ingestor._vectorstore.add_documents = MagicMock()
    ingestor._vectorstore.save_local = MagicMock()

    return ingestor


def test_ingest_json_files_empty_dir(dummy_ingestor: JSONIngestor, caplog: pytest.LogCaptureFixture) -> None:
    # No files created
    dummy_ingestor.ingest_json_files()
    assert any("No JSON files found" in m for m in caplog.messages)


def test_ingest_json_files_valid_file(dummy_ingestor: JSONIngestor, tmp_path: Path) -> None:
    # Create a minimal valid JSON file
    json_file: Path = dummy_ingestor._json_dir / "test.json"
    data = [{"content": "Hello world", "metadata": {"title": "test"}}]
    json_file.write_text(json.dumps(data), encoding="utf-8")

    # Patch the cleaner method and vectorstore methods safely
    with (
        patch.object(
            dummy_ingestor._cleaner,
            "prepare_for_ingestion",
            side_effect=lambda d: d,
        ),
        patch.object(
            dummy_ingestor._vectorstore,
            "add_documents",
            new_callable=MagicMock,
        ) as mock_add,
        patch.object(dummy_ingestor._vectorstore, "save_local", new_callable=MagicMock) as mock_save,
    ):
        dummy_ingestor.ingest_json_files()

        mock_add.assert_called_once()
        mock_save.assert_called_once()


def test_ingest_json_files_invalid_json(
    dummy_ingestor: JSONIngestor,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Write invalid JSON
    bad_file: Path = dummy_ingestor._json_dir / "bad.json"
    bad_file.write_text("{invalid json", encoding="utf-8")

    dummy_ingestor.ingest_json_files()
    assert any("Failed to read/parse" in m for m in caplog.messages)
