"""Tests for the Redmine content cleaner."""

import pytest

from euclid.rag.utils.redmine_cleaner import RedmineCleaner


@pytest.fixture
def cleaner() -> RedmineCleaner:
    """Return a RedmineCleaner instance for testing."""
    return RedmineCleaner(max_chunk_length=50)


def test_filter_valid_entries(cleaner: RedmineCleaner) -> None:
    """Test that only entries with 'OK' status are kept."""
    data = [
        {"metadata": {"status": "OK"}, "content": "valid"},
        {"metadata": {"status": "NOK"}, "content": "invalid"},
    ]
    result = cleaner.filter_valid_entries(data)
    assert len(result) == 1
    assert result[0]["content"] == "valid"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("h1. Title", "# Title"),
        ("h3. Subtitle", "### Subtitle"),
        ("Not a header", None),
    ],
)
def test_convert_headers(
    cleaner: RedmineCleaner, line: str, expected: str | None
) -> None:
    """Test Redmine header to Markdown conversion."""
    assert cleaner.convert_redmine_headers(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("* Item 1", "- Item 1"),
        ("** Subitem", "  - Subitem"),
        ("No list", None),
    ],
)
def test_convert_lists(
    cleaner: RedmineCleaner, line: str, expected: str | None
) -> None:
    """Test Redmine list to Markdown conversion."""
    assert cleaner.convert_redmine_lists(line) == expected


def test_convert_links(cleaner: RedmineCleaner) -> None:
    """Test Redmine link to Markdown conversion."""
    line = '"See more":http://example.com'
    expected = "[See more](http://example.com)"
    assert cleaner.convert_redmine_links(line) == expected


def test_convert_bold_italic(cleaner: RedmineCleaner) -> None:
    """Test Redmine bold/italic to Markdown conversion."""
    line = "This is *bold* and _italic_"
    expected = "This is **bold** and *italic*"
    assert cleaner.convert_redmine_bold_italic(line) == expected


def test_convert_images(cleaner: RedmineCleaner) -> None:
    """Test Redmine image to Markdown conversion."""
    line = "See !image.png! here"
    expected = "See ![image](image.png) here"
    assert cleaner.convert_redmine_images(line) == expected


def test_convert_linebreaks(cleaner: RedmineCleaner) -> None:
    """Test Redmine linebreak to Markdown conversion."""
    line = "line1\\nline2"
    expected = "line1  \nline2"
    assert cleaner.convert_redmine_linebreaks(line) == expected


def test_table_conversion(cleaner: RedmineCleaner) -> None:
    """Test Redmine table to Markdown conversion."""
    lines = ["| A | B |", "| 1 | 2 |"]
    expected, consumed = cleaner.convert_redmine_table(lines)
    assert consumed == 2
    assert expected[0].startswith("| A")
    assert expected[1].startswith("| ---")
    assert expected[2].startswith("| 1")


def test_split_content(cleaner: RedmineCleaner) -> None:
    """Test content splitting into chunks of max length."""
    content = "Sentence one. Sentence two! Sentence three?"
    chunks = cleaner.split_content(content)
    assert all(len(c) <= cleaner.max_chunk_length for c in chunks)
    assert len(chunks) >= 1


def test_enrich_with_context(cleaner: RedmineCleaner) -> None:
    """Test that context (hierarchy) is prepended to chunks."""
    entry = {"metadata": {"hierarchy": "ROOT > SUB"}}
    chunk = "Some content"
    enriched = cleaner.enrich_with_context(entry, chunk)
    assert enriched.startswith("[ROOT > SUB]")


def test_prepare_for_ingestion(cleaner: RedmineCleaner) -> None:
    """Test the end-to-end data preparation pipeline."""
    raw_data = [
        {
            "metadata": {
                "status": "OK",
                "created_on": "2024-10-01 12:00",
                "updated_on": "2024-10-02 12:00",
                "project": "proj",
                "page_name": "Test Page",
                "hierarchy": "Root > Page",
            },
            "content": "h1. Title\nThis is *bold* and _italic_.",
        }
    ]
    prepared = cleaner.prepare_for_ingestion(raw_data)
    assert isinstance(prepared, list)
    assert "content" in prepared[0]
    assert "metadata" in prepared[0]
    assert prepared[0]["content"].startswith("[Root > Page]")
    assert prepared[0]["metadata"]["project"] == "PROJ"
