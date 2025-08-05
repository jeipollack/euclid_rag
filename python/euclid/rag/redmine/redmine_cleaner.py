"""
Module providing a utility class for cleaningand preparing
Redmine-exported data.
"""

# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)


class RedmineCleaner:
    """
    A utility class for cleaning and preparing Redmine-exported data
    for ingestion in a RAG pipeline.
    """

    def __init__(self, max_chunk_length: int = 1000) -> None:
        """
        Initialize the cleaner with a maximum chunk size (in characters).

        Args:
            max_chunk_length (int): Max length for each split content chunk.
        """
        self.max_chunk_length = max_chunk_length

    def filter_valid_entries(
        self, data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Keep only entries whose metadata status is not 'NOK'.

        Args:
            data: A list of raw Redmine-exported entries.

        Returns
        -------
            A filtered list of entries with acceptable statuses.
        """
        filtered_entries = [
            entry
            for entry in data
            if entry.get("metadata", {}).get("status") != "NOK"
        ]
        logger.info(
            f"[ING] {len(filtered_entries)} from {len(data)} entries "
            "kept after filtering"
        )
        return filtered_entries

    def _convert_redmine_headers(self, line: str) -> str | None:
        """Convert Redmine headers (h1. to h6.) to Markdown (# to ######)."""
        match = re.match(r"^(h[1-6])\. (.+)", line)
        if not match:
            return None
        level = int(match.group(1)[1])
        title = match.group(2)
        return "#" * level + " " + title

    def _convert_redmine_lists(self, line: str) -> str | None:
        """
        Convert Redmine nested lists (*, **) to Markdown lists with
        indentation.
        """
        match = re.match(r"^(\*+)\s+(.*)", line)
        if not match:
            return None
        stars, content = match.groups()
        indent = "  " * (len(stars) - 1)
        return f"{indent}- {content.strip()}"

    def _convert_redmine_links(self, line: str) -> str:
        """Convert Redmine links "text":url to Markdown [text](url)."""
        return re.sub(r'"([^"]+)":(\S+)', r"[\1](\2)", line)

    def _convert_redmine_bold_italic(self, line: str) -> str:
        """
        Convert *bold* and _italic_ Redmine syntax to Markdown
        **bold** and *italic*.
        """
        # Bold: *text*
        line = re.sub(r"\*(\S(.*?\S)?)\*", r"**\1**", line)
        return re.sub(r"_(\S(.*?\S)?)_", r"*\1*", line)

    def _convert_redmine_linebreaks(self, line: str) -> str:
        """
        Convert explicit Redmine line breaks  in text to Markdown
        double spaces + newline.
        """
        return line.replace("\\n", "  \n")

    def _convert_redmine_images(self, line: str) -> str:
        """
        Convert Redmine image syntax !image.png! or !image.png|widthxheight!
        to Markdown ![alt](image.png).
        """

        def repl(match: re.Match) -> str:
            img_path = match.group(1)
            return f"![image]({img_path})"

        return re.sub(r"!(\S+?)!", repl, line)

    def _convert_redmine_code_blocks(self, lines: list[str]) -> list[str]:
        """
        Convert <pre>...</pre> blocks to Markdown code blocks (```).
        Supports multi-line <pre> sections.
        """
        in_code_block = False
        output = []

        for line in lines:
            if "<pre>" in line:
                in_code_block = True
                output.append("```")
                processed_line = line.replace("<pre>", "").strip()
                if processed_line:
                    output.append(processed_line)
                continue
            if "</pre>" in line:
                in_code_block = False
                processed_line = line.replace("</pre>", "").strip()
                if processed_line:
                    output.append(processed_line)
                output.append("```")
                continue
            if in_code_block:
                output.append(line)
            else:
                output.append(line)
        return output

    def _convert_redmine_table(
        self, lines: list[str]
    ) -> tuple[list[str], int]:
        """
        Convert Redmine table block lines starting with | to Markdown table.
        Returns tuple (converted_lines, number_of_lines_consumed).
        """
        table_lines = []
        i = 0

        # Gather consecutive table lines
        while i < len(lines) and lines[i].startswith("|"):
            table_lines.append(lines[i])
            i += 1
        # Split each row into cells, strip spaces
        rows = [
            [cell.strip() for cell in re.split(r"\|", line)[1:-1]]
            for line in table_lines
        ]
        # Build Markdown table
        # Header row is the first row, underline row is required
        header = rows[0]
        sep = ["---"] * len(header)
        md_table = []
        md_table.append("| " + " | ".join(header) + " |")
        md_table.append("| " + " | ".join(sep) + " |")
        md_table.extend(["| " + " | ".join(row) + " |" for row in rows[1:]])
        return md_table, i

    def redmine_to_markdown(self, text: str) -> str:
        """
        Convert a multiline Redmine-formatted text to Markdown,
        handling headers, lists, links, bold/italic, images, tables,
        code blocks, and line breaks.
        """
        lines = text.splitlines()
        lines = self._convert_redmine_code_blocks(
            lines
        )  # convert <pre> blocks to triple-backtick code blocks
        md_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].rstrip("\r\n")

            # Skip formatting inside Markdown code blocks
            if line.strip() == "```":
                md_lines.append(line)
                i += 1
                while i < len(lines):
                    md_lines.append(lines[i])
                    if lines[i].strip() == "```":
                        i += 1
                        break
                    i += 1
                continue

            # Tables: if line starts with |, handle full table block
            if line.startswith("|"):
                table_md, consumed = self._convert_redmine_table(lines[i:])
                md_lines.extend(table_md)
                i += consumed
                continue

            # Headers
            header_conv = self._convert_redmine_headers(line)
            if header_conv:
                md_lines.append(header_conv)
                i += 1
                continue

            # Lists
            list_conv = self._convert_redmine_lists(line)
            if list_conv:
                md_lines.append(list_conv)
                i += 1
                continue

            # Inline conversions
            line = self._convert_redmine_images(line)
            line = self._convert_redmine_links(line)
            line = self._convert_redmine_bold_italic(line)
            line = self._convert_redmine_linebreaks(line)

            md_lines.append(line)
            i += 1

        return "\n".join(md_lines)

    def normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Clean and normalize metadata fields (e.g., timestamps).

        Args:
            metadata: The metadata dictionary from a Redmine page.

        Returns
        -------
            A normalized metadata dictionary.
        """
        meta = metadata.copy()
        try:
            meta["created_on"] = datetime.strptime(
                meta["created_on"], "%Y-%m-%d %H:%M %z"
            ).replace(tzinfo=datetime.UTC)
            meta["updated_on"] = datetime.strptime(
                meta["updated_on"], "%Y-%m-%d %H:%M %z"
            ).replace(tzinfo=datetime.UTC)
            meta["hierarchy"] = f"{meta['project_path']} > {meta['page_name']}"
        except Exception:
            logger.exception("Failed to parse metadata timestamps")
        meta["project"] = meta.get("project", "").upper()
        return meta

    def split_content(self, content: str) -> list[str]:
        """
        Split long text into smaller chunks based on sentence boundaries.

        Args:
            content: The cleaned full page text.

        Returns
        -------
            A list of shorter text chunks.
        """
        sentences = re.split(r"(?<=[.!?]) +", content)
        chunks, chunk, length = [], [], 0
        for s in sentences:
            if length + len(s) > self.max_chunk_length:
                chunks.append(" ".join(chunk))
                chunk, length = [], 0
            chunk.append(s)
            length += len(s)
        if chunk:
            chunks.append(" ".join(chunk))
        logger.info(f"[ING] Content splint into {len(chunks)} chunks")
        return chunks

    def enrich_with_context(self, entry: dict[str, Any], chunk: str) -> str:
        """
        Add page hierarchy information as a context prefix to the content.

        Args:
            entry: Original Redmine entry.
            chunk: A chunk of cleaned text content.

        Returns
        -------
            Chunk prefixed with hierarchy context.
        """
        base = entry["metadata"].get("hierarchy", "")
        return f"[{base}] {chunk}"

    def prepare_for_ingestion(
        self, raw_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Full pipeline: filter, clean, split and enrich Redmine data.

        Args:
            raw_data: List of Redmine entries (JSON-like).

        Returns
        -------
            List of prepared documents ready for ingestion.
        """
        logger.info("[ING] Preparing files for ingestion...")
        cleaned_data = []
        for entry in self.filter_valid_entries(raw_data):
            content = self.redmine_to_markdown(entry["content"])
            chunks = self.split_content(content)
            metadata = self.normalize_metadata(entry["metadata"])
            for i, chunk in enumerate(chunks):
                logger.debug(metadata)
                cleaned_data.append(
                    {
                        "content": self.enrich_with_context(entry, chunk),
                        "metadata": {
                            **metadata,
                            "chunk_index": i,
                            "page_name": metadata.get("page_name", ""),
                            "hierarchy": (
                                f"{metadata.get('hierarchy', '')} > {i}"
                            ),
                        },
                    }
                )
        return cleaned_data
