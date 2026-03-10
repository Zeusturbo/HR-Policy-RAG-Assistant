"""PDF loading and page-level text extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import fitz

from src.utils import PageRecord, is_pdf_file, normalize_text

logger = logging.getLogger(__name__)


def load_pdf_pages(pdf_path: str | Path) -> list[PageRecord]:
    """Load a single PDF and return page-level extracted text records.

    Args:
        pdf_path: Path to a PDF file.

    Returns:
        A list of PageRecord entries for pages that contain non-empty text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a PDF file.
        RuntimeError: If the PDF cannot be opened.
    """
    path = Path(pdf_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file() or not is_pdf_file(path):
        raise ValueError(f"Expected a PDF file, got: {path}")

    records: list[PageRecord] = []
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{path}': {exc}") from exc

    with doc:
        for page_idx, page in enumerate(doc, start=1):
            try:
                text = normalize_text(page.get_text("text"))
            except Exception:
                logger.exception(
                    "Failed to read text from %s page %s", path.name, page_idx
                )
                continue

            if not text:
                continue

            records.append(
                PageRecord(text=text, page_number=page_idx, source_file=path.name)
            )

    return records


def iter_pdf_files(folder_path: str | Path, recursive: bool = False) -> Iterable[Path]:
    """Yield PDF files from a folder in deterministic order."""
    folder = Path(folder_path).expanduser()
    pattern = "**/*.pdf" if recursive else "*.pdf"
    yield from sorted(folder.glob(pattern))


def load_pdfs_from_folder(
    folder_path: str | Path, recursive: bool = False
) -> list[PageRecord]:
    """Load all PDFs from a folder and return combined page records."""
    folder = Path(folder_path).expanduser()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected a directory path, got: {folder}")

    records: list[PageRecord] = []
    pdf_files = list(iter_pdf_files(folder, recursive=recursive))
    if not pdf_files:
        logger.warning("No PDF files found in folder: %s", folder)
        return records

    for pdf_file in pdf_files:
        try:
            records.extend(load_pdf_pages(pdf_file))
        except Exception:
            logger.exception("Skipping unreadable PDF: %s", pdf_file)

    return records


def load_pdfs(path: str | Path, recursive: bool = False) -> list[PageRecord]:
    """Load one PDF file or all PDFs in a folder."""
    target = Path(path).expanduser()
    if target.is_file():
        return load_pdf_pages(target)
    return load_pdfs_from_folder(target, recursive=recursive)
