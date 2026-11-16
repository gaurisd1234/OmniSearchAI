"""Format-specific text and metadata extraction for supported documents."""

from __future__ import annotations

from io import BytesIO
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from docx import Document as OpenDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from omnisearch_ai.ingestion.models import PageContent


@dataclass(frozen=True, slots=True)
class ExtractedContent:
    pages: tuple[PageContent, ...]
    metadata: dict[str, Any]


def normalize_text(text: str) -> str:
    """Normalize Unicode and line endings without collapsing meaningful layout."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _base_metadata(path: Path, file_type: str) -> dict[str, Any]:
    return {"file_name": path.name, "file_type": file_type, "title": path.stem}


def read_txt(path: Path, raw: bytes) -> ExtractedContent:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Windows-created text commonly uses a single-byte Windows encoding.
        text = raw.decode("cp1252")
    if "\x00" in text:
        raise ValueError("TXT file contains NUL bytes and appears to be binary")
    page_texts = text.replace("\r\n", "\n").replace("\r", "\n").split("\f")
    pages = tuple(
        PageContent(page_number=i, text=normalize_text(page))
        for i, page in enumerate(page_texts, start=1)
    )
    metadata = _base_metadata(path, "txt")
    metadata["encoding"] = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8-or-cp1252"
    return ExtractedContent(pages, metadata)


def read_pdf(path: Path, raw: bytes) -> ExtractedContent:
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as pdf:
            metadata = _base_metadata(path, "pdf")
            metadata.update({key: value for key, value in (pdf.metadata or {}).items() if value})
            pages = tuple(
                PageContent(page_number=i, text=normalize_text(page.get_text("text")))
                for i, page in enumerate(pdf, start=1)
            )
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc
    metadata["page_count"] = len(pages)
    return ExtractedContent(pages, metadata)


def _iter_docx_paragraphs(container: Any, seen_cells: set[int] | None = None):
    """Yield paragraphs in body/table order, including nested table content."""
    seen_cells = seen_cells if seen_cells is not None else set()
    for block in container.iter_inner_content():
        if isinstance(block, Paragraph):
            yield block
        elif isinstance(block, Table):
            for row in block.rows:
                for cell in row.cells:
                    cell_id = id(cell._tc)
                    if cell_id not in seen_cells:
                        seen_cells.add(cell_id)
                        yield from _iter_docx_paragraphs(cell, seen_cells)


def _docx_page_texts(document: Any) -> tuple[str, ...]:
    """Extract paragraph text and split only on explicit Word page breaks.

    python-docx does not paginate documents. Page numbers here reflect explicit
    page breaks and page-break-before settings; otherwise content is logical page 1.
    """
    page_texts: list[str] = [""]
    for paragraph in _iter_docx_paragraphs(document):
        if paragraph.paragraph_format.page_break_before and page_texts[-1].strip():
            page_texts.append("")
        for element in paragraph._p.iter():
            if element.tag == qn("w:t"):
                page_texts[-1] += element.text or ""
            elif element.tag == qn("w:tab"):
                page_texts[-1] += "\t"
            elif element.tag == qn("w:br"):
                break_type = element.get(qn("w:type"))
                if break_type == "page":
                    page_texts.append("")
                else:
                    page_texts[-1] += "\n"
            elif element.tag == qn("w:lastRenderedPageBreak"):
                page_texts.append("")
    return tuple(page_texts)


def read_docx(path: Path, raw: bytes) -> ExtractedContent:
    try:
        document = OpenDocument(BytesIO(raw))
        page_texts = _docx_page_texts(document)
        core = document.core_properties
    except Exception as exc:
        raise ValueError(f"Could not read DOCX: {exc}") from exc
    pages = tuple(
        PageContent(page_number=i, text=normalize_text(page))
        for i, page in enumerate(page_texts, start=1)
    )
    metadata = _base_metadata(path, "docx")
    metadata.update(
        {
            key: value
            for key, value in {
                "title": core.title,
                "author": core.author,
                "subject": core.subject,
                "created": core.created.isoformat() if core.created else None,
                "modified": core.modified.isoformat() if core.modified else None,
            }.items()
            if value
        }
    )
    metadata["page_count"] = len(pages)
    return ExtractedContent(pages, metadata)


READERS = {".txt": read_txt, ".pdf": read_pdf, ".docx": read_docx}
