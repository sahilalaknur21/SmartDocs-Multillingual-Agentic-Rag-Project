# ingestion/pdf_loader.py
# WHY THIS EXISTS: Extracts clean text from any PDF — text, tables, images.
# pdfplumber handles text-heavy PDFs (GST notices, legal docs).
# Falls back gracefully if extraction partially fails.

import hashlib
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber
from langdetect import detect, LangDetectException


@dataclass
class ExtractedPage:
    """Represents a single extracted page from a PDF."""
    page_number: int
    text: str
    tables: list[list] = field(default_factory=list)
    char_count: int = 0
    extraction_method: str = "pdfplumber"


@dataclass
class ExtractedDocument:
    """Represents a fully extracted PDF document."""
    file_path: str
    file_name: str
    doc_hash: str
    pages: list[ExtractedPage]
    total_pages: int
    primary_language: str
    full_text: str
    extraction_success: bool
    error_message: Optional[str] = None


def compute_doc_hash(file_path: str) -> str:
    """
    Computes SHA-256 hash of PDF file for idempotency.
    If same file uploaded twice, hash matches → skip re-ingestion.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def detect_document_language(text: str) -> str:
    """
    Detects primary language of document from first 2000 characters.
    Used for document language badge in UI (LAW 17).

    Returns:
        ISO language code string e.g. "hi", "en", "ta"
    """
    sample = text[:2000].strip()
    if not sample:
        return "en"
    try:
        return detect(sample)
    except LangDetectException:
        return "en"


def extract_tables_from_page(page) -> list[list]:
    """
    Extracts tables from a pdfplumber page object.
    GST notices and legal documents are table-heavy.

    Returns:
        List of tables, each table is a list of rows
    """
    tables = []
    try:
        extracted = page.extract_tables()
        if extracted:
            for table in extracted:
                clean_table = [
                    [cell if cell is not None else "" for cell in row]
                    for row in table
                    if any(cell for cell in row if cell)
                ]
                if clean_table:
                    tables.append(clean_table)
    except Exception:
        pass
    return tables


def table_to_text(table: list[list]) -> str:
    """
    Converts extracted table to readable text format.
    Preserves table structure for context during retrieval.
    """
    if not table:
        return ""

    lines = []
    for row in table:
        row_text = " | ".join(str(cell).strip() for cell in row if str(cell).strip())
        if row_text:
            lines.append(row_text)

    return "\n".join(lines)


def load_pdf_pdfplumber(file_path: str) -> ExtractedDocument:
    """
    Primary PDF extraction using pdfplumber.
    Best for text-heavy PDFs: GST notices, legal agreements,
    insurance policies, government circulars.

    Args:
        file_path: Absolute path to PDF file

    Returns:
        ExtractedDocument with all pages extracted
    """
    path = Path(file_path)

    if not path.exists():
        return ExtractedDocument(
            file_path=file_path,
            file_name=path.name,
            doc_hash="",
            pages=[],
            total_pages=0,
            primary_language="en",
            full_text="",
            extraction_success=False,
            error_message=f"File not found: {file_path}",
        )

    doc_hash = compute_doc_hash(file_path)
    pages = []
    full_text_parts = []

    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text() or ""

                # Extract tables and convert to text
                tables = extract_tables_from_page(page)
                table_texts = []
                for table in tables:
                    table_text = table_to_text(table)
                    if table_text:
                        table_texts.append(table_text)

                # Combine text and tables
                combined_text = text
                if table_texts:
                    combined_text += "\n\n" + "\n\n".join(table_texts)

                # Clean basic artifacts
                combined_text = re.sub(r"\n{3,}", "\n\n", combined_text)
                combined_text = combined_text.strip()

                extracted_page = ExtractedPage(
                    page_number=page_num,
                    text=combined_text,
                    tables=tables,
                    char_count=len(combined_text),
                    extraction_method="pdfplumber",
                )
                pages.append(extracted_page)
                if combined_text:
                    full_text_parts.append(combined_text)

        full_text = "\n\n".join(full_text_parts)
        primary_language = detect_document_language(full_text)

        return ExtractedDocument(
            file_path=file_path,
            file_name=path.name,
            doc_hash=doc_hash,
            pages=pages,
            total_pages=total_pages,
            primary_language=primary_language,
            full_text=full_text,
            extraction_success=True,
        )

    except Exception as e:
        return ExtractedDocument(
            file_path=file_path,
            file_name=path.name,
            doc_hash=doc_hash,
            pages=pages,
            total_pages=len(pages),
            primary_language="en",
            full_text="\n\n".join(full_text_parts),
            extraction_success=False,
            error_message=str(e),
        )


def load_pdf(file_path: str) -> ExtractedDocument:
    """
    Main entry point for PDF loading.
    Uses pdfplumber as primary extractor.

    Args:
        file_path: Absolute path to PDF file

    Returns:
        ExtractedDocument
    """
    return load_pdf_pdfplumber(file_path)