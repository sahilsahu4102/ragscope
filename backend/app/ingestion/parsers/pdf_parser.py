"""
RAGScope — PDF Parser

Layout-aware PDF parsing with fallback chain: PyMuPDF → plain text extraction.
Preserves element types (titles, paragraphs, tables) and page structure.
"""

from pathlib import Path

import fitz  # PyMuPDF
import structlog

from app.ingestion.parsers.base import (
    BaseParser,
    ElementType,
    ParsedDocument,
    ParsedElement,
)

logger = structlog.get_logger()


class PDFParser(BaseParser):
    """
    PDF parser using PyMuPDF with layout-aware element extraction.

    Extracts text blocks with position info, classifies them by font size
    into titles/headings/paragraphs, and preserves reading order.
    """

    def supported_types(self) -> list[str]:
        return ["application/pdf"]

    async def parse(self, file_path: str, mime_type: str) -> ParsedDocument:
        """Parse a PDF file into structured elements."""
        path = Path(file_path)
        logger.info("Parsing PDF", filename=path.name)

        try:
            return await self._parse_with_layout(path)
        except Exception as e:
            logger.warning("Layout parsing failed, falling back to plain text", error=str(e))
            return await self._parse_plain(path)

    async def _parse_with_layout(self, path: Path) -> ParsedDocument:
        """Layout-aware parsing — classifies blocks by font size."""
        doc = fitz.open(str(path))
        elements: list[ParsedElement] = []
        font_sizes: list[float] = []

        # First pass: collect all font sizes to determine thresholds
        for page in doc:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span["text"].strip():
                                font_sizes.append(span["size"])

        if not font_sizes:
            doc.close()
            return await self._parse_plain(path)

        # Determine thresholds
        avg_size = sum(font_sizes) / len(font_sizes)
        title_threshold = avg_size * 1.4
        heading_threshold = avg_size * 1.15

        # Second pass: extract elements with classification
        current_section: str | None = None

        # doc.pages() is PyMuPDF's documented page generator. Iterating `doc`
        # directly only works via the legacy __getitem__ sequence protocol —
        # fitz.Document defines no __iter__.
        for page_num, page in enumerate(doc.pages(), 1):
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            for block in blocks:
                if block.get("type") == 0:  # Text block
                    block_text = ""
                    max_font_size = 0.0
                    is_bold = False

                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span["text"]
                            max_font_size = max(max_font_size, span["size"])
                            if "bold" in span.get("font", "").lower():
                                is_bold = True
                        block_text += line_text.strip() + "\n"

                    block_text = block_text.strip()
                    if not block_text or len(block_text) < 2:
                        continue

                    # Classify element type by font size
                    if max_font_size >= title_threshold:
                        el_type = ElementType.TITLE
                        current_section = block_text
                    elif max_font_size >= heading_threshold or (is_bold and len(block_text) < 200):
                        el_type = ElementType.HEADING
                        current_section = block_text
                    else:
                        el_type = ElementType.PARAGRAPH

                    elements.append(
                        ParsedElement(
                            content=block_text,
                            element_type=el_type,
                            page_number=page_num,
                            section_path=current_section,
                            metadata={"font_size": max_font_size, "is_bold": is_bold},
                        )
                    )

                elif block.get("type") == 1:  # Image block
                    elements.append(
                        ParsedElement(
                            content="[Image]",
                            element_type=ElementType.IMAGE_CAPTION,
                            page_number=page_num,
                            section_path=current_section,
                        )
                    )

        page_count = len(doc)
        doc.close()

        logger.info(
            "PDF parsed with layout",
            filename=path.name,
            elements=len(elements),
            pages=page_count,
        )

        return ParsedDocument(
            filename=path.name,
            elements=elements,
            page_count=page_count,
            metadata={"parser": "pymupdf_layout", "path": str(path)},
        )

    async def _parse_plain(self, path: Path) -> ParsedDocument:
        """Fallback: simple page-by-page text extraction."""
        doc = fitz.open(str(path))
        elements: list[ParsedElement] = []

        # doc.pages() is PyMuPDF's documented page generator. Iterating `doc`
        # directly only works via the legacy __getitem__ sequence protocol —
        # fitz.Document defines no __iter__.
        for page_num, page in enumerate(doc.pages(), 1):
            text = page.get_text("text").strip()
            if text:
                elements.append(
                    ParsedElement(
                        content=text,
                        element_type=ElementType.PARAGRAPH,
                        page_number=page_num,
                    )
                )

        page_count = len(doc)
        doc.close()

        logger.info(
            "PDF parsed (plain fallback)",
            filename=path.name,
            elements=len(elements),
            pages=page_count,
        )

        return ParsedDocument(
            filename=path.name,
            elements=elements,
            page_count=page_count,
            metadata={"parser": "pymupdf_plain", "path": str(path)},
        )
