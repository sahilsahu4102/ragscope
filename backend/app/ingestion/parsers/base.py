"""
RAGScope — Base Parser Interface

Abstract parser with a factory for routing by MIME type.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ElementType(str, Enum):
    """Types of document elements preserved during parsing."""

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    IMAGE_CAPTION = "image_caption"
    CODE = "code"
    FOOTER = "footer"
    HEADER = "header"
    PAGE_BREAK = "page_break"


@dataclass
class ParsedElement:
    """A single element extracted from a document."""

    content: str
    element_type: ElementType
    page_number: int | None = None
    section_path: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Result of parsing a document — preserves element hierarchy."""

    filename: str
    elements: list[ParsedElement]
    page_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all element content."""
        return "\n\n".join(el.content for el in self.elements)

    @property
    def element_count(self) -> int:
        return len(self.elements)


class BaseParser(ABC):
    """Abstract base for document parsers."""

    @abstractmethod
    async def parse(self, file_path: str, mime_type: str) -> ParsedDocument:
        """Parse a file and return structured elements."""
        ...

    @abstractmethod
    def supported_types(self) -> list[str]:
        """MIME types this parser supports."""
        ...
