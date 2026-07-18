"""
RAGScope — Recursive Text Chunker

Configurable recursive character text splitter with metadata preservation.
The workhorse chunker — a strong baseline that placed first (69% accuracy) 
in a Feb-2026 Vecta benchmark of 7 strategies.
"""

import structlog
from dataclasses import dataclass, field

from app.ingestion.parsers.base import ParsedDocument, ParsedElement

logger = structlog.get_logger()


@dataclass
class TextChunk:
    """A chunk of text ready for embedding."""
    content: str
    chunk_index: int
    token_count: int
    document_name: str
    element_type: str | None = None
    page_number: int | None = None
    section_path: str | None = None
    parent_content: str | None = None
    metadata: dict = field(default_factory=dict)


class RecursiveChunker:
    """
    Recursive character text splitter.
    
    Splits text hierarchically using separators (paragraphs → sentences → words),
    with configurable chunk size and overlap.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk_document(self, parsed_doc: ParsedDocument) -> list[TextChunk]:
        """Chunk a parsed document, preserving element metadata."""
        all_chunks: list[TextChunk] = []
        chunk_index = 0

        for element in parsed_doc.elements:
            if not element.content.strip():
                continue

            # Skip very short elements (e.g., page numbers)
            if len(element.content.strip()) < 10:
                continue

            text_pieces = self._split_text(element.content, self.separators)

            for piece in text_pieces:
                piece = piece.strip()
                if not piece:
                    continue

                token_count = self._estimate_tokens(piece)

                all_chunks.append(TextChunk(
                    content=piece,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    document_name=parsed_doc.filename,
                    element_type=element.element_type.value if element.element_type else None,
                    page_number=element.page_number,
                    section_path=element.section_path,
                    metadata={
                        **element.metadata,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "chunker": "recursive",
                    },
                ))
                chunk_index += 1

        logger.info(
            "Document chunked",
            filename=parsed_doc.filename,
            chunks=len(all_chunks),
            avg_tokens=sum(c.token_count for c in all_chunks) // max(len(all_chunks), 1),
        )

        return all_chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using hierarchical separators."""
        final_chunks: list[str] = []

        # Find the appropriate separator
        separator = separators[-1]  # Default to last (empty string)
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split by the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks of appropriate size
        current_chunk: list[str] = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length + len(separator) > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)

                    # If chunk is still too large, recurse with next separator
                    if len(chunk_text) > self.chunk_size and len(separators) > 1:
                        final_chunks.extend(
                            self._split_text(chunk_text, separators[1:])
                        )
                    else:
                        final_chunks.append(chunk_text)

                    # Handle overlap: keep some trailing splits
                    overlap_chunks: list[str] = []
                    overlap_length = 0
                    for prev_split in reversed(current_chunk):
                        if overlap_length + len(prev_split) > self.chunk_overlap:
                            break
                        overlap_chunks.insert(0, prev_split)
                        overlap_length += len(prev_split)
                    current_chunk = overlap_chunks
                    current_length = overlap_length

            current_chunk.append(split)
            current_length += split_length + len(separator)

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > self.chunk_size and len(separators) > 1:
                final_chunks.extend(
                    self._split_text(chunk_text, separators[1:])
                )
            else:
                final_chunks.append(chunk_text)

        return final_chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (~4 chars per token for English)."""
        return max(1, len(text) // 4)
