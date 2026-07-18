"""
RAGScope — Hierarchical Chunker (Parent-Child)

Creates two-tier chunks: large parent chunks for LLM context,
small child chunks for precise vector search.

When a child chunk matches during retrieval, we fetch the full
parent for generation — best of both worlds.
"""

import structlog
from dataclasses import dataclass, field

from app.ingestion.parsers.base import ParsedDocument
from app.ingestion.chunkers.recursive_chunker import RecursiveChunker, TextChunk

logger = structlog.get_logger()


@dataclass
class HierarchicalChunk:
    """A chunk with parent-child relationship."""
    content: str
    chunk_index: int
    token_count: int
    document_name: str
    level: str  # "parent" or "child"
    parent_index: int | None = None
    parent_content: str | None = None
    children_indices: list[int] = field(default_factory=list)
    element_type: str | None = None
    page_number: int | None = None
    section_path: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_text_chunk(self) -> TextChunk:
        """Convert to TextChunk for embedding pipeline compatibility."""
        return TextChunk(
            content=self.content,
            chunk_index=self.chunk_index,
            token_count=self.token_count,
            document_name=self.document_name,
            element_type=self.element_type,
            page_number=self.page_number,
            section_path=self.section_path,
            parent_content=self.parent_content,
            metadata={
                **self.metadata,
                "level": self.level,
                "parent_index": self.parent_index,
                "children_indices": self.children_indices,
            },
        )


class HierarchicalChunker:
    """
    Two-tier hierarchical chunker.

    1. Split document into large parent chunks (e.g. 1024 chars)
    2. Split each parent into small child chunks (e.g. 256 chars)
    3. Child chunks carry a reference to their parent

    During retrieval: search by child → return parent for generation.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1024,
        parent_overlap: int = 100,
        child_chunk_size: int = 256,
        child_overlap: int = 30,
    ):
        self.parent_chunker = RecursiveChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
        )
        self.child_chunker = RecursiveChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
        )
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

    def chunk_document(self, parsed_doc: ParsedDocument) -> list[HierarchicalChunk]:
        """
        Create hierarchical parent-child chunks.

        Returns both parent and child chunks. Child chunks are embedded;
        parent chunks are fetched at generation time for full context.
        """
        # Step 1: Create parent chunks
        parent_text_chunks = self.parent_chunker.chunk_document(parsed_doc)
        all_hierarchical: list[HierarchicalChunk] = []
        global_index = 0

        for parent_tc in parent_text_chunks:
            parent_idx = global_index

            parent_hc = HierarchicalChunk(
                content=parent_tc.content,
                chunk_index=parent_idx,
                token_count=parent_tc.token_count,
                document_name=parent_tc.document_name,
                level="parent",
                element_type=parent_tc.element_type,
                page_number=parent_tc.page_number,
                section_path=parent_tc.section_path,
                metadata={
                    **parent_tc.metadata,
                    "parent_chunk_size": self.parent_chunk_size,
                    "child_chunk_size": self.child_chunk_size,
                    "chunker": "hierarchical",
                },
            )
            all_hierarchical.append(parent_hc)
            global_index += 1

            # Step 2: Split parent into child chunks
            child_texts = self.child_chunker._split_text(
                parent_tc.content,
                self.child_chunker.separators,
            )

            child_indices = []
            for child_text in child_texts:
                child_text = child_text.strip()
                if not child_text or len(child_text) < 10:
                    continue

                child_idx = global_index
                child_indices.append(child_idx)

                child_hc = HierarchicalChunk(
                    content=child_text,
                    chunk_index=child_idx,
                    token_count=max(1, len(child_text) // 4),
                    document_name=parent_tc.document_name,
                    level="child",
                    parent_index=parent_idx,
                    parent_content=parent_tc.content,
                    element_type=parent_tc.element_type,
                    page_number=parent_tc.page_number,
                    section_path=parent_tc.section_path,
                    metadata={
                        **parent_tc.metadata,
                        "chunker": "hierarchical",
                        "level": "child",
                    },
                )
                all_hierarchical.append(child_hc)
                global_index += 1

            # Update parent with children references
            parent_hc.children_indices = child_indices

        # Count stats
        parents = sum(1 for c in all_hierarchical if c.level == "parent")
        children = sum(1 for c in all_hierarchical if c.level == "child")

        logger.info(
            "Hierarchical chunking complete",
            filename=parsed_doc.filename,
            parents=parents,
            children=children,
            total=len(all_hierarchical),
        )

        return all_hierarchical
