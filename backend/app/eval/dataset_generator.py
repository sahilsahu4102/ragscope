"""
RAGScope — Dataset Generator

Generates synthetic golden QA sets from ingested documents.
Uses LLM to create diverse question types:
- Single-hop factual (direct answer in one chunk)
- Multi-hop reasoning (requires combining multiple chunks)
- Analytical (requires interpretation/synthesis)
"""

from __future__ import annotations

import uuid

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk, Dataset, Document
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("eval")

QA_GENERATION_PROMPT = """You are a QA dataset generator for evaluating RAG systems.

Given the following document chunks, generate {num_questions} diverse question-answer pairs.

## Requirements
- Each question should be answerable from the provided chunks
- Include a mix of question types:
  * Factual (direct answer from one chunk)
  * Reasoning (requires combining info from multiple chunks)
  * Analytical (requires interpretation)
- Each answer should be 1-3 sentences and cite which chunk(s) it came from
- Questions should be natural and varied

## Chunks
{chunks}

## Output Format (one per line, JSON)
{{"question": "...", "answer": "...", "type": "factual|reasoning|analytical", "source_chunks": [0, 1]}}

Generate exactly {num_questions} question-answer pairs, one JSON per line:"""


class DatasetGenerator:
    """Generates synthetic evaluation datasets from ingested documents."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.model = settings.ollama_model
        self.base_url = settings.ollama_base_url

    async def _call_llm(self, prompt: str) -> str:
        """Call Ollama for QA generation."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 2048},
                },
            )
            response.raise_for_status()
            return response.json().get("response", "")

    async def generate(
        self,
        name: str,
        document_ids: list[str] | None = None,
        num_questions: int = 20,
        version: str = "1.0",
    ) -> Dataset:
        """
        Generate a golden evaluation dataset from ingested documents.

        Args:
            name: Human-readable dataset name
            document_ids: Specific docs to use (None = all)
            num_questions: Target number of QA pairs
            version: Dataset version string
        """
        with create_span(
            tracer,
            "generate_dataset",
            "CHAIN",
            {
                "dataset.name": name,
                "dataset.target_questions": num_questions,
            },
        ):
            # 1. Load chunks from database
            query = select(Chunk).join(Document)
            if document_ids:
                query = query.where(Document.id.in_([uuid.UUID(did) for did in document_ids]))
            query = query.order_by(Chunk.chunk_index).limit(50)

            result = await self.db.execute(query)
            chunks = result.scalars().all()

            if not chunks:
                raise ValueError("No chunks found for dataset generation")

            # 2. Format chunks for the prompt
            chunk_texts = []
            for i, chunk in enumerate(chunks):
                chunk_texts.append(
                    f"[Chunk {i}] (from: {chunk.document.filename})\n{chunk.content}"
                )
            chunks_str = "\n\n---\n\n".join(chunk_texts)

            # 3. Generate QA pairs via LLM
            prompt = QA_GENERATION_PROMPT.format(
                num_questions=num_questions,
                chunks=chunks_str,
            )

            import json

            response = await self._call_llm(prompt)
            samples = []

            for line in response.strip().split("\n"):
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    qa = json.loads(line)
                    # Map source chunks to actual content for gold contexts
                    gold_contexts = []
                    for idx in qa.get("source_chunks", []):
                        if 0 <= idx < len(chunks):
                            gold_contexts.append(chunks[idx].content)

                    samples.append(
                        {
                            "question": qa["question"],
                            "gold_answer": qa["answer"],
                            "gold_contexts": gold_contexts,
                            "question_type": qa.get("type", "factual"),
                            "metadata": {
                                "source_chunk_indices": qa.get("source_chunks", []),
                            },
                        }
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Failed to parse QA pair", line=line, error=str(e))
                    continue

            if not samples:
                raise ValueError("LLM failed to generate valid QA pairs")

            # 4. Store as Dataset
            source_doc_ids = list({str(c.document_id) for c in chunks})
            dataset = Dataset(
                name=name,
                version=version,
                description=(
                    f"Auto-generated from {len(source_doc_ids)} documents, {len(samples)} QA pairs"
                ),
                sample_count=len(samples),
                samples=samples,
                source_documents=source_doc_ids,
            )
            self.db.add(dataset)
            await self.db.commit()
            await self.db.refresh(dataset)

            logger.info(
                "Dataset generated",
                dataset_id=str(dataset.id),
                name=name,
                samples=len(samples),
            )

            return dataset
