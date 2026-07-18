"""
RAGScope — LLM Generation with Grounded Citations

Generates answers using Ollama (self-hosted LLM) with grounded prompts
that enforce citation from retrieved context. Supports SSE streaming.
"""

import json
import time
from collections.abc import AsyncGenerator

import httpx
import structlog
from opentelemetry import trace as otel_trace

from app.config import settings
from app.observability.cost import calculate_cost
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("generation")


GROUNDED_QA_PROMPT = """You are RAGScope, an AI assistant that answers questions strictly based on the provided context.

## Rules
1. Answer ONLY using information from the context below. Do not use prior knowledge.
2. If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this question."
3. Cite your sources using [1], [2], etc. corresponding to the chunk numbers below.
4. Be concise but thorough. Include all relevant details from the context.
5. If multiple chunks support the same point, cite all of them.

## Context (Retrieved Chunks)
{context}

## Question
{question}

## Instructions
Provide a well-structured answer with inline citations [1], [2], etc. At the end, list the sources used."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into numbered context for the prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("document_name", "Unknown")
        page = chunk.get("metadata", {}).get("page_number", "?")
        score = chunk.get("dense_score", chunk.get("rrf_score", 0))
        context_parts.append(
            f"[{i}] (Source: {source}, Page: {page}, Relevance: {score:.3f})\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


class Generator:
    """
    LLM generation service using self-hosted Ollama.

    Produces grounded answers with structured citations,
    tracked with OpenInference LLM spans (tokens, latency, cost).
    """

    def __init__(self):
        self.model = settings.ollama_model
        self.base_url = settings.ollama_base_url

    async def generate(
        self,
        question: str,
        chunks: list[dict],
    ) -> dict:
        """
        Generate a grounded answer with citations.

        Returns: {"answer": str, "citations": list, "tokens": int, "latency_ms": float}
        """
        context = format_context(chunks)
        prompt = GROUNDED_QA_PROMPT.format(context=context, question=question)

        start = time.perf_counter()

        with create_span(
            tracer,
            "llm_generate",
            "LLM",
            {
                "gen_ai.system": "ollama",
                "gen_ai.request.model": self.model,
                "gen_ai.operation.name": "chat",
            },
        ):
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_predict": 2048,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

            # Attach token usage + cost to the LLM span for trace/analytics.
            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)
            otel_trace.get_current_span().set_attributes(
                {
                    "gen_ai.usage.input_tokens": input_tokens,
                    "gen_ai.usage.output_tokens": output_tokens,
                    "gen_ai.usage.total_tokens": input_tokens + output_tokens,
                    "cost.usd": calculate_cost(self.model, input_tokens, output_tokens),
                }
            )

        latency_ms = (time.perf_counter() - start) * 1000
        answer = data.get("response", "")
        total_tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        # Extract citation indices from the answer
        citations = self._extract_citations(answer, chunks)

        logger.info(
            "Generation complete",
            model=self.model,
            tokens=total_tokens,
            latency_ms=round(latency_ms, 1),
            citations=len(citations),
        )

        return {
            "answer": answer,
            "citations": citations,
            "tokens_used": total_tokens,
            "cost_usd": calculate_cost(
                self.model,
                data.get("prompt_eval_count", 0),
                data.get("eval_count", 0),
            ),
            "latency_ms": round(latency_ms, 1),
        }

    async def generate_stream(
        self,
        question: str,
        chunks: list[dict],
    ) -> AsyncGenerator[str, None]:
        """
        Stream the generation via SSE-compatible token stream.

        Yields JSON strings: {"token": "...", "done": false}
        """
        context = format_context(chunks)
        prompt = GROUNDED_QA_PROMPT.format(context=context, question=question)

        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 2048,
                    },
                },
            ) as response,
        ):
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield json.dumps(
                        {
                            "token": data.get("response", ""),
                            "done": data.get("done", False),
                        }
                    )

    @staticmethod
    def _extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
        """Extract citation references [1], [2], etc. from the answer."""
        import re

        citations = []
        seen = set()

        # Find all [N] references in the answer
        refs = re.findall(r"\[(\d+)\]", answer)
        for ref in refs:
            idx = int(ref) - 1  # Convert to 0-indexed
            if 0 <= idx < len(chunks) and idx not in seen:
                seen.add(idx)
                chunk = chunks[idx]
                citations.append(
                    {
                        "chunk_id": chunk.get("chunk_id", ""),
                        "document_name": chunk.get("document_name", ""),
                        "content_snippet": chunk["content"][:200] + "..."
                        if len(chunk["content"]) > 200
                        else chunk["content"],
                        "score": chunk.get("dense_score", 0),
                        "page_number": chunk.get("metadata", {}).get("page_number"),
                    }
                )

        return citations
