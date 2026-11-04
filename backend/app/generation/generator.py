"""
RAGScope — LLM Generation with Grounded Citations

Generates answers using Ollama (self-hosted LLM) with grounded prompts
that enforce citation from retrieved context. Supports SSE streaming.
"""

import json
import time
from collections.abc import AsyncGenerator

import structlog
from opentelemetry import trace as otel_trace

from app.config import settings
from app.observability.cost import calculate_cost
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("generation")


# Until 2026-11 the prompt asked for "a well-structured answer" and a closing
# list of sources. The 3B model wrote 150-300 tokens per answer, and output
# length is most of the end-to-end time (docs/eval-results.md, Phase 9).
GROUNDED_QA_PROMPT = """You are RAGScope. Answer the question using only the numbered context passages.

## Context
{context}

## Question
{question}

## How to answer
- Start with the direct answer, in one to three sentences. Answer every part of the question.
- Use the exact numbers, names and units the context gives.
- Cite the passage after each claim with its number, like [1] or [2][3].
- If the context does not answer the question, reply only: "I don't have enough information in the provided documents to answer this question."
- If it answers only part, give that part and say which part the documents do not cover.
- No preamble and no list of sources."""


def format_context(chunks: list[dict], max_chunk_chars: int = 0) -> str:
    """Format retrieved chunks into numbered context for the prompt.

    max_chunk_chars > 0 truncates each chunk. Prefill size is what sets
    time-to-first-token, so this is the main TTFT dial on the generation side.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("document_name", "Unknown")
        page = chunk.get("metadata", {}).get("page_number", "?")
        content = chunk["content"]
        if max_chunk_chars and len(content) > max_chunk_chars:
            content = content[:max_chunk_chars]
        # No retrieval score: it is noise to the model and costs prompt tokens.
        context_parts.append(f"[{i}] (Source: {source}, Page: {page})\n{content}")
    return "\n\n---\n\n".join(context_parts)


class Generator:
    """
    LLM generation service using self-hosted Ollama.

    Produces grounded answers with structured citations,
    tracked with OpenInference LLM spans (tokens, latency, cost).
    """

    def __init__(
        self,
        model: str | None = None,
        num_predict: int | None = None,
        max_chunk_chars: int | None = None,
    ):
        """Overrides exist so a config sweep can vary generation without
        restarting the app or mutating global settings."""
        self.model = model or settings.ollama_model
        self.base_url = settings.ollama_base_url
        self.num_predict = (
            num_predict if num_predict is not None else settings.generation_num_predict
        )
        self.max_chunk_chars = (
            max_chunk_chars if max_chunk_chars is not None else settings.generation_max_chunk_chars
        )

    def seen_texts(self, chunks: list[dict]) -> list[str]:
        """Chunk texts as the prompt carries them, after max_chunk_chars truncation."""
        return [c["content"][: self.max_chunk_chars or None] for c in chunks]

    async def generate(
        self,
        question: str,
        chunks: list[dict],
    ) -> dict:
        """
        Generate a grounded answer with citations.

        Returns: {"answer": str, "citations": list, "tokens": int, "latency_ms": float}
        """
        context = format_context(chunks, self.max_chunk_chars)
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
            from app.http_client import get_http_client

            client = get_http_client()
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": settings.ollama_keep_alive_value,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": self.num_predict,
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
            # Ollama's own timings (ns -> ms): prefill sets time to first token,
            # decode is output length / generation speed.
            "prefill_ms": round(data.get("prompt_eval_duration", 0) / 1e6, 1),
            "decode_ms": round(data.get("eval_duration", 0) / 1e6, 1),
            "output_tokens": data.get("eval_count", 0),
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
        context = format_context(chunks, self.max_chunk_chars)
        prompt = GROUNDED_QA_PROMPT.format(context=context, question=question)

        from app.http_client import get_http_client

        # Stream from the shared pool. Only the response context is scoped to
        # this request — the client itself stays alive, so we reuse the existing
        # keep-alive connection instead of doing a fresh TCP handshake per query.
        client = get_http_client()

        async with (
            client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": settings.ollama_keep_alive_value,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": self.num_predict,
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
