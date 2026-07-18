"""
RAGScope — Query Transformation

Pre-retrieval query optimization:
  1. Rewrite — rephrase ambiguous queries into retrieval-friendly form
  2. HyDE — Hypothetical Document Embeddings (generate a fake answer, embed that)
  3. Decomposition — break complex multi-hop questions into sub-queries
"""

import httpx
import structlog

from app.config import settings
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("retrieval")


class QueryTransformer:
    """
    Pre-retrieval query optimization pipeline.

    Applied before the retrieval step to improve recall.
    Each method takes a raw user query and returns transformed query(ies).
    """

    def __init__(self):
        self.model = settings.ollama_model
        self.base_url = settings.ollama_base_url

    async def _llm_call(self, prompt: str) -> str:
        """Make a single Ollama LLM call."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512},
                },
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()

    async def rewrite(self, query: str) -> str:
        """
        Rewrite a user query into a retrieval-optimized form.

        Removes ambiguity, adds context, expands acronyms.
        """
        with create_span(
            tracer,
            "query_rewrite",
            "CHAIN",
            {
                "transform.type": "rewrite",
                "transform.original_query": query,
            },
        ):
            prompt = (
                "You are a query optimization expert. Rewrite the following user "
                "question to be more specific and effective for document retrieval.\n\n"
                "Rules:\n"
                "- Keep the same intent\n"
                "- Expand abbreviations and acronyms\n"
                "- Add relevant technical context\n"
                "- Remove filler words\n"
                "- Output ONLY the rewritten query, nothing else\n\n"
                f"Original query: {query}\n\n"
                "Rewritten query:"
            )
            rewritten = await self._llm_call(prompt)
            logger.info("Query rewritten", original=query, rewritten=rewritten)
            return rewritten or query

    async def hyde(self, query: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE).

        Generate a hypothetical passage that would answer the query,
        then use that passage for embedding-based retrieval instead
        of the original query. Improves recall for vague questions.

        Reference: Gao et al., 2023. "Precise Zero-Shot Dense Retrieval without Relevance Labels"
        """
        with create_span(
            tracer,
            "query_hyde",
            "CHAIN",
            {
                "transform.type": "hyde",
            },
        ):
            prompt = (
                "You are a technical document expert. Write a short passage (3-5 sentences) "
                "that would directly answer the following question. Write as if you are "
                "quoting from a real document.\n\n"
                f"Question: {query}\n\n"
                "Hypothetical passage:"
            )
            hypothetical = await self._llm_call(prompt)
            logger.info("HyDE generated", query=query, hyde_length=len(hypothetical))
            return hypothetical or query

    async def decompose(self, query: str) -> list[str]:
        """
        Decompose a complex multi-hop question into sub-queries.

        E.g., "Compare X and Y's approach to Z" →
              ["What is X's approach to Z?", "What is Y's approach to Z?"]
        """
        with create_span(
            tracer,
            "query_decompose",
            "CHAIN",
            {
                "transform.type": "decompose",
            },
        ):
            prompt = (
                "You are a research assistant. Break the following complex question "
                "into 2-4 simpler sub-questions that, when answered individually, "
                "would provide a complete answer.\n\n"
                "Rules:\n"
                "- Each sub-question should be self-contained\n"
                "- Number each sub-question\n"
                "- Output ONLY the numbered sub-questions\n\n"
                f"Complex question: {query}\n\n"
                "Sub-questions:"
            )
            result = await self._llm_call(prompt)

            # Parse numbered sub-questions
            sub_queries = []
            for line in result.split("\n"):
                line = line.strip()
                if line and any(line.startswith(f"{i}") for i in range(1, 10)):
                    # Remove numbering prefix
                    cleaned = line.lstrip("0123456789.)-: ")
                    if cleaned:
                        sub_queries.append(cleaned)

            if not sub_queries:
                sub_queries = [query]

            logger.info(
                "Query decomposed",
                original=query,
                sub_queries=len(sub_queries),
            )
            return sub_queries

    async def transform(
        self,
        query: str,
        method: str = "rewrite",
    ) -> str | list[str]:
        """
        Apply a query transformation method.

        Args:
            query: Original user query
            method: One of "rewrite", "hyde", "decompose", "none"

        Returns:
            Transformed query (str) or list of sub-queries (for decompose)
        """
        match method:
            case "rewrite":
                return await self.rewrite(query)
            case "hyde":
                return await self.hyde(query)
            case "decompose":
                return await self.decompose(query)
            case "none" | _:
                return query
