"""
RAGScope — Query Transformation

Pre-retrieval query optimization:
  1. Rewrite — rephrase ambiguous queries into retrieval-friendly form
  2. HyDE — Hypothetical Document Embeddings (generate a fake answer, embed that)
  3. Decomposition — break complex multi-hop questions into sub-queries
"""

import re

import structlog

from app.config import settings
from app.http_client import get_http_client
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("retrieval")


_FILLER = frozenset(
    "what which when where does were with have that this from much many used uses "
    "than they their there about into also".split()
)


def _terms(text: str) -> set[str]:
    """Words of four or more letters, minus question filler."""
    return set(re.findall(r"[a-z0-9]{4,}", text.lower())) - _FILLER


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
        """Make a single Ollama LLM call using the shared connection pool."""
        client = get_http_client()
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
            # One lookup per fact asked for. The old prompt asked for "2-4"
            # sub-questions, so a two-part question often came back as three or
            # four overlapping ones, and the merge only has room for one chunk
            # per hop. The example uses placeholders: given a concrete example
            # (BERT's batch size), llama3.2:3b returned it verbatim for 4 of 10
            # multi-hop dev questions.
            prompt = (
                "Split the question into the separate searches needed to answer it: "
                "one short, self-contained question per fact it asks for. A question "
                "that asks for one fact needs one search.\n\n"
                "Example:\n"
                "Question: What is the X of A, and what is the Y of B?\n"
                "1. What is the X of A?\n"
                "2. What is the Y of B?\n\n"
                f"Question: {query}\n"
                "Output only the numbered searches."
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

            # A sub-question sharing no content word with the question is the
            # model drifting (or echoing the example); retrieving it adds noise.
            sub_queries = [q for q in sub_queries if _terms(q) & _terms(query)] or [query]

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
