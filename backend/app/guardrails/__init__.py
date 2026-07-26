"""
RAGScope — Guardrails (Phase 5)

Unified guardrails pipeline:
  - PII Redaction: Regex-based detection + redaction of PII in input/output
  - Injection Detection: Heuristic prompt injection detection
  - Hallucination Detection: LLM-based groundedness scoring
"""

from app.guardrails.hallucination import HallucinationDetector
from app.guardrails.injection import InjectionDetector
from app.guardrails.pii import PIIRedactor

__all__ = [
    "GuardrailsPipeline",
    "HallucinationDetector",
    "InjectionDetector",
    "PIIRedactor",
]


class GuardrailsPipeline:
    """
    Unified guardrails runner that applies all configured checks.

    Usage:
        guardrails = GuardrailsPipeline(
            enable_pii=True,
            enable_injection=True,
            enable_hallucination=True,
        )

        # Pre-generation: check input
        input_result = guardrails.check_input(query)
        if input_result["blocked"]:
            return 400, input_result["reason"]

        # Post-generation: check + redact output
        output_result = await guardrails.check_output(answer, context_chunks)
    """

    def __init__(
        self,
        enable_pii: bool = True,
        enable_injection: bool = True,
        enable_hallucination: bool = True,
        injection_threshold: float = 0.5,
        hallucination_threshold: float = 0.7,
    ):
        self.pii_redactor = PIIRedactor() if enable_pii else None
        self.injection_detector = (
            InjectionDetector(threshold=injection_threshold)
            if enable_injection
            else None
        )
        self.hallucination_detector = (
            HallucinationDetector(threshold=hallucination_threshold)
            if enable_hallucination
            else None
        )

    def check_input(self, query: str) -> dict:
        """Run pre-generation guardrails on user query.

        Returns:
            {
                "blocked": bool,
                "reason": str | None,
                "redacted_query": str,
                "pii_findings": list,
                "injection_result": dict | None,
            }
        """
        redacted_query = query
        pii_findings: list = []
        injection_result = None

        # PII check on input
        if self.pii_redactor:
            redacted_query, pii_findings = self.pii_redactor.redact(query)

        # Injection detection
        if self.injection_detector:
            injection_result = self.injection_detector.detect(redacted_query)
            if injection_result["is_injection"]:
                return {
                    "blocked": True,
                    "reason": (
                        f"Prompt injection detected "
                        f"(confidence: {injection_result['confidence']}, "
                        f"risk: {injection_result['risk_level']})"
                    ),
                    "redacted_query": redacted_query,
                    "pii_findings": pii_findings,
                    "injection_result": injection_result,
                }

        return {
            "blocked": False,
            "reason": None,
            "redacted_query": redacted_query,
            "pii_findings": pii_findings,
            "injection_result": injection_result,
        }

    def redact_output(self, answer: str) -> dict:
        """Fast, synchronous half of the output guardrails.

        PII redaction is regex-only and mutates the answer, so it must run
        before the response is sent. Split out from check_output() so the
        LLM-based groundedness check can run after the response is flushed
        (it measured ~8.9s on the critical path).

        Returns:
            {"redacted_answer": str, "pii_findings": list}
        """
        if not self.pii_redactor:
            return {"redacted_answer": answer, "pii_findings": []}

        redacted_answer, pii_findings = self.pii_redactor.redact(answer)
        return {"redacted_answer": redacted_answer, "pii_findings": pii_findings}

    async def score_groundedness(
        self,
        answer: str,
        context_chunks: list[dict] | None = None,
    ) -> dict | None:
        """Slow, LLM-based half of the output guardrails.

        Safe to run off the critical path — it observes the answer rather than
        modifying it. Returns None when disabled or when there is no context.
        """
        if not self.hallucination_detector or not context_chunks:
            return None

        return await self.hallucination_detector.detect(
            answer=answer,
            context_chunks=context_chunks,
        )

    async def check_output(
        self,
        answer: str,
        context_chunks: list[dict] | None = None,
    ) -> dict:
        """Run post-generation guardrails on the answer.

        Returns:
            {
                "redacted_answer": str,
                "pii_findings": list,
                "hallucination_result": dict | None,
            }
        """
        redacted_answer = answer
        pii_findings: list = []
        hallucination_result = None

        # PII redaction on output
        if self.pii_redactor:
            redacted_answer, pii_findings = self.pii_redactor.redact(answer)

        # Hallucination check
        if self.hallucination_detector and context_chunks:
            hallucination_result = await self.hallucination_detector.detect(
                answer=redacted_answer,
                context_chunks=context_chunks,
            )

        return {
            "redacted_answer": redacted_answer,
            "pii_findings": pii_findings,
            "hallucination_result": hallucination_result,
        }
