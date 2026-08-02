"""
RAGScope — Golden dataset revision (v1 -> v2)

golden-v1 was LLM-generated and then reviewed by hand. Seven samples were
flagged as unusable for evaluation:

  Revised (question was answerable but badly posed):
    1  "recommended size" mischaracterised a stated 1MB-4GB range
    2  "differ in terms of parameters" was too vague to grade
    3  asked for a "minimum number of failures" the source never states
    21 referenced "Figure 6", which a retriever cannot resolve, and carried
       a second gold context from an unrelated section
    25 was a yes/no question — binary answers make faithfulness scoring noisy

  Dropped (cannot be made defensible without inventing information):
    13 pulled a single number out of a table whose column headers are not in
       the retrieved chunk, so no unambiguous gold answer exists
    26 duplicated 25's source passage and asked a vaguer version of it

Run:
    python -m app.scripts.revise_golden
"""

from __future__ import annotations

import io
import json
from pathlib import Path

SRC = Path("eval-datasets/golden-v1.jsonl")
DST = Path("eval-datasets/golden-v2.jsonl")

# 1-indexed line numbers in golden-v1.
DROP = {13, 26}

REVISIONS: dict[int, dict] = {
    1: {
        "question": (
            "What is the range of per-GPU model state size saved during Llama 3 checkpointing?"
        ),
        "gold_answer": ("Each GPU's model state ranges from 1 MB to 4 GB per GPU."),
        "question_type": "factual",
    },
    2: {
        "question": (
            "In the Transformer's position-wise feed-forward network, how do the "
            "linear transformations vary across positions compared to across layers?"
        ),
        "gold_answer": (
            "They are the same across different positions but use different "
            "parameters from layer to layer, equivalently described as two "
            "convolutions with kernel size 1."
        ),
        "question_type": "reasoning",
    },
    3: {
        "question": (
            "Why does the synchronous nature of Llama 3's 16K-GPU training make "
            "it less fault-tolerant?"
        ),
        "gold_answer": ("Because a single GPU failure may require a restart of the entire job."),
        "question_type": "reasoning",
    },
    21: {
        "question": (
            "Why did the Llama 3 authors modify their pipeline parallelism "
            "schedule to allow setting N flexibly?"
        ),
        "gold_answer": (
            "So an arbitrary number of micro-batches can run in each batch, "
            "including fewer micro-batches than the number of stages when there "
            "is a batch size limit at large scale."
        ),
        "question_type": "analytical",
        # v1 attached a second, unrelated gold context (a Llama Guard passage).
        # Keep only contexts that actually discuss the pipeline schedule.
        "context_filter": "pipeline schedule",
    },
    25: {
        "question": (
            "What effect does reducing the attention key size d_k have on "
            "Transformer model quality, and what does that suggest?"
        ),
        "gold_answer": (
            "It hurts model quality, which suggests that determining "
            "compatibility is not easy and that a more sophisticated "
            "compatibility function than dot product may be beneficial."
        ),
        "question_type": "analytical",
    },
}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} - run from the repo root")

    kept: list[dict] = []
    revised = dropped = 0

    with io.open(SRC, encoding="utf-8") as f:
        for idx, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            sample = json.loads(line)

            if idx in DROP:
                dropped += 1
                continue

            patch = REVISIONS.get(idx)
            if patch:
                ctx_filter = patch.pop("context_filter", None)
                if ctx_filter:
                    filtered = [c for c in sample["gold_contexts"] if ctx_filter in c]
                    # Only narrow if the filter actually matched something —
                    # never leave a sample with zero gold contexts.
                    if filtered:
                        sample["gold_contexts"] = filtered
                sample.update(patch)
                sample.setdefault("metadata", {})["revised_from_v1"] = idx
                revised += 1

            kept.append(sample)

    with io.open(DST, "w", encoding="utf-8") as f:
        for s in kept:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"wrote {DST}: {len(kept)} samples ({revised} revised, {dropped} dropped)")


if __name__ == "__main__":
    main()
