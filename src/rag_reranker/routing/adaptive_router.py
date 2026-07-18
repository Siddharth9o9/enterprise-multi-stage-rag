"""
adaptive_router.py
────────────────────
Classifies queries by complexity before retrieval runs, to avoid paying the full pipeline cost for queries that do not need it.

CONCEPT: Adaptive RAG / Self-RAG (from theory) 
Not every query needs the same treatment. A router decides the retrieval strategy per query, 
rather than always running one fixed expensive pipeline regardless of what was actually asked.
"""

from __future__ import annotations
from enum import Enum
import re
from rag_reranker.config import settings


class RetrievalStrategy(str, Enum):
    """
    why str, Enum (inheriting from BOTH):
    This makes RetrievalStrategy.NO_RETRIEVAL behave like the string "no_retrieval" when compared or serialized 
    (e.g. into JSON for the API layer in Step 18), while still giving us the type safety and autocomplete benefits of an Enum
    in code. Without str inheritance, you would need explicit .value access everywhere you wanted the string form — 
    for example when returning this in an API response.
    """
    NO_RETRIEVAL = "no_retrieval"
    SIMPLE_RETRIEVAL = "simple_retrieval"
    COMPLEX_RETRIEVAL = "complex_retrieval"


# ── NO_RETRIEVAL patterns ────────────────────────────────────────────────
# Regex patterns for detecting queries that likely need NO retrieval — general conversations & greetings the LLM already has from pretraining,
# unrelated to our specific knowledge base content.
_NO_RETRIEVAL_PATTERNS = [
    r"^(hi|hello|hey|greetings)\b",
    r"^thanks?( you)?\b",
    r"^(good ?bye|bye|see you)\b",
    r"^what can you (do|help( me)?( with)?)\b",
    r"^who are you\b",
    r"^what are you\b",
    r"^(can you )?help( me)?$",
]

# ── COMPLEX_RETRIEVAL signals ────────────────────────────────────────────
# Comparison, causal reasoning, or synthesis language — these genuinely indicate a query needs careful, 
# multi-document reasoning regardless of domain, so this list remains valid for an arXiv research corpus without modification.
_COMPLEX_SIGNALS = [
    r"\bcompare\b",
    r"\bdifference between\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bwhy\b",
    r"\btradeoffs?\b",
    r"\bpros and cons\b",
    r"\brelationship between\b",
    r"\bhow does .+ (affect|impact|influence)\b",
    r"\bsynthesi[sz]e\b",
    r"\bacross (multiple|several|different) papers\b",
]


class HeuristicQueryClassifier:
    """
    Fast, free, rule-based classifier. Zero API calls, near-instant.

    WHY THIS IS THE DEFAULT, DESPITE LLMQueryClassifier BEING MORE ACCURATE:
    Classification must be cheap — that is the entire point of adaptive
    routing. Paying an API call just to DECIDE whether to retrieve
    defeats the purpose for the majority of queries that are neither
    pure conversational plumbing nor obviously complex comparisons.
    Use this as the zero-cost default; upgrade to LLMQueryClassifier
    (below) when classification accuracy matters more than the small
    latency/cost of one extra fast API call per query.
    """

    def classify(self, query: str) -> RetrievalStrategy:
        normalized = query.lower().strip()
        word_count = len(normalized.split())

        # Check NO_RETRIEVAL first — conversational plumbing is the strongest, most unambiguous signal when it matches at all
        for pattern in _NO_RETRIEVAL_PATTERNS:
            if re.search(pattern, normalized):
                return RetrievalStrategy.NO_RETRIEVAL

        # Check COMPLEX signals next — comparison/causal/synthesis language
        for pattern in _COMPLEX_SIGNALS:
            if re.search(pattern, normalized):
                return RetrievalStrategy.COMPLEX_RETRIEVAL

        # WHY WORD COUNT AS THE FINAL TIEBREAKER:
        # Short queries without explicit complexity signals are usually direct factual lookups ("What is BM25?" = 4 words). 
        # Longer queries without an explicit trigger word still often bundle multiple sub-questions or nuanced framing that 
        # benefits from the more careful COMPLEX pipeline.
        if word_count <= settings.simple_query_word_limit:
            return RetrievalStrategy.SIMPLE_RETRIEVAL

        return RetrievalStrategy.COMPLEX_RETRIEVAL


class LLMQueryClassifier:
    """
    RECOMMENDED default for production use on a research-paper corpus.

    WHY THIS IS RECOMMENDED OVER PURE HEURISTICS FOR SIMPLE VS COMPLEX:
    Hardcoded regex patterns for "is this ML/NLP concept well-known
    enough to skip retrieval" cannot keep pace with a fast-moving
    research field — "what is a state space model" was cutting-edge
    in 2023 and increasingly foundational by 2025. An LLM classifier
    adapts naturally without needing its pattern list rewritten every
    time the field advances, at the cost of one small, fast API call
    (using the 8B model, not the 70B generator) per query.

    We still keep the NO_RETRIEVAL decision restricted to conversational
    plumbing in the prompt below — this classifier's real value is in
    distinguishing SIMPLE from COMPLEX among genuine knowledge questions,
    which the regex word-count heuristic can only approximate.
    """

    CLASSIFY_PROMPT = """Classify this query into exactly one category, for a \
system that answers questions using a corpus of arXiv computer science \
research papers (primarily NLP/ML topics):

- "no_retrieval": the query is conversational plumbing only (a greeting, \
thanks, or a question about the assistant itself) — NOT a knowledge question
- "simple_retrieval": a direct factual question likely answerable by looking \
up ONE relevant paper or concept
- "complex_retrieval": requires comparing, synthesizing, or reasoning across \
MULTIPLE papers or concepts, or asks "why"/"how" something works mechanistically

Query: {query}

Respond with ONLY one of these exact words: no_retrieval, simple_retrieval, complex_retrieval"""

    def __init__(self, model: str | None = None):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
        )
        self.model = model or settings.reranker_llm_model

    async def classify(self, query: str) -> RetrievalStrategy:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.CLASSIFY_PROMPT.format(query=query)}
                ],
                temperature=0.0,
                max_tokens=20,
            )
            label = response.choices[0].message.content.strip().lower()

            for strategy in RetrievalStrategy:
                if strategy.value in label:
                    return strategy

        except Exception as e:
            print(f"  [warn] LLM classification failed: {e}")

        # SAFE DEFAULT: if classification fails for any reason, fall back
        # to SIMPLE_RETRIEVAL — never silently skip retrieval on failure
        # (which could produce an ungrounded answer), and never default
        # to the most expensive COMPLEX path either.
        return RetrievalStrategy.SIMPLE_RETRIEVAL