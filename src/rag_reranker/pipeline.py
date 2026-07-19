"""
Orchestrates every module into one end-to-end RAG pipeline.

FLOW:
query
if:
  |> [Adaptive router] -- No Retrieval --> direct generator call
  
else:
  |
  | Simple/Complex
  |

  |> [Dense Retriever]
                           ----- [RRF fusion] -> [Reranker] -> [Compression] -> [Generator]
  |>  [Sparse Retriever]
"""

from __future__ import annotations
from dataclasses import dataclass, field

from rag_reranker.config import settings
from rag_reranker.ingestion.loader import Document, load_documents
from rag_reranker.ingestion.chunker import chunk_documents
from rag_reranker.retrieval.dense_retriever import DenseRetriever
from rag_reranker.retrieval.sparse_retriever import SparseRetriever
from rag_reranker.retrieval.fusion import reciprocal_rank_fusion
from rag_reranker.reranking.base import BaseReranker
from rag_reranker.reranking.cross_encoder_reranker import LocalCrossEncoderReranker
from rag_reranker.reranking.llm_reranker import PointwiseLLMReranker, ListwiseLLMReranker
from rag_reranker.reranking.compression import EmbeddingsFilterCompressor
from rag_reranker.routing.adaptive_router import HeuristicQueryClassifier, RetrievalStrategy
from rag_reranker.generation.generator import AnswerGenerator

@dataclass
class PipelineResult:
    """
    Full decision trace for one query — not just the final answer.

    result.reranked_docs tells exactly which documents the LLM saw when generating the answer — critical for debugging a wrong or hallucinated answer. result.strategy_used tells you which router path fired, letting us verify routing decisions look sane across many test queries.  The API and evaluation module both consumes this.
    """
    query: str
    strategy_used: str
    initial_docs: list[Document] = field(default_factory=list)
    fused_docs: list[Document] = field(default_factory=list)
    reranked_docs: list[Document] = field(default_factory=list)
    answer: str = ""
    
class RAGPipeline:
    def __init__(self, reranker_strategy: str = "cross_encoder"):
        """
        reranker_strategy: one of "cross_encoder" | "llm_pointwise" | "llm_listwise"

        WHY THIS IS A STRING, NOT A CLASS PASSED DIRECTLY:
        Strings are what come from config files, .env variables, and API request bodies. _build_reranker() is the single place that translates a string into a concrete BaseReranker instance — the Strategy Pattern payoff.
        """
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.compressor = EmbeddingsFilterCompressor()
        self.classifier = HeuristicQueryClassifier()
        self.generator = AnswerGenerator()
        self.reranker: BaseReranker = self._build_reranker(reranker_strategy)
        if isinstance(self.reranker, LocalCrossEncoderReranker):
            self._cheap_reranker: BaseReranker = self.reranker
        else:
            self._cheap_reranker = LocalCrossEncoderReranker()
        self._indexed = False

    def _build_reranker(self, strategy: str) -> BaseReranker:
        if strategy == "llm_pointwise":
            return PointwiseLLMReranker()
        if strategy == "llm_listwise":
            return ListwiseLLMReranker()
        # Default: local cross-encoder — fast, free, no API call needed
        return LocalCrossEncoderReranker()

    def index(self) -> None:
        """
        Builds both retrieval indexes from the knowledge base.

        WHY THIS IS A SEPARATE METHOD FROM __init__:
        Index-building involves loading the embedding model onto GPU
        and encoding every document — this takes real time (seconds,
        not milliseconds). Separating it from __init__ means you can
        construct a RAGPipeline instance cheaply and control exactly
        when the expensive indexing work happens — critical for the
        FastAPI lifespan pattern in Step 18, where we want indexing
        to happen ONCE at server startup, not on every request.
        """
        raw_docs = load_documents()
        chunks = chunk_documents(
            raw_docs,
            strategy="recursive",
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.dense_retriever.build_index(chunks)
        self.sparse_retriever.build_index(chunks)
        self._indexed = True

    async def run(self, query: str) -> PipelineResult:
        """
        The full end-to-end flow for one query. This is the ONE
        public method external code calls (pipeline.run(query))
        regardless of which internal path the router chooses.
        """
        if not self._indexed:
            # Safety net: auto-index on first call if index() was
            # never explicitly called. In production (Step 18) we
            # call index() explicitly at startup, but this prevents
            # a confusing crash if someone forgets that step during
            # development or testing.
            self.index()

        strategy = self.classifier.classify(query)
        result = PipelineResult(query=query, strategy_used=strategy.value)

        # ── NO_RETRIEVAL PATH ────────────────────────────────────────
        if strategy == RetrievalStrategy.NO_RETRIEVAL:
            # Skip retrieval entirely — conversational plumbing only,
            # per our domain-specific router design from Step 15.
            result.answer = await self.generator.generate(query, [])
            return result

        # ── RETRIEVAL (shared by SIMPLE and COMPLEX) ─────────────────
        dense_results = self.dense_retriever.retrieve(
            query, top_k=settings.dense_top_k
        )
        sparse_results = self.sparse_retriever.retrieve(
            query, top_k=settings.sparse_top_k
        )
        result.initial_docs = dense_results

        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        result.fused_docs = fused

        # WHY WE CAP CANDIDATES AT 15 BEFORE RERANKING:
        # RRF already ranked the full candidate pool. Asking an
        # expensive reranker (especially LLM-based) to re-score
        # documents RRF already placed far down the list wastes
        # API calls / GPU time for negligible benefit — the correct
        # answer is virtually never at RRF rank 16+ if retrieval
        # is working reasonably well.
        candidates = fused[:15]

        # ── SIMPLE_RETRIEVAL PATH ─────────────────────────────────────
        if strategy == RetrievalStrategy.SIMPLE_RETRIEVAL:
            reranked = await self._cheap_reranker.rerank(
                query, candidates, top_k=settings.rerank_top_k
            )
            result.reranked_docs = reranked
            result.answer = await self.generator.generate(query, reranked)
            return result

        # ── COMPLEX_RETRIEVAL PATH ────────────────────────────────────
        # Uses whichever reranker this pipeline instance was configured
        # with (cross_encoder / llm_pointwise / llm_listwise), since
        # COMPLEX queries are exactly where a more expensive, more
        # precise reranker earns its cost.
        reranked = await self.reranker.rerank(
            query, candidates, top_k=settings.rerank_top_k
        )
        result.reranked_docs = reranked

        # Compression runs ONLY for COMPLEX queries — this is where
        # multiple nuanced sources are being synthesized, and trimming
        # each document to its most relevant sentences meaningfully
        # helps the generator avoid the Lost-in-the-Middle problem.
        compressed = self.compressor.compress_all(query, reranked)

        result.answer = await self.generator.generate(query, compressed)
        return result



