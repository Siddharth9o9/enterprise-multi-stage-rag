"""
Abstract base class defining the reranker interface.

Pattern: Strategy Pattern

- Every reranker (CrossEncoder, LLM pointwise, LLM listwise) implements this single interface.
- pipeline.py() calls rerank() without knowing which concrete implementation it received.

In case if we have to change the reranker in future, out interface would be same and no changes required in the main code.
Only new file for the new reranker needs to be created within the same interface that inherit from BaseReranker.
Every reranker will have same input output structure.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from rag_reranker.ingestion.loader import Document

class BaseReranker(ABC):
    """
    Every concrete reranker must implement exactly one method: rerank().
    """
    @abstractmethod
    async def rerank(self, query: str, document: list[Document], top_k: int = 5,) -> list[Document]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: the original user query string
            documents: candidate documents from RRF fusion, already sorted by RRF score.
            top_k: how many documents to return after reranking
            
        Returns:
            documents re-sorted by this reranker's relevance judgement, truncated to top_k, each with .score
            updated (overwrite the RRF score)
        """
        @property
        def name(self) -> str:
            """
            Human-readable name for logging and evaluation output.
            """
            return self.__class__.__name__
        
            
        
    