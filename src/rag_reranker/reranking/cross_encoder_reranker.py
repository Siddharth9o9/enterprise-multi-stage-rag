"""
Implements cross-encoder reranking using joint query-document scoring.
"""
from __future__ import annotations
import httpx
from rag_reranker.reranking.base import BaseReranker
from rag_reranker.ingestion.loader import Document
from rag_reranker.config import settings

class LocalCrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranking using a local sentence-transformers model.
    """
    def __init__(self,model_name: str | None = None):
        from sentence_transformers import CrossEncoder
        
        self.model_name = model_name or settings.local_cross_encoder_model
        self.model = CrossEncoder(
            self.model_name, device = "cuda" if self._is_cuda_available() else "cpu",)
        
    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        
    async def rerank(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        """
        Score every (query,document) pair jointly, return top_k.
        """
        if not documents:
            return []
        top_k = top_k or settings.rerank_top_k
        
        # Query Document pairs
        pairs = [(query, f"{doc.title}. {doc.content}") for doc in documents]
        
        # predict() runs all pairs through the cross-encoder, returns numpy array of raw logit scores (not probabilities)
        raw_scores = self.model.predict(pairs, show_progress_bar = True)
        
        # Attach score to document copies
        scored_documents = []
        for doc, score in zip(documents,raw_scores):
            scored_doc = doc.model_copy()
            scored_doc.score = float(score)
            scored_documents.append(scored_doc)
            
        # Sort by cross-encoder scores descending
        scored_documents.sort(key=lambda d: d.score, reverse=True)
        
        return scored_documents[:top_k]
    
class NvidiaCrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder reranking via NVIDIA NIM's /v1/ ranking endpoint.
    
    NVIDIA's reranker NIM uses a completely different
    endpoint shape:

    REQUEST:
        POST /v1/ranking
        {
            "model": "nvidia/nv-rerankqa-mistral-4b-v3",
            "query": {"text": "user query here"},
            "passages": [
                {"text": "document 1 text"},
                {"text": "document 2 text"}
            ]
        }

    RESPONSE:
        {
            "rankings": [
                {"index": 2, "logit": 4.81},
                {"index": 0, "logit": 1.02},
                {"index": 1, "logit": -0.34}
            ]
        }

    The response gives our INDICES into our passages array, sorted
    by relevance, with logit scores. We use the indices to reorder
    our original documents list.
    
    Cross-encoder raw scores: roughly -10 to +10
    
    These are raw logits from the final linear layer, not probabilities. A higher number means more relevant. 
    The absolute values are not meaningful — only the relative ordering matters.

    httpx is the async HTTP client that handles raw POST requests.
    """
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.nvidia_reranker_model
        self.base_url = settings.nvidia_base_url
        self.api_key = settings.nvidia_api_key
        
    async def rerank(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        if not documents:
            return []
        
        top_k = top_k or settings.rerank_top_k
        
        payload = {
            "model": self.model_name,
            "query": {"text": query},
            "passage": [{"text":f"{doc.title}. {doc.content}"} for doc in documents],
        }
    
        headers = {
            "Authorization":f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # async with: opens the HTTP connection, makes the request,
        # closes the connection — all within this context manager.
        # timeout=30.0: fail fast if NVIDIA API is slow/down rather
        # than hanging indefinitely and blocking the pipeline.
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/ranking",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
        
        # NVIDIA returns rankings as indices into our passages list
        # We use those indices to reorder our original Document objects    
        reranked: list[Document] = []
        
        for item in result.get("rankings",[]):
            original_index = item["index"]
            logit_score = item["logit"]
            
            reranked_doc = documents[original_index].model_copy()
            reranked_doc.score = float(logit_score)
            reranked.append(reranked_doc)
        
        # NVIDIA returns them pre-sorted by relevance descending
        # We sort again defensively in case API behavior changes    
        reranked.sort(key=lambda d:d.score, reverse=True)
        
        return reranked[:top_k]
        
    
