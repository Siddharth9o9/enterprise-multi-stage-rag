"""
Reciprocal Rank Fusion - combines multiple ranked lists into one.

Inputs: Multiple lists of Documents, each sorted by their own retriever's score (best first)

Output: Single fused list sorted by RRF score (best first)

Formula: RRF_score(doc) = summation of [ 1 / ( k + rank)]
"""

from __future__ import annotations
from rag_reranker.ingestion.loader import Document
from rag_reranker.config import settings

def _document_key(doc: Document) -> tuple[str, str]:
    """
    Identity key for recognizing the same document across retrievers.
    
    Using (title, content) and not id:
    Every call to dense_retriever and sparse_retriever uses model_copy() which generates a brand new UUID for every Document copy.
    Two copies of the same document have different ids but identical (title,content). We match on what is stable, not what is ephemeral.
     
    """
    return (doc.title, doc.content)

def reciprocal_rank_fusion(ranked_lists: list[list[Document]], k: int | None = None,) -> list[Document]:
    """
    Fuse multiple ranked lists into one superior ranking.
    
    Args:
        ranked_lists: e.g. [dense_results, sparse_results] sorted best-to-worse by its own retriever
        
        k: RRF damping constant. Defaults to settings.rrf_k_constants which is 60.
        
    Returns:
        Single deduplicated list sorted by descending RRF score.
        Each Document's .score field contains its RRF score.
        Each Document's .retrieved_by field lists all retirvers
        that found it(["dense"],["sparse"], or["dense","sparse"]).
        
    """
    k = k if k is not None else settings.rrf_k_constant
    
    # Creating three parallel dict keyed by document identity.
    # Clean separation and easy to reason and debug
    
    rrf_scores: dict[tuple[str,str], float] = {}
    doc_registry: dict[tuple[str,str], Document] = {}
    retrieved_by: dict[tuple[str,str],set[str]] = {}
    
    for ranked_list in ranked_lists:
        for rank_zero_indexed, doc in enumerate(ranked_list):
            # RRF uses 1-indexed rank that is first document = rank 1
            rank = rank_zero_indexed + 1
            contribution = 1.0 / (k + rank)
            
            key = _document_key(doc)
            
            # Accumalate RRF score
            rrf_scores[key] = rrf_scores.get(key, 0.0) + contribution
            
            # Register the document itself
            
            if key not in doc_registry:
                doc_registry[key] = doc
                retrieved_by[key] = set()
                
            # Track which retriever found this document
            retrieved_by[key].update(doc.retrieved_by)
            
        # Build the final fused list
    fused_documents: list[Document] = []
        
    for key, rrf_score in rrf_scores.items():
        # Create a fresh copy so we do not mutate anything in the registry
        fused_doc = doc_registry[key].model_copy()
        fused_doc.score = round(rrf_score, 6)
        
        # Sort retrieved_by for deterministic output
        # ["dense", "sparse"] not ["sparse", "dense"] depending on which retriever happened to process this doc first
        fused_doc.retrieved_by = sorted(retrieved_by[key])
        
        fused_documents.append(fused_doc)
        
    # Sort by RRF score descending. Highest fusion score = best rank
    fused_documents.sort(
    key=lambda d: (d.score, len(d.retrieved_by)),
    reverse=True,
    )
    
    return fused_documents
    
    
    
