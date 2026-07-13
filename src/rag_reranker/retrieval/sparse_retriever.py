from __future__ import annotations
import re
from rag_reranker.ingestion.loader import Document
from rag_reranker.config import settings

def _tokenize(text: str) -> list[str]:
    """
    Convert text to a list of lowercase tokens for BM25
    Using this tokenizer for our corpus (clean acedemic english), lowercase + punctuation removal is sufficient 
    and keeps the module dependency free.
    
    BM25 is case-sensitive by default.
    
    Without punctuation, similar word should match the same tokens
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]"," ",text)
    return [token for token in text.split( ) if token]

class SparseRetriever:
    
    def __init__(self):
        self.bm25 = None
        self.documents: list[Document] = []
        
    def build_index(self, documents: list[Document]) -> None:
        """
        Offline phase: tokenize all documents and build BM25 index.
        
        BM25Okapi pre-computes:
        - Inverted index
        - IDF scores for every term in the corpus
        - Document lengths for normalization
        
        rank_bm25 offers three variants:
            - BM25Okapi -> standard BM25 - the most common
            - BM25L -> adjusts for very long documents
            - BM25Plus -> prevents zero scores for short documents
            
        Selecting BM25Okapi as it is the reseach & production standard.
        The others are marginal improvements for edge cases.
        """
        from rank_bm25 import BM25Okapi
        
        self.documents = documents
        
        tokenized_corpus = [
            _tokenize(f"{doc.title}. {doc.content}")
            for doc in documents
        ]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built: {len(documents)} documents")
        
    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """
        Online phase: score all documents against query, return top_k.
        
        BM25: scores all documents every time
            - 0(n) time: cannot avoid this with an inverted index
            - but each individual scoring is cheap (integer ops)
            - For our corpus size (14-50000), this is still fast
            
        For millions of documents move BM25 to elasticsearch which uses optimized inverted index structures to avoid scoring every documents.
        
        """
        if self.bm25 is None:
            raise RuntimeError(
                "build_index must be called before retrieve()."
            )
            
        top_k = top_k or settings.sparse_top_k
        
        query_tokens = _tokenize(query)
        
        # get_scores() returns one BM25 score per document in corpus
        # Higher score = more relevant to this query
        # Score of 0.0 = query terms not found in document at all
        
        scores = self.bm25.get_scores(query_tokens)
        
        # Pair each score with its document index, sort descending
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        
        results: list[Document] = []
        for idx, score in ranked:
            # Skipping document with 0 score 
            
            if score <= 0:
                continue
            
            doc = self.documents[idx].model_copy()
            doc.score = float(score)
            doc.retrieved_by = ["sparse"]
            results.append(doc)
            
        return results
        
        
        
